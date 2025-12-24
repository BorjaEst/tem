"""Loss functions for the Tolman-Eichenbaum Machine (TEM).

This module implements the Evidence Lower Bound (ELBO) components used to train TEM.
The total loss decomposes the variational objective into three interpretable components
that correspond to distinct cognitive functions:

1. Sensory Reconstruction Loss (L_x): Measures the model's ability to predict
   sensory observations from hippocampal place cells. This is the core generative
   objective: can the model reconstruct what it sees given where it thinks it is?

2. Abstract Location Loss (L_g): Enforces consistency between path integration
   (dead reckoning from movement) and landmark-based localization. This KL divergence
   ensures the MEC grid cells learn accurate transition dynamics that don't require
   constant sensory correction.

3. Grounded Location Loss (L_p): Ensures hippocampal place cells (conjunctive
   representations) are consistent with memory retrievals from both sensory input
   and abstract location. This constrains the hippocampus to form valid memory indices.

These losses collectively enforce consistency between the "What" (LEC sensory pathway)
and "Where" (MEC spatial pathway), enabling the model to generalize spatial knowledge
across different sensory contexts.

References:
    Whittington et al. (2020). The Tolman-Eichenbaum Machine: Unifying Space and
    Relational Memory through Generalization in the Hippocampal Formation.
    Cell, 183(5), 1249-1263.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_tem.config import TrainingConfig
from torch_tem.types import AbstractLocation, GroundedLocation, SensoryPrediction, Transition


@dataclass
class LossOutput:
    """Container for TEM loss components.

    Provides structured access to all loss terms for logging, monitoring, and
    debugging. The total loss is used for backpropagation, while individual
    components can be tracked to diagnose training dynamics.

    This class supports accumulation via __add__ and averaging via __truediv__
    operators, enabling concise loss aggregation in training loops:
        accumulated = loss1 + loss2 + loss3
        averaged = accumulated / 3

    Attributes:
        total: Weighted sum of all losses (for backprop).
        lx: Sensory reconstruction loss. Monitors how well the model predicts
            observations from place cells.
        lg: Abstract location loss (KL divergence). Monitors consistency between
            path integration and landmark-based localization.
        lp: Grounded location loss. Monitors consistency between inferred and
            retrieved hippocampal representations.
        l_reg_g: Abstract location regularization loss (L2 penalty). Prevents
            unbounded growth of grid cell activations.
        l_reg_p: Grounded location regularization loss (L1 penalty). Enforces
            sparsity in hippocampal place cell activations.
    """

    total: Tensor
    lx: Tensor
    lg: Tensor
    lp: Tensor
    l_reg_g: Optional[Tensor] = None
    l_reg_p: Optional[Tensor] = None

    @staticmethod
    def zero() -> "LossOutput":
        """Create a zero-initialized LossOutput for accumulation.

        Returns:
            LossOutput with all components set to 0.0 tensors.

        Example:
            >>> accumulated = LossOutput.zero()
            >>> for t in range(rollout_length):
            ...     loss_output = model.loss(x[t], state)
            ...     accumulated = accumulated + loss_output
            >>> averaged = accumulated / rollout_length
        """
        return LossOutput(
            total=torch.tensor(0.0),
            lx=torch.tensor(0.0),
            lg=torch.tensor(0.0),
            lp=torch.tensor(0.0),
            l_reg_g=torch.tensor(0.0),
            l_reg_p=torch.tensor(0.0),
        )

    def __add__(self, other: "LossOutput") -> "LossOutput":
        """Add two LossOutput instances component-wise.

        Handles optional regularization terms gracefully (treats None as 0).

        Args:
            other: Another LossOutput instance to add.

        Returns:
            New LossOutput with summed components.

        Example:
            >>> loss_sum = loss1 + loss2 + loss3
        """
        # Add optional regularization terms (treat None as 0)
        l_reg_g = None
        if self.l_reg_g is not None or other.l_reg_g is not None:
            self_reg_g = self.l_reg_g if self.l_reg_g is not None else torch.tensor(0.0)
            other_reg_g = other.l_reg_g if other.l_reg_g is not None else torch.tensor(0.0)
            l_reg_g = self_reg_g + other_reg_g

        l_reg_p = None
        if self.l_reg_p is not None or other.l_reg_p is not None:
            self_reg_p = self.l_reg_p if self.l_reg_p is not None else torch.tensor(0.0)
            other_reg_p = other.l_reg_p if other.l_reg_p is not None else torch.tensor(0.0)
            l_reg_p = self_reg_p + other_reg_p

        return LossOutput(
            total=self.total + other.total,
            lx=self.lx + other.lx,
            lg=self.lg + other.lg,
            lp=self.lp + other.lp,
            l_reg_g=l_reg_g,
            l_reg_p=l_reg_p,
        )

    def __truediv__(self, divisor: int | float) -> "LossOutput":
        """Divide all loss components by a scalar.

        Used to compute average losses over multiple timesteps.

        Args:
            divisor: Number to divide by (typically number of timesteps).

        Returns:
            New LossOutput with divided components.

        Raises:
            ValueError: If divisor is zero.

        Example:
            >>> avg_loss = accumulated_loss / n_timesteps
        """
        if divisor == 0:
            raise ValueError("Cannot divide LossOutput by zero")

        return LossOutput(
            total=self.total / divisor,
            lx=self.lx / divisor,
            lg=self.lg / divisor,
            lp=self.lp / divisor,
            l_reg_g=self.l_reg_g / divisor if self.l_reg_g is not None else None,
            l_reg_p=self.l_reg_p / divisor if self.l_reg_p is not None else None,
        )

    def as_dict(self) -> dict:
        """Convert loss components to dictionary for logging.

        Returns:
            Dictionary mapping component names to scalar values.
        """
        return {
            "lx": self.lx.item(),
            "lg": self.lg.item(),
            "lp": self.lp.item(),
            "l_reg_g": self.l_reg_g.item() if self.l_reg_g is not None else 0.0,
            "l_reg_p": self.l_reg_p.item() if self.l_reg_p is not None else 0.0,
        }


class SensoryReconstructionLoss(nn.Module):
    """Computes the reconstruction loss for sensory observations (L_x).

    This is the fundamental generative objective of TEM. It measures how well
    the model can predict sensory observations given its belief about location
    (encoded in hippocampal place cells).

    Mathematical Formulation:
        L_x = - E_q [ ln p(x | p) ]

    where:
        - x: True sensory observation (one-hot encoded categorical)
        - p: Hippocampal place cell activation (grounded location)
        - q: Variational posterior distribution

    In practice, this is computed as categorical cross-entropy between the
    predicted observation distribution (decoded from p) and the true observation.

    The loss can be computed from multiple pathways:
        1. p_x → x: Decode from sensory-retrieved place cells
        2. p_g → x: Decode from abstract-retrieved place cells
        3. p → x: Decode from inferred conjunctive place cells

    This "teacher forcing" between pathways ensures all representations learn
    to support accurate sensory prediction.
    """

    def __init__(self, reduction: Literal["sum", "mean"] = "sum"):
        """Initialize SensoryReconstructionLoss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'sum': Sum the loss over the batch.
                'mean': Take the mean of the loss over the batch.
                Default: 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, prediction: SensoryPrediction, target: Tensor) -> Tensor:
        """Compute L_x using categorical cross-entropy.

        Args:
            prediction: Predicted sensory observation containing:
                - logits: List of unnormalized log-probabilities (multi-scale).
                - values: List of softmax-normalized probabilities.
            target: Ground truth observation. Can be:
                - Class indices: shape (B,) or (B, 1)
                - One-hot encoded: shape (B, C) where C is number of classes

        Returns:
            Scalar loss value.

        Note:
            Uses the reduction method specified in __init__.
        """
        # Extract logits from the highest frequency scale (index 0)
        # Multi-scale predictions allow hierarchical sensory processing,
        # but for the main reconstruction loss we use the finest resolution
        logits = prediction.logits[0]

        # Handle different target formats flexibly
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            # Class indices: (B,) or (B, 1)
            # Convert to 1D long tensor for cross_entropy
            loss = F.cross_entropy(logits, target.view(-1).long(), reduction=self.reduction)
        elif target.dim() == 2 and target.shape[1] == logits.shape[1]:
            # One-hot or soft targets: (B, C)
            # Cross-entropy handles both hard (one-hot) and soft (probability) targets
            loss = F.cross_entropy(logits, target, reduction=self.reduction)
        else:
            raise ValueError(f"Shape mismatch: logits {logits.shape}, target {target.shape}")

        return loss


class AbstractLocationLoss(nn.Module):
    """Computes the consistency loss for abstract locations (L_g).

    This loss enforces that the MEC grid cells learn accurate transition dynamics.
    It penalizes discrepancies between two sources of spatial information:

    1. Path Integration (Prior): Where you should be based on movement alone
    2. Landmark Correction (Posterior): Where you are after sensory correction

    Mathematical Formulation:
        L_g = D_KL( q(g | x, a) || p(g | a) )

    where:
        - g: Abstract location (MEC grid cell activations)
        - x: Sensory observation
        - a: Action taken
        - q(g | x, a): Posterior after incorporating sensory evidence
        - p(g | a): Prior from path integration alone

    Implementation:
        We compute an analytical KL divergence assuming:
        - Prior p(g|a) is Gaussian with learned mean μ_prior and std σ_prior
        - Posterior q(g|x,a) is approximated as a delta function at μ_post

        This gives:
        KL ≈ 0.5 * ((μ_post - μ_prior) / σ_prior)² + ln(σ_prior)

        The first term is the Mahalanobis distance (squared error normalized by
        uncertainty), and the second is the entropy of the prior.

    Intuition:
        If this loss is high, the grid cells are failing to predict future location
        from movement, requiring constant sensory correction. Training reduces this
        by learning accurate transition models.
    """

    def __init__(self, reduction: Literal["sum", "mean"] = "sum"):
        """Initialize AbstractLocationLoss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'sum': Sum the loss over the batch.
                'mean': Take the mean of the loss over the batch.
                Default: 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, g: AbstractLocation, g_gen: Transition) -> Tensor:
        """Compute L_g using analytical KL divergence.

        Args:
            g: Posterior abstract location (inferred from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Prior abstract location (from path integration alone).
                Contains:
                - mean: List of predicted location means
                - uncertainty: List of predicted location standard deviations

        Returns:
            Scalar loss value.

        Note:
            Uses the reduction method specified in __init__.
        """
        total_kl = torch.tensor(0.0, device=g[0].device)

        # Iterate over frequency modules (typically 4-6 modules at different spatial scales)
        for i, (mu_post, mu_prior, sigma_prior) in enumerate(zip(g, g_gen.mean, g_gen.uncertainty)):
            # Compute KL divergence between posterior (delta function at mu_post)
            # and prior (Gaussian with mean mu_prior and std sigma_prior)
            #
            # Mathematical derivation:
            #   KL(δ(g - μ_post) || N(μ_prior, σ_prior²))
            #   = -ln p(μ_post | μ_prior, σ_prior)
            #   = 0.5 * ((μ_post - μ_prior) / σ_prior)² + ln(σ_prior) + const
            #
            # where the constant (0.5 * ln(2π)) is dropped as it doesn't affect gradients

            # Clamp sigma to avoid numerical instability from division by zero
            # Small sigma means high confidence in the path integration prediction
            sigma_prior = torch.clamp(sigma_prior, min=1e-6)

            # Mahalanobis distance: squared error weighted by inverse uncertainty
            # High uncertainty → low penalty for mismatch
            # Low uncertainty → high penalty for mismatch
            diff = mu_post - mu_prior
            mahalanobis = 0.5 * (diff / sigma_prior).pow(2)

            # Log-determinant term: entropy of the prior distribution
            # Encourages the model to maintain reasonable uncertainty estimates
            log_det = torch.log(sigma_prior)

            # Sum over all dimensions (batch, grid cells) for this frequency
            kl_f = (mahalanobis + log_det).sum() if self.reduction == "sum" else (mahalanobis + log_det).mean()
            total_kl += kl_f

        return total_kl


class GroundedLocationLoss(nn.Module):
    """Computes the consistency loss for grounded locations (L_p).

    This loss ensures that the hippocampal place cells form valid conjunctive
    representations that are consistent with memory retrievals from both pathways.

    The hippocampus binds "What" (sensory) and "Where" (spatial) into a unified
    representation. This loss constrains this binding to be consistent with:
    1. Memory retrieved via abstract location: p_g (from MEC pathway)
    2. Memory retrieved via sensory input: p_x (from LEC pathway)

    Mathematical Formulation:
        L_p = L_p_g + L_p_x

    where:
        L_p_g = || p - p_g ||²  (consistency with generative memory)
        L_p_x = || p - p_x ||²  (consistency with inference memory)

    Implementation:
        We use MSE (L2 loss) as a proxy for KL divergence. This is valid when
        the distributions are approximately Gaussian and have similar variance.

    Intuition:
        This loss prevents the hippocampus from "hallucinating" impossible
        conjunctions. If you see a landmark at location X, and grid cells say
        you're at location Y, the place cells must reconcile this information
        rather than inventing a third incompatible representation.
    """

    def __init__(self, reduction: Literal["sum", "mean"] = "sum"):
        """Initialize GroundedLocationLoss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'sum': Sum the loss over the batch.
                'mean': Take the mean of the loss over the batch.
                Default: 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, p: GroundedLocation, p_g: Optional[GroundedLocation] = None, p_x: Optional[GroundedLocation] = None) -> Tensor:
        """Compute L_p as MSE between inferred and retrieved place cells.

        Args:
            p: Inferred grounded location (conjunctive code from x ⊗ g).
                List of tensors, one per frequency. Shape: [(B, n_p_f1), (B, n_p_f2), ...]
            p_g: Retrieved grounded location (from memory via abstract location g).
                This represents "Where you think you are → What you expect to see".
                Optional; if None, this component is skipped.
            p_x: Retrieved grounded location (from memory via sensory input x).
                This represents "What you see → Where you might be".
                Optional; if None, this component is skipped.

        Returns:
            Scalar loss value.

        Note:
            Both p_g and p_x can be provided to enforce consistency across both
            inference and generative pathways ("teacher forcing").
        """
        total_loss = torch.tensor(0.0, device=p[0].device)

        # L_p_g: Consistency with generative memory (retrieved from g)
        # Ensures that inferring location from (x, g) gives similar place cells
        # as retrieving from memory using g alone
        if p_g is not None:
            for p_inf, p_ret in zip(p, p_g):
                total_loss += F.mse_loss(p_inf, p_ret, reduction=self.reduction)

        # L_p_x: Consistency with inference memory (retrieved from x)
        # Ensures that inferring location from (x, g) gives similar place cells
        # as retrieving from memory using x alone
        if p_x is not None:
            for p_inf, p_ret in zip(p, p_x):
                total_loss += F.mse_loss(p_inf, p_ret, reduction=self.reduction)

        return total_loss


class RegularizationLoss(nn.Module):
    """Computes regularization losses for latent representations.

    Regularization is critical for preventing degenerate solutions and encouraging
    biologically plausible representations.

    Components:
        1. L2 penalty on abstract locations (g): Prevents unbounded growth of grid
           cell activations, which could destabilize training. This encourages
           representations to remain in a bounded dynamic range.

        2. L1 penalty on grounded locations (p): Enforces sparsity in hippocampal
           place cells, matching biological observations where only a small fraction
           of place cells are active at any given location. Sparsity also improves
           memory capacity and generalization.

    Mathematical Formulation:
        L_reg_g = λ_g * Σ ||g_f||²₂  (L2 norm, summed over frequencies)
        L_reg_p = λ_p * Σ ||p_f||₁   (L1 norm, summed over frequencies)
    """

    def __init__(self, reduction: Literal["sum", "mean"] = "sum"):
        """Initialize RegularizationLoss.

        Args:
            reduction: Specifies the reduction to apply to the output:
                'sum': Sum the loss over the batch.
                'mean': Take the mean of the loss over the batch.
                Default: 'sum'.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, g: AbstractLocation, p: GroundedLocation) -> Tuple[Tensor, Tensor]:
        """Compute regularization penalties.

        Args:
            g: Abstract location (grid cells). List of tensors per frequency.
            p: Grounded location (place cells). List of tensors per frequency.

        Returns:
            Tuple of (l_reg_g, l_reg_p):
                - l_reg_g: L2 penalty on grid cells
                - l_reg_p: L1 penalty on place cells
        """
        # L2 regularization on grid cells: penalizes large activations
        # Σ_f Σ_b Σ_i g_{f,b,i}²
        if self.reduction == "sum":
            l_reg_g = sum(g_i.pow(2).sum() for g_i in g)
        else:
            l_reg_g = sum(g_i.pow(2).mean() for g_i in g)

        # L1 regularization on place cells: enforces sparsity
        # Σ_f Σ_b Σ_i |p_{f,b,i}|
        if self.reduction == "sum":
            l_reg_p = sum(p_i.abs().sum() for p_i in p)
        else:
            l_reg_p = sum(p_i.abs().mean() for p_i in p)

        return l_reg_g, l_reg_p


class TEMLoss(nn.Module):
    """Aggregates all TEM loss components into the final ELBO objective.

    This class combines the three main ELBO components (L_x, L_g, L_p) with
    optional regularization terms, using learnable or fixed weights to balance
    their contributions.

    The final objective is:
        L_total = loss_weights_x * L_x + loss_weights_g * L_g + loss_weights_p * L_p + loss_weights_reg_g * L_reg_g + loss_weights_reg_p * L_reg_p

    Weight Selection Guidelines:
        - loss_weights_x: Typically 1.0 (baseline)
        - loss_weights_g: Controls path integration accuracy (0.1 - 1.0)
        - loss_weights_p: Controls memory consistency (0.1 - 1.0)
        - loss_weights_reg_g: Prevents grid cell saturation (0.01 - 0.1)
        - loss_weights_reg_p: Enforces place cell sparsity (0.01 - 0.1)

    These weights may require tuning based on environment complexity and model size.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize TEM loss aggregator.

        Args:
            config: Configuration object containing loss weights.
        """
        super().__init__()
        self.config = config

    def forward(self, lx: Tensor, lp: Tensor, lg: Tensor, l_reg_g: Optional[Tensor] = None, l_reg_p: Optional[Tensor] = None) -> LossOutput:
        """Compute weighted sum of all loss components.

        Args:
            lx: Sensory reconstruction loss (from SensoryReconstructionLoss).
            lp: Grounded location consistency loss (from GroundedLocationLoss).
            lg: Abstract location KL divergence (from AbstractLocationLoss).
            l_reg_g: Optional grid cell regularization (from RegularizationLoss).
            l_reg_p: Optional place cell regularization (from RegularizationLoss).

        Returns:
            LossOutput containing:
                - total: Weighted sum for backpropagation
                - Individual components for logging and monitoring
        """
        # Compute weighted sum of main ELBO components
        total = self.config.loss_weights_x * lx + self.config.loss_weights_p * lp + self.config.loss_weights_g * lg

        # Add regularization terms if provided
        if l_reg_g is not None:
            total += self.config.loss_weights_reg_g * l_reg_g

        if l_reg_p is not None:
            total += self.config.loss_weights_reg_p * l_reg_p

        return LossOutput(total=total, lx=lx, lp=lp, lg=lg, l_reg_g=l_reg_g, l_reg_p=l_reg_p)
