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
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.types import AbstractLocation, GroundedLocation, SensoryPrediction, Transition


class LossConfig(BaseModel):
    """Loss weight configuration for TEM training.

    This configuration encapsulates the weights assigned to each loss component
    during training. These weights influence the relative importance of different
    objectives such as reconstruction accuracy, location grounding, and regularization.
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # ===================================================================================
    # LOSS WEIGHTS
    # ===================================================================================

    weights_x: float = Field(default=1.0, ge=0, description="Weight of reconstruction/prediction losses on x")
    weights_p: float = Field(default=1.0, ge=0, description="Weight of grounded location losses on p")
    weights_g: float = Field(default=1.0, ge=0, description="Weight of abstract location losses on g")
    weights_reg_g: float = Field(default=0.01, ge=0, description="Weight of regularisation loss on abstract location")
    weights_reg_p: float = Field(default=0.02, ge=0, description="Weight of regularisation loss on grounded location")


@dataclass
class LossX:
    """Sensory reconstruction section (L_x) with 3 pathways.

    Attributes:
        infer: p_inf → x (from inferred grounded location)
        retrieved: g_inf → p → x (from abstract location via memory)
        ancestral: g_prev → g → p → x (generative pathway)
    """

    infer: Tensor  # x_p in legacy
    retrieved: Tensor  # x_g in legacy
    ancestral: Tensor  # x_gen in legacy

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossX":
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(infer=z.clone(), retrieved=z.clone(), ancestral=z.clone())

    def __add__(self, other: "LossX") -> "LossX":
        return LossX(
            infer=self.infer + other.infer,
            retrieved=self.retrieved + other.retrieved,
            ancestral=self.ancestral + other.ancestral,
        )

    def __truediv__(self, divisor: int | float) -> "LossX":
        return LossX(
            infer=self.infer / divisor,
            retrieved=self.retrieved / divisor,
            ancestral=self.ancestral / divisor,
        )

    @property
    def total(self) -> Tensor:
        return self.infer + self.retrieved + self.ancestral


@dataclass
class LossP:
    """Grounded location section (L_p) with 2 pathways.

    Attributes:
        abstract: Consistency vs p retrieved via g (p_g in legacy)
        sensory: Consistency vs p retrieved via x (p_x in legacy)
    """

    abstract: Tensor
    sensory: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossP":
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(abstract=z.clone(), sensory=z.clone())

    def __add__(self, other: "LossP") -> "LossP":
        return LossP(
            abstract=self.abstract + other.abstract,
            sensory=self.sensory + other.sensory,
        )

    def __truediv__(self, divisor: int | float) -> "LossP":
        return LossP(
            abstract=self.abstract / divisor,
            sensory=self.sensory / divisor,
        )

    @property
    def total(self) -> Tensor:
        return self.abstract + self.sensory


@dataclass
class LossG:
    """Abstract location section (L_g).

    Attributes:
        transition: Surrogate ||g_inf - g_gen||² (or KL for probabilistic mode)
    """

    transition: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossG":
        return cls(transition=torch.zeros((), device=device, dtype=dtype))

    def __add__(self, other: "LossG") -> "LossG":
        return LossG(transition=self.transition + other.transition)

    def __truediv__(self, divisor: int | float) -> "LossG":
        return LossG(transition=self.transition / divisor)

    @property
    def total(self) -> Tensor:
        return self.transition


@dataclass
class LossReg:
    """Regularization section.

    Attributes:
        g_l2: L2 regularization on abstract location (reg_g in legacy)
        p_l1: L1 regularization on grounded location (reg_p in legacy)
    """

    g_l2: Tensor
    p_l1: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossReg":
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(g_l2=z.clone(), p_l1=z.clone())

    def __add__(self, other: "LossReg") -> "LossReg":
        return LossReg(
            g_l2=self.g_l2 + other.g_l2,
            p_l1=self.p_l1 + other.p_l1,
        )

    def __truediv__(self, divisor: int | float) -> "LossReg":
        return LossReg(
            g_l2=self.g_l2 / divisor,
            p_l1=self.p_l1 / divisor,
        )

    @property
    def total(self) -> Tensor:
        return self.g_l2 + self.p_l1


@dataclass
class LossOutput:
    """TEM loss container (composed sections + helpers for training/logging).

    Compositional structure following the TEM paper's ELBO decomposition:
    - L_x: Sensory reconstruction (3 pathways)
    - L_p: Grounded location consistency (2 pathways)
    - L_g: Abstract location transition
    - L_reg: Regularization (L2 on g, L1 on p)

    Attributes:
        x: Sensory reconstruction loss section
        p: Grounded location loss section
        g: Abstract location loss section
        reg: Regularization loss section
    """

    x: LossX
    p: LossP
    g: LossG
    reg: LossReg

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossOutput":
        return cls(
            x=LossX.zero(device=device, dtype=dtype),
            p=LossP.zero(device=device, dtype=dtype),
            g=LossG.zero(device=device, dtype=dtype),
            reg=LossReg.zero(device=device, dtype=dtype),
        )

    def __add__(self, other: "LossOutput") -> "LossOutput":
        return LossOutput(
            x=self.x + other.x,
            p=self.p + other.p,
            g=self.g + other.g,
            reg=self.reg + other.reg,
        )

    def __truediv__(self, divisor: int | float) -> "LossOutput":
        return LossOutput(
            x=self.x / divisor,
            p=self.p / divisor,
            g=self.g / divisor,
            reg=self.reg / divisor,
        )

    @property
    def total(self) -> Tensor:
        """Compute total loss (sum of all components)."""
        return self.x.total + self.p.total + self.g.total + self.reg.total


StepLoss = LossOutput  # Alias for clarity in training context
AccumLoss = LossOutput  # Alias for accumulated losses over rollouts


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
        L_total = weights_x * L_x + weights_g * L_g + weights_p * L_p + weights_reg_g * L_reg_g + weights_reg_p * L_reg_p

    Weight Selection Guidelines:
        - weights_x: Typically 1.0 (baseline)
        - weights_g: Controls path integration accuracy (0.1 - 1.0)
        - weights_p: Controls memory consistency (0.1 - 1.0)
        - weights_reg_g: Prevents grid cell saturation (0.01 - 0.1)
        - weights_reg_p: Enforces place cell sparsity (0.01 - 0.1)

    These weights may require tuning based on environment complexity and model size.
    """

    def __init__(self, config: Optional[LossConfig] = None):
        """Initialize TEM loss aggregator.

        Args:
            config: Configuration object containing loss weights. If None, uses default weights (all 1.0).
        """
        super().__init__()
        self.config = config or LossConfig()

    def forward(self, lx: Tensor, lp: Tensor, lg: Tensor, l_reg_g: Optional[Tensor] = None, l_reg_p: Optional[Tensor] = None) -> LossOutput:
        """Compute weighted sum of all loss components.

        Args:
            lx: Sensory reconstruction loss (from SensoryReconstructionLoss).
            lp: Grounded location consistency loss (from GroundedLocationLoss).
            lg: Abstract location KL divergence (from AbstractLocationLoss).
            l_reg_g: Optional grid cell regularization (from RegularizationLoss).
            l_reg_p: Optional place cell regularization (from RegularizationLoss).

        Returns:
            LossOutput containing weighted components.

        Note:
            This method expects ALREADY weighted inputs (from individual loss modules).
            The config weights are applied here as additional scaling.
        """
        # Note: For now, this legacy interface creates a simple LossOutput
        # In practice, the training loop uses _compute_step_losses directly
        # This is kept for backward compatibility with the fbdd1b1 branch

        # Create dummy component structure (this path is not used in current training)
        # If this is actually called, it means we need to refactor the caller
        raise NotImplementedError(
            "TEMLoss.forward() is deprecated. Use _compute_step_losses() in training.py instead. " "This class remains only for backward compatibility documentation."
        )
