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

Loss Mode Architecture Decision
=================================

This implementation supports both "surrogate ELBO" (mean-only) and "full ELBO"
(probabilistic) training modes for the abstract location loss (L_g).

**MSE Mode (default, legacy-compatible)**:
- Uses simple squared error: 0.5 * ||g_inf - g_gen||²
- Matches the paper's approach for tasks without uncertainty
- Faster computation, stable gradients
- Sufficient when environment is fully observable and deterministic
- Default to preserve legacy training dynamics

**KL Mode (probabilistic ELBO)**:
- Uses KL divergence with learned uncertainty
- Requires Transition dataclass with .mean and .uncertainty
- Enables Bayesian treatment of spatial uncertainty
- Useful for partially observable or stochastic environments

Migration Path
--------------
1. Legacy code uses MSE mode for reproducibility (mode='mse')
2. New architectures can opt into KL mode via LossConfig.lg_mode
3. Both modes produce LossG dataclass for uniform interface

Implementation Notes
--------------------
- All squared errors include 0.5 factor by convention (matching utils.squared_error)
- Loss modules return dataclasses (LossX, LossP, LossG, LossReg) not scalars
- Reduction is configurable: 'none' (per-env), 'sum', or 'mean'
- Training loop uses reduction='none' for per-env weighting and masking

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
    # LOSS COMPUTATION MODE
    # ===================================================================================

    lg_mode: Literal["mse", "kl"] = Field(default="mse", description="Abstract location loss mode: 'mse'(legacy surrogate), 'kl'(KL divergence)")

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

    def forward(self, x_logits: list[Tensor], x: Tensor) -> LossX:
        """Compute L_x for all three pathways using categorical cross-entropy.

        Args:
            x_logits: List of 3 logit tensors [infer, retrieved, ancestral].
                Each tensor has shape (B, n_classes).
            x: Ground truth observation. Can be:
                - One-hot encoded: shape (B, n_classes)
                - Class indices: shape (B,) or (B, 1)

        Returns:
            LossX dataclass with per-pathway losses.

        Note:
            Uses the reduction method specified in __init__ (default 'sum').
            Returns per-env vectors [B] when reduction='none', scalars otherwise.
        """
        # Extract class labels from ground truth observation
        if x.dim() == 2 and x.shape[1] > 1:
            # One-hot encoded: take argmax
            labels = torch.argmax(x, dim=1)
        else:
            # Already class indices
            labels = x.view(-1).long()

        # Compute cross-entropy for each pathway
        # x_logits[0]: p_inf → x (inferred grounded location)
        # x_logits[1]: g_inf → p → x (retrieved via abstract location)
        # x_logits[2]: g_prev → g → p → x (ancestral/generative pathway)
        loss_infer = F.cross_entropy(x_logits[0], labels, reduction=self.reduction)
        loss_retrieved = F.cross_entropy(x_logits[1], labels, reduction=self.reduction)
        loss_ancestral = F.cross_entropy(x_logits[2], labels, reduction=self.reduction)

        return LossX(infer=loss_infer, retrieved=loss_retrieved, ancestral=loss_ancestral)


class AbstractLocationLoss(nn.Module):
    """Computes the consistency loss for abstract locations (L_g).

    This loss enforces that the MEC grid cells learn accurate transition dynamics.
    It penalizes discrepancies between two sources of spatial information:

    1. Path Integration (Prior): Where you should be based on movement alone
    2. Landmark Correction (Posterior): Where you are after sensory correction

    Dual-Mode Architecture:
        This module supports two computational modes to accommodate different
        training paradigms:

        **MSE Mode (default, legacy-compatible)**:
            L_g = ||g_inf - g_gen||²
            Simple squared error between inferred and generated abstract locations.
            Matches the paper's "mean-only surrogate" loss for tasks without uncertainty.
            This is the default to preserve legacy training dynamics.

        **KL Mode (probabilistic ELBO)**:
            L_g = D_KL( q(g | x, a) || p(g | a) )
            Analytical KL divergence assuming Gaussian prior with learned uncertainty.
            Approximates posterior as delta function at μ_post.

            KL ≈ 0.5 * ((μ_post - μ_prior) / σ_prior)² + ln(σ_prior)

            The first term is Mahalanobis distance (uncertainty-weighted squared error),
            and the second is the entropy of the prior.

    Usage:
        - For reproducing legacy results: use mode='mse' (default)
        - For probabilistic TEM with uncertainty: use mode='kl'

    Mathematical Formulation (KL mode):
        L_g = D_KL( q(g | x, a) || p(g | a) )

    where:
        - g: Abstract location (MEC grid cell activations)
        - x: Sensory observation
        - a: Action taken
        - q(g | x, a): Posterior after incorporating sensory evidence
        - p(g | a): Prior from path integration alone
    """

    def __init__(self, mode: Literal["mse", "kl"] = "mse", reduction: Literal["sum", "mean"] = "sum"):
        """Initialize AbstractLocationLoss.

        Args:
            mode: Computation mode:
                'mse': Legacy squared error (default for backward compatibility)
                'kl': Probabilistic KL divergence with uncertainty
            reduction: Specifies the reduction to apply to the output:
                'sum': Sum the loss over the batch.
                'mean': Take the mean of the loss over the batch.
                Default: 'sum'.
        """
        super().__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, g_inf: AbstractLocation, g_gen: AbstractLocation | Transition) -> LossG:
        """Compute L_g using either MSE or KL divergence based on mode.

        Args:
            g_inf: Inferred abstract location (posterior from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Generated/predicted abstract location.
                For MSE mode: List of tensors (same structure as g_inf)
                For KL mode: Transition dataclass with mean and uncertainty

        Returns:
            LossG dataclass with transition loss.

        Note:
            Uses the reduction method specified in __init__.
        """
        if self.mode == "mse":
            # Legacy surrogate: simple squared error between inferred and generated
            g_gen_list = g_gen.mean if isinstance(g_gen, Transition) else g_gen

            # Accumulate squared errors across all frequency modules
            loss_per_env = None
            for g_i, g_g in zip(g_inf, g_gen_list):
                se = 0.5 * (g_i - g_g).pow(2).sum(dim=-1)  # 0.5 factor by convention, sum over features, keep batch dim: [B]
                loss_per_env = se if loss_per_env is None else loss_per_env + se

            # Apply reduction
            if self.reduction == "sum":
                return LossG(transition=loss_per_env.sum())
            elif self.reduction == "mean":
                return LossG(transition=loss_per_env.mean())
            else:  # reduction == "none"
                return LossG(transition=loss_per_env)

        elif self.mode == "kl":
            # Probabilistic ELBO: KL divergence with uncertainty
            if not isinstance(g_gen, Transition):
                raise ValueError(f"KL mode requires g_gen to be Transition with uncertainty, got {type(g_gen)}")

            # Accumulate KL across frequency modules
            kl_per_env = None
            for mu_post, mu_prior, sigma_prior in zip(g_inf, g_gen.mean, g_gen.uncertainty):
                # Clamp sigma to avoid numerical instability
                sigma_prior = torch.clamp(sigma_prior, min=1e-6)

                # Mahalanobis distance: squared error weighted by inverse uncertainty
                diff = mu_post - mu_prior
                mahalanobis = 0.5 * (diff / sigma_prior).pow(2)

                # Log-determinant term: entropy of the prior
                log_det = torch.log(sigma_prior)

                # Sum over features, keep batch dim: [B]
                kl_f = (mahalanobis + log_det).sum(dim=-1)
                kl_per_env = kl_f if kl_per_env is None else kl_per_env + kl_f

            # Apply reduction
            if self.reduction == "sum":
                return LossG(transition=kl_per_env.sum())
            elif self.reduction == "mean":
                return LossG(transition=kl_per_env.mean())
            else:  # reduction == "none"
                return LossG(transition=kl_per_env)

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'mse' or 'kl'.")


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

    def forward(self, p_inf: GroundedLocation, p_gen: GroundedLocation, p_inf_x: Optional[GroundedLocation] = None, use_p_inf: bool = True) -> LossP:
        """Compute L_p using mean squared error for both pathways.

        Args:
            p_inf: Inferred grounded location (conjunctive place cells).
                List of tensors, one per frequency module. Shape: [(B, n_p_f1), (B, n_p_f2), ...]
            p_gen: Grounded location retrieved via abstract location (g_inf → p).
                Required for L_p_g computation.
            p_inf_x: Grounded location from sensory memory retrieval (x → p).
                Used for L_p_x if use_p_inf=True.
            use_p_inf: Whether to compute L_p_x (sensory consistency term).
                Set to False to disable this term (returns zeros).

        Returns:
            LossP dataclass with abstract and sensory components.

        Note:
            Uses the reduction method specified in __init__.
            Matches legacy behavior: sum of squared errors across all frequencies.
        """
        # L_p_g (abstract): consistency between inferred p and p retrieved via g
        # This is always computed
        loss_abstract_per_env = None
        for p_i, p_g in zip(p_inf, p_gen):
            se = 0.5 * (p_i - p_g).pow(2).sum(dim=-1)  # 0.5 factor by convention, sum over features, keep batch: [B]
            loss_abstract_per_env = se if loss_abstract_per_env is None else loss_abstract_per_env + se

        # L_p_x (sensory): consistency between inferred p and p retrieved via x
        # Only computed if use_p_inf=True and p_inf_x is provided
        if use_p_inf and p_inf_x is not None:
            loss_sensory_per_env = None
            for p_i, p_x in zip(p_inf, p_inf_x):
                se = 0.5 * (p_i - p_x).pow(2).sum(dim=-1)  # [B]
                loss_sensory_per_env = se if loss_sensory_per_env is None else loss_sensory_per_env + se
        else:
            # Return zeros if disabled (matching legacy behavior)
            batch_size = p_inf[0].shape[0]
            loss_sensory_per_env = torch.zeros(batch_size, device=p_inf[0].device, dtype=p_inf[0].dtype)

        # Apply reduction
        if self.reduction == "sum":
            return LossP(abstract=loss_abstract_per_env.sum(), sensory=loss_sensory_per_env.sum())
        elif self.reduction == "mean":
            return LossP(abstract=loss_abstract_per_env.mean(), sensory=loss_sensory_per_env.mean())
        else:  # reduction == "none"
            return LossP(abstract=loss_abstract_per_env, sensory=loss_sensory_per_env)


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

    def forward(self, g: AbstractLocation, p: GroundedLocation) -> LossReg:
        """Compute regularization penalties.

        Args:
            g: Abstract location (grid cells). List of tensors per frequency.
            p: Grounded location (place cells). List of tensors per frequency.

        Returns:
            LossReg dataclass with g_l2 and p_l1 components.

        Note:
            Uses the reduction method specified in __init__.
            Matches legacy: sum over all frequencies and dimensions.
        """
        # L2 regularization on grid cells: penalizes large activations
        # Σ_f g_{f,b,i}²
        reg_g_per_env = None
        for g_f in g:
            l2 = (g_f**2).sum(dim=-1)  # Sum over features, keep batch: [B]
            reg_g_per_env = l2 if reg_g_per_env is None else reg_g_per_env + l2

        # L1 regularization on place cells: enforces sparsity
        # Σ_f |p_{f,b,i}|
        reg_p_per_env = None
        for p_f in p:
            l1 = torch.abs(p_f).sum(dim=-1)  # [B]
            reg_p_per_env = l1 if reg_p_per_env is None else reg_p_per_env + l1

        # Apply reduction
        if self.reduction == "sum":
            return LossReg(g_l2=reg_g_per_env.sum(), p_l1=reg_p_per_env.sum())
        elif self.reduction == "mean":
            return LossReg(g_l2=reg_g_per_env.mean(), p_l1=reg_p_per_env.mean())
        else:  # reduction == "none"
            return LossReg(g_l2=reg_g_per_env, p_l1=reg_p_per_env)
