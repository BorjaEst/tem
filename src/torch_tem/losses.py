"""TEM loss components and structured loss outputs.

This module provides two layers:

1) Lightweight dataclasses (e.g., `LossX`, `LossP`, `LossG`, `LossReg`,
   `LossOutput`) that hold *structured* loss values. These support accumulation
   (addition) and averaging (division) in the training loop.
2) `torch.nn.Module` implementations that compute those dataclasses.

The default behaviour matches the legacy TEM training objective (mean-only
surrogate; no sampling). `AbstractLocationLoss` also supports a KL-based mode
for probabilistic training when transition uncertainty is available.

Conventions:
    - Squared-error terms include a 0.5 factor (matches `utils.squared_error`).
    - `reduction="none"` returns per-environment vectors of shape `[B]`, which
      the training loop relies on for visit masking and per-env weighting.

References:
    Whittington et al. (2020). The Tolman-Eichenbaum Machine.
"""

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.core import TEMState
from torch_tem.types import AbstractLocation, GroundedLocation, Transition

Reduction: TypeAlias = Literal["none", "sum", "mean"]


class LossConfig(BaseModel):
    """Configuration for loss computation and weighting.

    Defaults are chosen to match the legacy training behaviour.

    Attributes:
        lg_mode: Abstract-location loss mode used by `AbstractLocationLoss`.
        weights_x: Scalar multiplier for `LossX.total`.
        weights_p: Scalar multiplier for `LossP.total`.
        weights_g: Scalar multiplier for `LossG.total`.
        weights_reg_g: Scalar multiplier for `LossReg.g_l2`.
        weights_reg_p: Scalar multiplier for `LossReg.p_l1`.
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # ===================================================================================
    # LOSS COMPUTATION MODE
    # ===================================================================================

    lg_mode: Literal["mse", "kl"] = Field(default="mse", description="Abstract location loss mode: 'mse' (legacy), 'kl'.")

    x_reduction: Reduction = Field(default="none", description="Reduction for sensory reconstruction loss (L_x).")
    p_reduction: Reduction = Field(default="none", description="Reduction for grounded location loss (L_p).")
    g_reduction: Reduction = Field(default="none", description="Reduction for abstract location loss (L_g).")
    reg_reduction: Reduction = Field(default="none", description="Reduction for regularization losses.")

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
    """Sensory reconstruction section (L_x).

    Attributes:
        infer: Prediction from inferred grounded location (p_inf → x).
        retrieved: Prediction from memory retrieval via abstract location
            (g_inf → p → x).
        ancestral: Prediction from ancestral/generative rollout
            (g_{t-1} → g_t → p → x).
    """

    infer: Tensor  # x_p in legacy
    retrieved: Tensor  # x_g in legacy
    ancestral: Tensor  # x_gen in legacy

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossX":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossX` instance with all fields set to zero scalars.
        """
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
        """Total reconstruction loss (sum of pathways)."""
        return self.infer + self.retrieved + self.ancestral


@dataclass
class LossP:
    """Grounded location consistency section (L_p).

    Attributes:
        abstract: Consistency with memory retrieved via abstract location.
        sensory: Consistency with memory retrieved via the sensory pathway.
    """

    abstract: Tensor
    sensory: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossP":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossP` instance with all fields set to zero scalars.
        """
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
        """Total grounded-location loss."""
        return self.abstract + self.sensory


@dataclass
class LossG:
    """Abstract location transition consistency section (L_g).

    Attributes:
        transition: Transition consistency loss.
    """

    transition: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossG":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossG` instance with the field set to a zero scalar.
        """
        return cls(transition=torch.zeros((), device=device, dtype=dtype))

    def __add__(self, other: "LossG") -> "LossG":
        return LossG(transition=self.transition + other.transition)

    def __truediv__(self, divisor: int | float) -> "LossG":
        return LossG(transition=self.transition / divisor)

    @property
    def total(self) -> Tensor:
        """Total abstract-location loss."""
        return self.transition


@dataclass
class LossReg:
    """Regularization section (auxiliary penalties).

    Attributes:
        g_l2: L2 penalty on abstract location codes.
        p_l1: L1 penalty on grounded location codes.
    """

    g_l2: Tensor
    p_l1: Tensor

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossReg":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossReg` instance with all fields set to zero scalars.
        """
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
        """Total regularization loss."""
        return self.g_l2 + self.p_l1


@dataclass
class LossOutput:
    """Container for all loss sections.

    This is the primary object passed through the training loop. It supports
    elementwise addition and scalar division, which makes it convenient for
    accumulating per-timestep losses.

    Attributes:
        x: Sensory reconstruction losses.
        p: Grounded location consistency losses.
        g: Abstract location transition losses.
        reg: Regularization losses.
    """

    x: LossX
    p: LossP
    g: LossG
    reg: LossReg

    @classmethod
    def zero(cls, *, device, dtype=torch.float32) -> "LossOutput":
        """Create a zero-initialized instance.

        Args:
            device: Torch device.
            dtype: Torch dtype.

        Returns:
            A `LossOutput` instance with all fields set to zero scalars.
        """
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
        """Total loss (unweighted sum of all sections)."""
        return self.x.total + self.p.total + self.g.total + self.reg.total


StepLoss = LossOutput  # Alias for clarity in training context
AccumLoss = LossOutput  # Alias for accumulated losses over rollouts


class SensoryReconstructionLoss(nn.Module):
    """Compute sensory reconstruction losses (L_x).

    This computes three cross-entropy losses, one per TEM prediction pathway.
    It expects logits that already correspond to the three pathways used in the
    legacy implementation.
    """

    def __init__(self, reduction: Reduction = "sum"):
        """Initialize the loss module.

        Args:
            reduction: Output reduction.
                - "none": return vectors of shape `[B]`.
                - "sum": return a scalar.
                - "mean": return a scalar.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, x_logits: list[Tensor], x: Tensor) -> LossX:
        """Compute `LossX` from logits and ground-truth observations.

        Args:
            x_logits: Three logit tensors `[infer, retrieved, ancestral]`, each
                shaped `(B, n_classes)`.
            x: Ground-truth observation. Accepts either:
                - one-hot: `(B, n_classes)`
                - class indices: `(B,)` or `(B, 1)`

        Returns:
            LossX: Per-pathway cross-entropy losses.

        Raises:
            ValueError: If `x_logits` does not contain exactly 3 tensors.
        """
        if len(x_logits) != 3:
            raise ValueError(f"Expected 3 logit tensors for L_x, got {len(x_logits)}")
        if x.dim() == 2 and x.shape[1] > 1:
            labels = torch.argmax(x, dim=1)
        else:
            labels = x.view(-1).long()

        # Pathway order is fixed to match the legacy implementation.
        loss_infer = F.cross_entropy(x_logits[0], labels, reduction=self.reduction)
        loss_retrieved = F.cross_entropy(x_logits[1], labels, reduction=self.reduction)
        loss_ancestral = F.cross_entropy(x_logits[2], labels, reduction=self.reduction)

        return LossX(infer=loss_infer, retrieved=loss_retrieved, ancestral=loss_ancestral)


class AbstractLocationLoss(nn.Module):
    """Compute abstract location transition consistency losses (L_g).

    The default mode (`mode="mse"`) matches the legacy surrogate objective:
    $0.5\,\lVert g_{\mathrm{inf}} - g_{\mathrm{gen}} \rVert^2$.

    In `mode="kl"`, this computes an uncertainty-weighted term that requires a
    `Transition` (mean + uncertainty) and returns a per-frequency KL-like cost.
    """

    def __init__(self, mode: Literal["mse", "kl"] = "mse", reduction: Reduction = "sum"):
        """Initialize the loss module.

        Args:
            mode: Computation mode:
                'mse': Legacy squared error (default for backward compatibility)
                'kl': Probabilistic KL divergence with uncertainty
            reduction: Output reduction.
                - "none": return vectors of shape `[B]`.
                - "sum": return a scalar.
                - "mean": return a scalar.
        """
        super().__init__()
        self.mode = mode
        self.reduction = reduction

    def forward(self, g_inf: AbstractLocation, g_gen: AbstractLocation | Transition) -> LossG:
        """Compute `LossG` from inferred and generated abstract locations.

        Args:
            g_inf: Inferred abstract location (posterior from sensory + path integration).
                List of tensors, one per frequency module. Shape: [(B, n_g_f1), (B, n_g_f2), ...]
            g_gen: Generated/predicted abstract location.
                For MSE mode: List of tensors (same structure as g_inf)
                For KL mode: Transition dataclass with mean and uncertainty

        Returns:
            LossG dataclass with transition loss.

        Raises:
            ValueError: If `mode='kl'` is used without a `Transition` input.
        """
        if self.mode == "mse":
            g_gen_list = g_gen.mean if isinstance(g_gen, Transition) else g_gen

            loss_per_env = None
            for g_i, g_g in zip(g_inf, g_gen_list):
                se = 0.5 * (g_i - g_g).pow(2).sum(dim=-1)
                loss_per_env = se if loss_per_env is None else loss_per_env + se

            # Apply reduction
            if self.reduction == "sum":
                return LossG(transition=loss_per_env.sum())
            elif self.reduction == "mean":
                return LossG(transition=loss_per_env.mean())
            else:  # reduction == "none"
                return LossG(transition=loss_per_env)

        elif self.mode == "kl":
            if not isinstance(g_gen, Transition):
                raise ValueError(f"KL mode requires g_gen to be Transition with uncertainty, got {type(g_gen)}")

            kl_per_env = None
            for mu_post, mu_prior, sigma_prior in zip(g_inf, g_gen.mean, g_gen.uncertainty):
                sigma_prior = torch.clamp(sigma_prior, min=1e-6)
                diff = mu_post - mu_prior
                mahalanobis = 0.5 * (diff / sigma_prior).pow(2)
                log_det = torch.log(sigma_prior)
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
    """Compute grounded location consistency losses (L_p).

    This uses squared-error terms (with a 0.5 factor) to keep inferred grounded
    location codes consistent with memory retrieval via:
    - abstract location (g → p), and
    - sensory input (x → p), optionally.
    """

    def __init__(self, reduction: Reduction = "sum"):
        """Initialize the loss module.

        Args:
            reduction: Output reduction.
                - "none": return vectors of shape `[B]`.
                - "sum": return a scalar.
                - "mean": return a scalar.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, p_inf: GroundedLocation, p_gen: GroundedLocation, p_inf_x: Optional[GroundedLocation] = None, use_p_inf: bool = True) -> LossP:
        """Compute `LossP` from grounded location codes.

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

        Notes:
            Squared errors include a 0.5 factor by convention. When `use_p_inf`
            is False, the sensory term is zeroed.
        """
        loss_abstract_per_env = None
        for p_i, p_g in zip(p_inf, p_gen):
            se = 0.5 * (p_i - p_g).pow(2).sum(dim=-1)
            loss_abstract_per_env = se if loss_abstract_per_env is None else loss_abstract_per_env + se

        if use_p_inf and p_inf_x is not None:
            loss_sensory_per_env = None
            for p_i, p_x in zip(p_inf, p_inf_x):
                se = 0.5 * (p_i - p_x).pow(2).sum(dim=-1)
                loss_sensory_per_env = se if loss_sensory_per_env is None else loss_sensory_per_env + se
        else:
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
    """Compute auxiliary regularization losses.

    Computes:
    - L2 penalty on abstract locations (g)
    - L1 penalty on grounded locations (p)
    """

    def __init__(self, reduction: Reduction = "sum"):
        """Initialize the loss module.

        Args:
            reduction: Output reduction.
                - "none": return vectors of shape `[B]`.
                - "sum": return a scalar.
                - "mean": return a scalar.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, g: AbstractLocation, p: GroundedLocation) -> LossReg:
        """Compute `LossReg`.

        Args:
            g: Abstract location (grid cells). List of tensors per frequency.
            p: Grounded location (place cells). List of tensors per frequency.

        Returns:
            LossReg dataclass with g_l2 and p_l1 components.

        Notes:
            This matches legacy behaviour (sum across frequencies and features).
        """
        reg_g_per_env = None
        for g_f in g:
            l2 = (g_f**2).sum(dim=-1)
            reg_g_per_env = l2 if reg_g_per_env is None else reg_g_per_env + l2

        reg_p_per_env = None
        for p_f in p:
            l1 = torch.abs(p_f).sum(dim=-1)
            reg_p_per_env = l1 if reg_p_per_env is None else reg_p_per_env + l1

        # Apply reduction
        if self.reduction == "sum":
            return LossReg(g_l2=reg_g_per_env.sum(), p_l1=reg_p_per_env.sum())
        elif self.reduction == "mean":
            return LossReg(g_l2=reg_g_per_env.mean(), p_l1=reg_p_per_env.mean())
        else:  # reduction == "none"
            return LossReg(g_l2=reg_g_per_env, p_l1=reg_p_per_env)


class TEMLoss(nn.Module):
    """Compute weighted per-env loss components for a single TEM step.

    This module returns a `LossOutput` whose fields are already weighted according
    to `LossConfig`. With reductions set to "none", each field is a vector of
    shape [B], suitable for visited masking and per-env aggregation.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self._config = config

        # Per-env outputs are required by the training loop's visited mask,
        # so the default config should use reduction="none".
        self.loss_x_fn = SensoryReconstructionLoss(reduction=config.x_reduction)
        self.loss_p_fn = GroundedLocationLoss(reduction=config.p_reduction)
        self.loss_g_fn = AbstractLocationLoss(mode=config.lg_mode, reduction=config.g_reduction)
        self.loss_reg_fn = RegularizationLoss(reduction=config.reg_reduction)

    def forward(self, step: TEMState, use_p_inf: bool) -> LossOutput:
        """Compute weighted loss components for one timestep.

        Args:
            step: TEMState at current timestep.
            use_p_inf: Whether to include sensory grounded-location term.

        Returns:
            LossOutput where each component is already multiplied by config weights.
        """
        # Raw (possibly per-env) losses
        lx: LossX = self.loss_x_fn(step.x_logits, step.x)
        lp: LossP = self.loss_p_fn(step.p_inf, step.p_gen, step.p_inf_x, use_p_inf)
        lg: LossG = self.loss_g_fn(step.g_inf, step.g_gen)
        lreg: LossReg = self.loss_reg_fn(step.g_inf, step.p_inf)

        # Apply weights (scalar multipliers)
        wx = self._config.weights_x
        wp = self._config.weights_p
        wg = self._config.weights_g
        wrg = self._config.weights_reg_g
        wrp = self._config.weights_reg_p

        lx = LossX(infer=wx * lx.infer, retrieved=wx * lx.retrieved, ancestral=wx * lx.ancestral)
        lp = LossP(abstract=wp * lp.abstract, sensory=wp * lp.sensory)
        lg = LossG(transition=wg * lg.transition)
        lreg = LossReg(g_l2=wrg * lreg.g_l2, p_l1=wrp * lreg.p_l1)

        return LossOutput(x=lx, p=lp, g=lg, reg=lreg)
