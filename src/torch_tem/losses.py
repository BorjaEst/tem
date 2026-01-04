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
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem import utils
from torch_tem.core import TEMState
from torch_tem.types import AbstractLocation, GroundedLocation, Reduction, Scalar, Transition


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

    def __mul__(self, scalar: Scalar) -> "LossX":
        return LossX(
            infer=self.infer * scalar,
            retrieved=self.retrieved * scalar,
            ancestral=self.ancestral * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossX":
        return self.__mul__(scalar)

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

    def __mul__(self, scalar: Scalar) -> "LossP":
        return LossP(
            abstract=self.abstract * scalar,
            sensory=self.sensory * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossP":
        return self.__mul__(scalar)

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

    def __mul__(self, scalar: Scalar) -> "LossG":
        return LossG(transition=self.transition * scalar)

    def __rmul__(self, scalar: Scalar) -> "LossG":
        return self.__mul__(scalar)

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

    def __mul__(self, scalar: Scalar) -> "LossReg":
        """Uniformly scale all regularization components."""
        return LossReg(g_l2=self.g_l2 * scalar, p_l1=self.p_l1 * scalar)

    def __rmul__(self, scalar: Scalar) -> "LossReg":
        return self.__mul__(scalar)

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

    def __mul__(self, scalar: Scalar) -> "LossOutput":
        return LossOutput(
            x=self.x * scalar,
            p=self.p * scalar,
            g=self.g * scalar,
            reg=self.reg * scalar,
        )

    def __rmul__(self, scalar: Scalar) -> "LossOutput":
        return self.__mul__(scalar)

    @property
    def total(self) -> Tensor:
        """Total loss (unweighted sum of all sections)."""
        return self.x.total + self.p.total + self.g.total + self.reg.total


StepLoss = LossOutput  # Alias for clarity in training context
AccumLoss = LossOutput  # Alias for accumulated losses over rollouts


class SensoryReconstructionConfig(BaseModel):
    """Configuration for sensory reconstruction loss (L_x)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    reduction: Reduction = Field(default="none", description="Reduction for sensory reconstruction loss.")
    weight: float = Field(default=1.0, ge=0, description="Weight multiplier for all L_x components.")


class SensoryReconstructionLoss(nn.Module):
    """Compute sensory reconstruction losses (L_x).

    This computes three cross-entropy losses, one per TEM prediction pathway.
    It expects logits that already correspond to the three pathways used in the
    legacy implementation.
    """

    def __init__(self, config: Optional[SensoryReconstructionConfig] = None):
        """Initialize the loss module.

        Args:

        """
        super().__init__()
        self.config = config or SensoryReconstructionConfig()

    @property
    def reduction(self) -> Reduction:
        """Map reduction from config to PyTorch string."""
        return self.config.reduction

    @property
    def weight(self) -> float:
        return self.config.weight

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

        return LossX(infer=loss_infer, retrieved=loss_retrieved, ancestral=loss_ancestral) * self.weight


class AbstractLocationConfig(BaseModel):
    """Configuration for abstract location transition loss (L_g)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    mode: Literal["mse", "kl"] = Field(default="mse", description="Loss mode: 'mse' (legacy surrogate), 'kl' (with uncertainty).")
    reduction: Reduction = Field(default="none", description="Reduction for abstract location loss.")
    weight: float = Field(default=1.0, ge=0, description="Weight multiplier for L_g.transition.")


class AbstractLocationLoss(nn.Module):
    """Compute abstract location transition consistency losses (L_g).

    The default mode (`mode="mse"`) matches the legacy surrogate objective:
    $0.5\,\lVert g_{\mathrm{inf}} - g_{\mathrm{gen}} \rVert^2$.

    In `mode="kl"`, this computes an uncertainty-weighted term that requires a
    `Transition` (mean + uncertainty) and returns a per-frequency KL-like cost.
    """

    def __init__(self, config: Optional[AbstractLocationConfig] = None):
        """Initialize the loss module.

        Args:
        """
        super().__init__()
        self.config = config or AbstractLocationConfig()

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def reduction(self) -> Reduction:
        return self.config.reduction

    @property
    def weight(self) -> float:
        return self.config.weight

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

            transition = utils.reduce_per_env(loss_per_env, self.reduction)
            return LossG(transition=transition) * self.weight

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

            transition = utils.reduce_per_env(kl_per_env, self.reduction)
            return LossG(transition=transition) * self.weight

        else:
            raise ValueError(f"Unknown mode: {self.mode}. Expected 'mse' or 'kl'.")


class GroundedLocationConfig(BaseModel):
    """Configuration for grounded location consistency loss (L_p)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    reduction: Reduction = Field(default="none", description="Reduction for grounded location loss.")
    weight: float = Field(default=1.0, ge=0, description="Weight multiplier for all L_p components.")


class GroundedLocationLoss(nn.Module):
    """Compute grounded location consistency losses (L_p).

    This uses squared-error terms (with a 0.5 factor) to keep inferred grounded
    location codes consistent with memory retrieval via:
    - abstract location (g → p), and
    - sensory input (x → p), optionally.
    """

    def __init__(self, config: Optional[GroundedLocationConfig] = None):
        """Initialize the loss module.

        Args:
        """
        super().__init__()
        self.config = config or GroundedLocationConfig()

    @property
    def reduction(self) -> Reduction:
        return self.config.reduction

    @property
    def weight(self) -> float:
        return self.config.weight

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

        abstract = utils.reduce_per_env(loss_abstract_per_env, self.reduction)
        sensory = utils.reduce_per_env(loss_sensory_per_env, self.reduction)
        return LossP(abstract=abstract, sensory=sensory) * self.weight


class RegularizationConfig(BaseModel):
    """Configuration for regularization penalties."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    reduction: Reduction = Field(default="none", description="Reduction for regularization losses.")
    weight_g_l2: float = Field(default=0.01, ge=0, description="Weight for abstract location L2 penalty.")
    weight_p_l1: float = Field(default=0.02, ge=0, description="Weight for grounded location L1 penalty.")


class RegularizationLoss(nn.Module):
    """Compute auxiliary regularization losses.

    Computes:
    - L2 penalty on abstract locations (g)
    - L1 penalty on grounded locations (p)
    """

    def __init__(self, config: Optional[RegularizationConfig] = None):
        """Initialize the loss module.

        Args:
        """
        super().__init__()
        self.config = config or RegularizationConfig()

    @property
    def reduction(self) -> Reduction:
        return self.config.reduction

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

        g_l2 = utils.reduce_per_env(reg_g_per_env, self.reduction)
        p_l1 = utils.reduce_per_env(reg_p_per_env, self.reduction)
        return LossReg(g_l2=g_l2 * self.config.weight_g_l2, p_l1=p_l1 * self.config.weight_p_l1)


class LossConfig(BaseModel):

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    x: SensoryReconstructionConfig = Field(default_factory=SensoryReconstructionConfig, description="Sensory reconstruction loss config.")
    g: AbstractLocationConfig = Field(default_factory=AbstractLocationConfig, description="Abstract location loss config.")
    p: GroundedLocationConfig = Field(default_factory=GroundedLocationConfig, description="Grounded location loss config.")
    reg: RegularizationConfig = Field(default_factory=RegularizationConfig, description="Regularization loss config.")


class TEMLoss(nn.Module):
    """Compute weighted per-env loss components for a single TEM step.
    This module encapsulates all loss computation and weighting, returning a
    `LossOutput` where each component is already weighted according to config.
    With default reduction="none", outputs are per-env vectors suitable for
    visited masking in the training loop.
    """

    def __init__(self, config: LossConfig):
        super().__init__()
        self._config = config

        # Per-env outputs are required by the training loop's visited mask,
        self.loss_x_fn = SensoryReconstructionLoss(config.x)
        self.loss_p_fn = GroundedLocationLoss(config.p)
        self.loss_g_fn = AbstractLocationLoss(config.g)
        self.loss_reg_fn = RegularizationLoss(config.reg)

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

        return LossOutput(x=lx, p=lp, g=lg, reg=lreg)
