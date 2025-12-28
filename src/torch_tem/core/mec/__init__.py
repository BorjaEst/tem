"""MEC Model with Facade + Strategy pattern.

Architecture:
- Facade: MECModel provides clean public API, delegates complexity to strategies
- Strategy: OVCInferenceStrategy handles 3 modes (NoOVC, Merged, Separate)
- Factory: Helper methods encapsulate construction logic
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.core.mec.abstract import AbstractLocConfig, AbstractLocModel
from torch_tem.core.mec.object import ObjectInference, ObjectInferenceConfig
from torch_tem.core.mec.projection import Projection, ProjectionConfig
from torch_tem.core.mec.transition import TransitionConfig, TransitionModel
from torch_tem.types import AbstractLocation, GroundedLocation, MultiScaleCode, Transition

__all__ = ["MECConfig", "MECState", "MECModel"]


class MECConfig(BaseModel):
    """MEC model configuration parameters (hyperparameters only)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # Learning projection matrices
    learn_W_down: bool = Field(default=False, description="If True, downsampling matrices W_down are learnable")
    learn_W_repeat: bool = Field(default=False, description="If True, expansion matrices W_repeat are learnable")

    # Submodule configurations
    abstract: AbstractLocConfig = Field(default_factory=AbstractLocConfig, description="Abstract location inference configuration")
    transition: TransitionConfig = Field(default_factory=TransitionConfig, description="Transition model configuration")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig, description="Projection configuration")

    # Optional OVC extension
    ovc: Optional[ObjectInferenceConfig] = Field(default=None, description="OVC configuration. None = disabled")


class MECContext(Protocol):
    """Protocol for MEC model initialization parameters (architectural constants).

    Attributes:
        n_f_grid: Number of grid-only modules (determines OVC mode)
        W_down: Downsampling matrices defining n_g dimensions
        W_repeat: Expansion matrices defining n_p dimensions
    """

    n_f_grid: int
    W_down: List[Tensor]
    W_repeat: List[Tensor]


@dataclass
class MECState:
    """State container for MEC pathway."""

    transition_stats: Transition
    abstract_location: AbstractLocation
    projection: MultiScaleCode

    def detach(self) -> "MECState":
        """Detach all tensors from computation graph."""
        g_mean = [g.detach() for g in self.transition_stats.mean]
        g_uncertainty = [s.detach() for s in self.transition_stats.uncertainty]

        return MECState(
            transition_stats=Transition(mean=g_mean, uncertainty=g_uncertainty),
            abstract_location=[g.detach() for g in self.abstract_location],
            projection=[p.detach() for p in self.projection],
        )


class MECModel(nn.Module):
    """Medial Entorhinal Cortex (MEC) pathway for spatial navigation.

    FACADE PATTERN: Provides clean, simple API hiding internal complexity.
    STRATEGY PATTERN: Delegates OVC mode handling to pluggable strategies.

    Public API (Facade):
        - Properties: n_g, n_p, n_f, n_f_grid, n_f_ovc
        - Methods: init_state(), forward()

    Internal Complexity (hidden from users):
        - Dimension resolution
        - OVC mode selection
        - Module initialization
        - Inference coordination

    Supports 3 OVC modes (auto-detected from context + config):
        - No OVC: Grid cells only
        - Merged OVC: OVCs share grid frequencies
        - Separate OVC: OVCs in independent modules
    """

    def __init__(self, context: MECContext, config: MECConfig):
        """Initialize MEC model.

        Args:
            context: Architectural constants (W_down, W_repeat, n_f_grid)
            config: Hyperparameters (learning, OVC config, submodule configs)
        """
        super().__init__()
        self._config = config

        # Resolve dimensions and build submodules
        self._dims = resolve_dimensions(context, config)
        self._build_projection_matrices(context)
        self._build_submodules()
        self._strategy = _create_strategy(config)

    def _build_projection_matrices(self, context: MECContext) -> None:
        """Register projection matrices as parameters."""
        self._W_down = nn.ParameterList([nn.Parameter(matrix, requires_grad=self._config.learn_W_down) for matrix in context.W_down])
        self._W_repeat = nn.ParameterList([nn.Parameter(matrix, requires_grad=self._config.learn_W_repeat) for matrix in context.W_repeat])

    def _build_submodules(self) -> None:
        """Initialize all MEC submodules."""
        self.projection = Projection(self._W_down, self._W_repeat, self._config.projection)
        self.abstract = AbstractLocModel(self._dims.n_g_grid, self._dims.n_p[: len(self._dims.n_g_grid)], self._config.abstract)
        self.transition = TransitionModel(self._config.transition)
        self.ovc = ObjectInference(self._dims.n_g, self._dims.n_g_ovc, self._config.ovc)

    @property
    def n_g(self) -> List[int]:
        """Abstract location dimensions per frequency."""
        return self._dims.n_g

    @property
    def n_p(self) -> List[int]:
        """Place cell dimensions per frequency."""
        return self._dims.n_p

    @property
    def n_f(self) -> int:
        """Total number of frequency modules."""
        return self._dims.n_f_total

    @property
    def n_f_grid(self) -> int:
        """Number of grid frequency modules."""
        return self._dims.n_f_grid

    @property
    def n_f_ovc(self) -> int:
        """Number of OVC frequency modules."""
        return self._dims.n_f_ovc

    @property
    def W_down(self) -> nn.ParameterList:
        """Downsampling matrices (read-only for debugging)."""
        return self._W_down

    @property
    def W_repeat(self) -> nn.ParameterList:
        """Expansion matrices (read-only for debugging)."""
        return self._W_repeat

    def init_state(self, batch_size: int, device: torch.device) -> MECState:
        """Initialize MEC state with zeros.

        Args:
            batch_size: Batch size for tensor allocation
            device: Device for tensor allocation

        Returns:
            Initial MECState with zero-initialized abstract locations
        """
        # Create zero tensors
        zeros = lambda n_g_f: torch.zeros(batch_size, n_g_f, dtype=torch.float, device=device)

        g = [zeros(n_g_f) for n_g_f in self.n_g]
        g_mean = [zeros(n_g_f) for n_g_f in self.n_g]
        g_uncertainty = [zeros(n_g_f) for n_g_f in self.n_g]

        return MECState(
            transition_stats=Transition(mean=g_mean, uncertainty=g_uncertainty),
            abstract_location=g,
            projection=self.projection(g),
        )

    def forward(self, p_x: Optional[GroundedLocation], locations: List[Dict], a: Optional[Tensor], state: MECState) -> MECState:
        """Forward pass through MEC pathway.

        DECLARATIVE PIPELINE:
        1. Transition: Predict next abstract location via path integration
        2. Inference: Fuse prediction with memory/landmarks (strategy-based)
        3. Projection: Map to hippocampal input space

        Args:
            p_x: Hippocampal pattern from sensory (None in generative mode)
            locations: Environment descriptors for landmark cues
            a: Action taken
            state: Previous MEC state

        Returns:
            Updated MEC state with new abstract location
        """
        # Step 1: Transition (path integration)
        g_gen = self.transition(state.abstract_location, a)

        # Step 2: Inference (strategy delegates to appropriate mode)
        g = self._strategy.infer(g_gen, p_x, locations, self)

        # Step 3: Projection
        g_ = self.projection(g)

        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)


# ============================================================================
# Dimension Resolution (Configuration Analysis)
# ============================================================================


@dataclass(frozen=True)
class DimensionConfig:
    """Resolved dimension configuration for MEC architecture.

    Attributes:
        n_g: Abstract location dimensions per frequency
        n_p: Place cell dimensions per frequency
        n_g_grid: Grid cell dimensions (subset or full depending on mode)
        n_g_ovc: OVC dimensions (empty if no OVC)
        n_f_total: Total frequency modules
        n_f_grid: Grid frequency modules
        n_f_ovc: OVC frequency modules
    """

    n_g: List[int]
    n_p: List[int]
    n_g_grid: List[int]
    n_g_ovc: List[int]
    n_f_total: int
    n_f_grid: int
    n_f_ovc: int


def _extract_base_dimensions(context: MECContext) -> Tuple[List[int], List[int], int, int, int]:
    """Extract basic dimensions from context matrices.

    Returns:
        Tuple of (n_g, n_p, n_f_total, n_f_grid, n_f_ovc)
    """
    n_g = [W.shape[0] for W in context.W_down]
    n_p = [W.shape[1] for W in context.W_repeat]
    n_f_total = len(n_g)
    n_f_grid = context.n_f_grid
    n_f_ovc = n_f_total - n_f_grid
    return n_g, n_p, n_f_total, n_f_grid, n_f_ovc


def _validate_no_ovc_mode(n_f_ovc: int, n_f_total: int) -> None:
    """Validate configuration for No OVC mode."""
    if n_f_ovc != 0:
        raise ValueError(
            f"No OVC mode requires n_f_grid == n_f_total. "
            f"Context allocates {n_f_ovc} OVC modules. "
            f"Set n_f_grid={n_f_total} to disable OVC."
        )  # fmt: skip


def _validate_separate_ovc_mode(n_f_ovc: int, n_f_total: int, n_ovc_list: List[int], frequencies: List[float], n_g: List[int], n_f_grid: int) -> None:
    """Validate configuration for Separate OVC mode."""
    if n_f_ovc == 0:
        raise ValueError(
            f"Separate OVC mode requires n_f_grid < n_f_total. "
            f"Context has n_f_grid={n_f_grid}, n_f_total={n_f_total}. "
            f"Set n_f_grid < {n_f_total} to allocate OVC modules. "
        )  # fmt: skip

    if len(n_ovc_list) != n_f_ovc:
        raise ValueError(
            f"Separate OVC mode: Config specifies {len(n_ovc_list)} OVC modules "
            f"but context allocates {n_f_ovc} modules (n_f_total - n_f_grid). "
            f"Ensure len(config.ovc.n_ovc) == {n_f_ovc}."
        )  # fmt: skip

    # Validate dimension consistency (n_g_ovc = 3 * n_ovc)
    n_g_ovc = n_g[n_f_grid:]
    for f, (actual, expected_base) in enumerate(zip(n_g_ovc, n_ovc_list)):
        expected = 3 * expected_base
        if actual != expected:
            raise ValueError(
                f"Separate OVC module {f}: Dimension mismatch. "
                f"Context has {actual} neurons but config expects {expected} (3 * {expected_base}). "
                f"Verify W_down matrices match config.ovc.n_ovc."
            )  # fmt: skip


def _validate_merged_ovc_mode(n_f_ovc: int, n_f_total: int, n_ovc_list: List[int]) -> None:
    """Validate configuration for Merged OVC mode."""
    if n_f_ovc != 0:
        raise ValueError(
            f"Merged OVC mode requires n_f_grid == n_f_total. "
            f"Context allocates {n_f_ovc} separate OVC modules. "
            f"Set n_f_grid={n_f_total} for merged mode."
        )  # fmt: skip

    if len(n_ovc_list) != n_f_total:
        raise ValueError(
            f"Merged OVC mode: config.ovc.n_ovc must match all frequency modules. "
            f"Expected {n_f_total} modules but got {len(n_ovc_list)}. "
            f"Ensure len(config.ovc.n_ovc) == {n_f_total}."
        )  # fmt: skip


def resolve_dimensions(context: MECContext, config: MECConfig) -> DimensionConfig:
    """Resolve dimensions from context and config.

    Single source of truth for dimension calculation. Validates consistency
    between context (architectural constants) and config (hyperparameters).

    Args:
        context: Architectural constants (W_down, W_repeat, n_f_grid)
        config: Hyperparameters (ovc config)

    Returns:
        Validated dimension configuration

    Raises:
        ValueError: If context and config are incompatible
    """
    n_g, n_p, n_f_total, n_f_grid, n_f_ovc = _extract_base_dimensions(context)

    # Mode A: No OVC
    if config.ovc is None:
        _validate_no_ovc_mode(n_f_ovc, n_f_total)
        return DimensionConfig(n_g=n_g, n_p=n_p, n_g_grid=n_g, n_g_ovc=[], n_f_total=n_f_total, n_f_grid=n_f_grid, n_f_ovc=0)

    # Mode C: Separate OVC modules
    if config.ovc.frequencies is not None:
        _validate_separate_ovc_mode(n_f_ovc, n_f_total, config.ovc.n_ovc, config.ovc.frequencies, n_g, n_f_grid)
        return DimensionConfig(n_g=n_g, n_p=n_p, n_g_grid=n_g[:n_f_grid], n_g_ovc=n_g[n_f_grid:], n_f_total=n_f_total, n_f_grid=n_f_grid, n_f_ovc=n_f_ovc)

    # Mode B: Merged OVC (frequencies=None)
    _validate_merged_ovc_mode(n_f_ovc, n_f_total, config.ovc.n_ovc)
    return DimensionConfig(n_g=n_g, n_p=n_p, n_g_grid=n_g, n_g_ovc=[3 * ovc for ovc in config.ovc.n_ovc], n_f_total=n_f_total, n_f_grid=n_f_grid, n_f_ovc=0)


# ============================================================================
# Strategy Pattern: OVC Inference Modes
# ============================================================================


class MECSubmodules(Protocol):
    """Protocol defining MEC submodules needed by inference strategies.

    This allows strategies to access only what they need without holding
    references or receiving long parameter lists.
    """

    abstract: AbstractLocModel
    ovc: ObjectInference
    n_f_grid: int


class OVCInferenceStrategy(ABC):
    """Strategy interface for OVC inference modes.

    Three implementations:
    - NoOVCStrategy: No object vector cells (grid only)
    - MergedOVCStrategy: OVCs merged within grid modules
    - SeparateOVCStrategy: OVCs in separate frequency modules
    """

    @abstractmethod
    def infer(self, g_gen: Transition, p_x: Optional[GroundedLocation], locations: List[Dict], mec: MECSubmodules) -> AbstractLocation:
        """Infer abstract location (grid + ovc).

        Args:
            g_gen: Path integration prediction
            p_x: Memory retrieval from sensory (None in generative mode)
            locations: Environment descriptors for landmark cues
            mec: MEC submodules (protocol - only accesses abstract, ovc, n_f_grid)

        Returns:
            Fused abstract location [n_f] of [B, n_g[f]]
        """
        pass


class NoOVCStrategy(OVCInferenceStrategy):
    """Strategy for No OVC mode (grid cells only)."""

    def infer(self, g_gen: Transition, p_x: Optional[GroundedLocation], locations: List[Dict], mec: MECSubmodules) -> AbstractLocation:
        """Pure grid cell inference."""
        return mec.abstract(g_gen, p_x)


class MergedOVCStrategy(OVCInferenceStrategy):
    """Strategy for Merged OVC mode (OVCs share grid frequencies)."""

    def infer(self, g_gen: Transition, p_x: Optional[GroundedLocation], locations: List[Dict], mec: MECSubmodules) -> AbstractLocation:
        """Grid inference handles merged grid+ovc modules."""
        return mec.abstract(g_gen, p_x)


class SeparateOVCStrategy(OVCInferenceStrategy):
    """Strategy for Separate OVC mode (independent OVC modules)."""

    def infer(self, g_gen: Transition, p_x: Optional[GroundedLocation], locations: List[Dict], mec: MECSubmodules) -> AbstractLocation:
        """Separate grid and ovc inference, then concatenate."""
        # Split inputs for grid-only modules
        g_gen_grid = Transition(mean=g_gen.mean[: mec.n_f_grid], uncertainty=g_gen.uncertainty[: mec.n_f_grid])
        p_x_grid = p_x[: mec.n_f_grid] if p_x is not None else None

        # Infer grid and ovc separately
        g_grid = mec.abstract(g_gen_grid, p_x_grid)
        g_ovc = mec.ovc(g_gen, locations)

        # Concatenate: [grid modules] + [ovc modules]
        return g_grid + g_ovc


def _create_strategy(config: MECConfig) -> OVCInferenceStrategy:
    """Factory method to create appropriate OVC inference strategy.

    Args:
        config: MEC configuration

    Returns:
        OVC inference strategy instance
    """
    if config.ovc is None:
        return NoOVCStrategy()
    if config.ovc.frequencies is not None:
        return SeparateOVCStrategy()
    if config.ovc.frequencies is None:
        return MergedOVCStrategy()
    raise ValueError("Invalid OVC configuration: Unable to determine inference strategy.")
