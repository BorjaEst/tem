"""Medial Entorhinal Cortex (MEC) module for spatial navigation.

This module implements the MEC pathway using Facade and Strategy design patterns
to provide clean, extensible spatial navigation with optional object vector cells (OVCs).

Architecture:
    - Facade: MECModel provides clean public API, delegates complexity to strategies
    - Strategy: OVCInferenceStrategy handles 3 modes (NoOVC, Merged, Separate)
    - Factory: Helper methods encapsulate construction logic

OVC Backward Allocation Model:
    OVC portions allocated backwards from end of n_g_grid:
    - Full merged: n_g=[20,30,40], n_g_ovc=[10,10,10], f_ovc=[] → n_g_grid=[10,20,30]
    - Partial merged: n_g=[10,30,40], n_g_ovc=[10,10], f_ovc=[] → n_g_grid=[10,20,30]
    - Merged+separate: n_g=[10,30,40,10], n_g_ovc=[10,10,10], f_ovc=[0.1] → n_g_grid=[10,20,30]
    - Full separate: n_g=[10,20,30,10], n_g_ovc=[10], f_ovc=[0.1] → n_g_grid=[10,20,30]

Typical usage example:
    >>> config = MECConfig(ovc=ObjectInferenceConfig(n_g_ovc=[10,10], frequencies=[]))
    >>> context = ... # MECContext with W_down, W_repeat, f_initial
    >>> mec = MECModel(context, config)
    >>> state = mec.init_state(batch_size=4, device=torch.device('cpu'))
    >>> state = mec.forward(p_x, locations, action, state)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Optional, Protocol, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.core.mec.abstract import AbstractLocConfig, AbstractLocModel
from torch_tem.core.mec.object import ObjectInference, ObjectInferenceConfig
from torch_tem.core.mec.projection import Projection, ProjectionConfig
from torch_tem.core.mec.transition import TransitionConfig, TransitionModel
from torch_tem.types import AbstractLocation, GroundedLocation, MultiScaleCode, Transition

__all__ = ["MECConfig", "MECContext", "MECState", "MECModel"]


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

    # OVC extension
    ovc: ObjectInferenceConfig = Field(default_factory=ObjectInferenceConfig, description="OVC configuration. Empty = disabled")


class MECContext(Protocol):
    """Protocol for MEC model initialization parameters (architectural constants).

    Attributes:
        n_a: Number of possible actions from environment.
        f_initial: Base frequency values for hierarchical connections [n_f_grid].
        W_down: Downsampling matrices defining n_g dimensions [n_f].
        W_repeat: Expansion matrices defining n_p dimensions [n_f].
    """

    n_a: int
    f_initial: List[float]
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
        - Dimension resolution (backward OVC allocation)
        - OVC mode selection
        - Module initialization
        - Inference coordination

    OVC Backward Allocation:
        OVC portions allocated backwards from end of n_g_grid.
        See resolve_dimensions() for examples and implementation.
    """

    def __init__(self, context: MECContext, config: MECConfig):
        """Initialize MEC model.

        Args:
            context: Architectural constants (W_down, W_repeat, n_f_grid, f_initial, n_actions)
            config: Hyperparameters (learning, OVC config, submodule configs)
        """
        super().__init__()
        self._config = config
        self._dims = dims = resolve_dimensions(context, config)

        # Register projection matrices as parameters or buffers based on config
        self._W_down = nn.ParameterList([nn.Parameter(matrix, requires_grad=config.learn_W_down) for matrix in context.W_down])
        self._W_repeat = nn.ParameterList([nn.Parameter(matrix, requires_grad=config.learn_W_repeat) for matrix in context.W_repeat])

        # Initialize submodules
        self.projection = Projection(self._W_down, self._W_repeat, config.projection)
        self.abstract = AbstractLocModel(dims.n_g_grid, context.W_repeat[: dims.n_f_grid], config.abstract)
        self.transition = TransitionModel(n_g=dims.n_g, n_f_grid=dims.n_f_grid, n_actions=context.n_actions, f_initial=context.f_initial, config=config.transition)
        self.ovc = ObjectInference(dims.n_g, config.ovc)

    @property
    def W_down(self) -> nn.ParameterList:
        """Downsampling matrices (read-only for debugging)."""
        return self._W_down

    @property
    def W_repeat(self) -> nn.ParameterList:
        """Expansion matrices (read-only for debugging)."""
        return self._W_repeat

    @property
    def n_p(self) -> List[int]:
        """Place cell dimensions per frequency."""
        return self._dims.n_p

    @property
    def n_g(self) -> List[int]:
        """Abstract location dimensions per frequency."""
        return self._dims.n_g

    @property
    def n_f(self) -> int:
        """Total number of frequency modules."""
        return self._dims.n_f

    @property
    def n_f_grid(self) -> int:
        """Number of grid cell frequency modules."""
        return len(self._dims.n_g_grid)

    @property
    def n_f_ovc(self) -> int:
        """Number of object vector cell frequency modules."""
        return len(self._dims.n_g_ovc)

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
        g_gen: Transition = self.transition(state.abstract_location, a)

        # Step 2a: Grid cell inference (always uses n_g_grid portion)
        g_gen_grid = Transition(mean=g_gen.mean[: self.n_f_grid], uncertainty=g_gen.uncertainty[: self.n_f_grid])
        p_x_grid = p_x[: self.n_f_grid] if p_x is not None else None
        g_grid = self.abstract(g_gen_grid, p_x_grid)

        # Step 2b: OVC inference (only if separate modules exist)
        g_ovc = self.ovc(g_gen, locations)  # Returns [] if no separate OVC

        # Step 2c: Combine grid + ovc
        g = g_grid + g_ovc

        # Step 3: Projection
        g_ = self.projection(g)

        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)


# ============================================================================
# Dimension Resolution (Configuration Analysis)
# ============================================================================


@dataclass(frozen=True)
class DimensionConfig:
    """Resolved MEC dimensions from context and config.

    Computed by resolve_dimensions() using backward allocation:
    - n_g_ovc portions allocated backwards from end of n_g_grid
    - Element-wise addition: n_g[i] = n_g_grid[i] + n_g_ovc[i]

    Attributes:
        n_p: Hippocampal input dimensions per module (from W_repeat)
        n_g_grid: Grid cell dimensions per module (after OVC subtraction)
        n_g_ovc: OVC dimensions per module (zero-padded for element-wise addition)
    """

    n_p: List[int]
    n_g_grid: List[int]
    n_g_ovc: List[int]

    @property
    def n_g(self) -> List[int]:
        """Total abstract location dimensions per frequency"""
        return [g1 + g2 for g1, g2 in zip_longest(self.n_g_grid, self.n_g_ovc, fillvalue=0)]

    @property
    def n_f(self) -> int:
        """Total number of frequency modules"""
        return len(self.n_p)


def _validate_dimensions(context: MECContext, config: MECConfig) -> None:
    """Validate user-provided OVC configuration against matrix dimensions.

    Only validates what users control (config.ovc.n_g_ovc) against what's
    fixed by matrices (W_down, W_repeat). Internal consistency is guaranteed
    by construction.
    """
    # Validate matrix consistency (W_down and W_repeat must match)
    if (n_f := len(context.W_repeat)) != len(context.W_down):
        raise ValueError(f"Matrix mismatch: W_down has {len(context.W_down)} modules, W_repeat has {len(context.W_repeat)}")

    # Validate user-provided OVC config
    if not (n_g_ovc := config.ovc.n_g_ovc):
        return  # No OVC, nothing to validate

    # Check OVC doesn't exceed total modules
    if (n_f_ovc := len(n_g_ovc)) > n_f:
        raise ValueError(f"OVC config invalid: {n_f_ovc} OVC dimensions > {n_f} total modules")

    # Ensure OVC portions don't exceed grid dimensions
    n_g = [W.shape[0] for W in context.W_down]
    for i, ovc_dim in enumerate(n_g_ovc):
        if ovc_dim > n_g[i]:
            raise ValueError(f"Merged mode: OVC portion ({ovc_dim}) > grid dimension ({n_g[i]}) at module {i}")


def resolve_dimensions(context: MECContext, config: MECConfig) -> DimensionConfig:
    """Resolve dimensions: OVC portions allocated backwards from end of n_g_grid.

    Examples:
        n_g=[20,30,40], n_g_ovc=[10,10,10], f_ovc=[] → n_g_grid=[10,20,30] (full merged)
        n_g=[10,30,40], n_g_ovc=[10,10], f_ovc=[] → n_g_grid=[10,20,30] (partial merged)
        n_g=[10,30,40,10], n_g_ovc=[10,10,10], f_ovc=[0.1] → n_g_grid=[10,20,30] (merged+separate)
        n_g=[10,20,30,10], n_g_ovc=[10], f_ovc=[0.1] → n_g_grid=[10,20,30] (full separate)
    """
    _validate_dimensions(context, config)

    # Extract dimensions
    n_g = [W.shape[0] for W in context.W_down]
    n_p = [W.shape[1] for W in context.W_repeat]
    n_g_ovc_config = config.ovc.n_g_ovc
    n_f_ovc_separate = len(config.ovc.frequencies) if config.ovc.frequencies else 0

    if not n_g_ovc_config:
        # No OVC: all modules are grid
        return DimensionConfig(n_p=n_p, n_g_grid=n_g, n_g_ovc=[])

    # Determine grid module count
    n_f_grid = len(n_g) - n_f_ovc_separate

    # Allocate OVC portions backwards from end of grid modules
    n_g_grid = list(n_g[:n_f_grid])  # Start with total dimensions
    n_ovc_merged = len(n_g_ovc_config) - n_f_ovc_separate  # How many OVC portions are merged

    # Subtract merged OVC portions from the end backwards
    for i in range(n_ovc_merged):
        grid_idx = n_f_grid - 1 - i  # Count backwards from last grid module
        ovc_idx = n_ovc_merged - 1 - i  # Count backwards from merged OVC portions
        n_g_grid[grid_idx] -= n_g_ovc_config[ovc_idx]

    # Build full n_g_ovc: merged portions + separate modules
    n_g_ovc = [0] * (n_f_grid - n_ovc_merged)  # Modules without OVC
    n_g_ovc += n_g_ovc_config[:n_ovc_merged]  # Merged OVC portions
    n_g_ovc += n_g_ovc_config[n_ovc_merged:]  # Separate OVC modules

    return DimensionConfig(n_p=n_p, n_g_grid=n_g_grid, n_g_ovc=n_g_ovc)


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """MEC module usage example.

    Demonstrates spatial navigation with path integration, abstract location
    inference, and projection to hippocampal space.
    """
    print("=" * 80)
    print("MEC Module Example - Spatial Navigation")
    print("=" * 80)

    # Configuration
    n_g = [48, 40, 32]  # Grid cells per frequency
    n_p = [96, 80, 64]  # Place cells per frequency
    n_f_grid = 3
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Frequencies: {len(n_g)}")
    print(f"  Grid cells per frequency: {n_g}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Batch size: {batch_size}")

    # Create projection matrices (context)
    W_down = [torch.randn(n_g_f, n_g_f // 2) for n_g_f in n_g]
    W_repeat = [torch.randn(n_g_f // 2, n_p_f) for n_g_f, n_p_f in zip(n_g, n_p)]

    # Create mock context
    from dataclasses import dataclass

    @dataclass
    class MockContext:
        n_f_grid: int
        W_down: list
        W_repeat: list

    context = MockContext(n_f_grid=n_f_grid, W_down=W_down, W_repeat=W_repeat)

    # Create MEC model (No OVC mode)
    config = MECConfig(ovc=None)
    mec = MECModel(context, config)
    print(f"\n✓ MEC model initialized (mode: No OVC)")
    print(f"  n_g: {mec.n_g}")
    print(f"  n_p: {mec.n_p}")

    # Initialize state
    device = torch.device("cpu")
    state = mec.init_state(batch_size, device)
    print(f"✓ MEC state initialized")

    # Simulate forward pass
    p_x = None  # Generative mode (no sensory retrieval)
    locations = [{"shiny": None} for _ in range(batch_size)]
    action = torch.randint(0, 4, (batch_size,))

    print(f"\n✓ Simulated inputs:")
    print(f"  p_x: {p_x} (generative mode)")
    print(f"  action: {action.shape}")

    # Forward pass
    with torch.no_grad():
        state = mec.forward(p_x, locations, action, state)

    print(f"\n✓ Forward pass complete")
    print(f"  Abstract location: {[g.shape for g in state.abstract_location]}")
    print(f"  Projection: {[g.shape for g in state.projection]}")
    print(f"  Transition mean: {[g.shape for g in state.transition_stats.mean]}")

    print("\n" + "=" * 80)
