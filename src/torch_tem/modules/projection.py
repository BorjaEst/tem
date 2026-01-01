"""
Unified projection module for mapping between different neural codes.

Supports 4 projection modes via strategy pattern:
- identity: pass-through (n_in == n_out)
- tiling: repeat input into output (LEC default)
- low_rank: two-stage compression (MEC default, combines downsample+repeat and low-rank)
- random: sparse/dense random projection
"""

from math import gcd
from typing import Dict, List, Literal, Optional, Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from torch_tem import utils
from torch_tem.types import Matrix, MultiScaleCode

__all__ = ["ProjectionConfig", "LECPConfig", "MECPConfig", "Projection"]

Activation = Literal["sigmoid", "none"]
ProjectionMode = Literal["identity", "tiling", "low_rank", "random"]
InitStrategy = Literal["identity", "random"]


# ======================================================================================
# CONFIGURATION
# ======================================================================================


class ProjectionConfig(BaseModel):
    """Unified projection configuration using mode-based strategy selection.

    Modes:
    - identity: no transformation (requires n_in == n_out)
    - tiling: repeat-based expansion (requires n_out % n_in == 0)
    - low_rank: two-stage compression/expansion
        - init="identity": structured downsample → repeat (MEC default)
        - init="random": learnable low-rank factorization A @ B
    - random: sparse/dense random projection
    """

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    mode: ProjectionMode = Field(..., description="Projection mode strategy")
    init: InitStrategy = Field(default="identity", description="Initialization strategy: 'identity' (structured) or 'random'")
    activation: Activation = Field(default="none", description="Activation applied after projection: 'sigmoid' or 'none'")
    learn: bool = Field(default=False, description="If True, projection matrices are learnable")
    rank: Optional[int] = Field(default=None, description="Low_rank rank for low_rank mode (auto-derived via GCD if None)")
    sparsity: float = Field(default=1.0, ge=0.0, le=1.0, description="Sparsity for random mode (1.0=dense)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible initialization")


class LECPConfig(ProjectionConfig):
    """LEC projection defaults: tiling with sigmoid activation."""

    mode: ProjectionMode = Field(default="tiling", description="LEC default: tiling")
    activation: Activation = Field(default="sigmoid", description="LEC default: sigmoid")
    learn: bool = Field(default=False, description="LEC default: fixed weights")


class MECPConfig(ProjectionConfig):
    """MEC projection defaults: low_rank with identity init (structured downsample+repeat)."""

    mode: ProjectionMode = Field(default="low_rank", description="MEC default: low_rank")
    init: InitStrategy = Field(default="identity", description="MEC default: structured (downsample+repeat)")
    activation: Activation = Field(default="none", description="MEC default: no activation")
    learn: bool = Field(default=False, description="MEC default: fixed weights")


# ======================================================================================
# STRATEGY PATTERN: MODE IMPLEMENTATIONS
# ======================================================================================


class IProjectionMode(Protocol):
    """Protocol defining the interface for projection mode strategies."""

    def validate(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> None:
        """Validate configuration and dimensions for this mode.

        Raises:
            ValueError: If constraints are violated.
        """
        ...

    def build(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[List[Matrix]]:
        """Build projection pipeline (list of matrix steps).

        Returns:
            List of steps, where each step is a list of matrices (one per frequency).
            - 0 steps: identity
            - 1 step: single matrix transformation
            - 2 steps: two-stage transformation
        """
        ...


class IdentityMode:
    """Identity projection: pass-through with no transformation."""

    def validate(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> None:
        n_f = len(n_in)
        if len(n_out) != n_f:
            raise ValueError(f"n_in and n_out must have same length, got {n_f} vs {len(n_out)}")
        for f in range(n_f):
            if n_in[f] != n_out[f]:
                raise ValueError(f"Identity requires n_in == n_out; freq {f}: {n_in[f]} != {n_out[f]}")

    def build(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[List[Matrix]]:
        return []  # No matrices needed


class TilingMode:
    """Tiling projection: repeat input to expand dimensions."""

    def validate(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> None:
        n_f = len(n_in)
        if len(n_out) != n_f:
            raise ValueError(f"n_in and n_out must have same length, got {n_f} vs {len(n_out)}")
        for f in range(n_f):
            if n_out[f] % n_in[f] != 0:
                raise ValueError(f"Tiling requires n_out % n_in == 0; freq {f}: {n_out[f]} % {n_in[f]} != 0")

    def build(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[List[Matrix]]:
        if config.init == "random":
            # Random tiling: use random matrices instead of structured repeat
            return [utils.create_random_projection(n_in, n_out, sparsity=config.sparsity, seed=config.seed)]
        else:
            # Structured tiling: deterministic repeat pattern
            return [utils.create_tiling_matrices(n_in, n_out)]


class LowRankMode:
    """LowRank projection: two-stage compression/expansion.

    init="identity": structured downsample → repeat (MEC default)
    init="random": learnable low-rank factorization A @ B
    """

    def validate(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> None:
        n_f = len(n_in)
        if len(n_out) != n_f:
            raise ValueError(f"n_in and n_out must have same length, got {n_f} vs {len(n_out)}")

        # Compute or validate ranks
        ranks = self._get_ranks(n_in, n_out, config)

        for f in range(n_f):
            if config.init == "identity":
                # Structured: rank must allow downsample + repeat
                if ranks[f] > n_in[f]:
                    raise ValueError(f"Low_rank init=identity requires rank <= n_in; freq {f}: {ranks[f]} > {n_in[f]}")
                if n_out[f] % ranks[f] != 0:
                    raise ValueError(f"Low_rank init=identity requires n_out % rank == 0; freq {f}: {n_out[f]} % {ranks[f]} != 0")
            else:
                # Random: rank must be valid for factorization
                if not (1 <= ranks[f] <= min(n_in[f], n_out[f])):
                    raise ValueError(f"Low_rank init=random requires 1 <= rank <= min(n_in, n_out); " f"freq {f}: rank={ranks[f]}, n_in={n_in[f]}, n_out={n_out[f]}")

    def build(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[List[Matrix]]:
        n_f = len(n_in)
        ranks = self._get_ranks(n_in, n_out, config)

        if config.init == "identity":
            # Structured: downsample → repeat
            W_down = utils.create_downsample_matrix(n_in, ranks)
            W_repeat = utils.create_repeat_matrices(ranks, n_out)
            return [W_down, W_repeat]
        else:
            # Random: low-rank factorization A @ B
            if config.seed is not None:
                torch.manual_seed(config.seed)

            A_matrices = []
            B_matrices = []
            for f in range(n_f):
                A = torch.randn(n_in[f], ranks[f]) / (n_in[f] ** 0.5)
                B = torch.randn(ranks[f], n_out[f]) / (ranks[f] ** 0.5)
                A_matrices.append(A)
                B_matrices.append(B)

            return [A_matrices, B_matrices]

    def _get_ranks(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[int]:
        """Get ranks per frequency, auto-deriving via GCD if not specified."""
        n_f = len(n_in)

        if config.rank is not None:
            # Use specified rank for all frequencies
            return [config.rank] * n_f
        else:
            # Auto-derive: use GCD to ensure divisibility for both init strategies
            return [gcd(n_in[f], n_out[f]) for f in range(n_f)]


class RandomMode:
    """Random projection: sparse or dense random connectivity."""

    def validate(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> None:
        n_f = len(n_in)
        if len(n_out) != n_f:
            raise ValueError(f"n_in and n_out must have same length, got {n_f} vs {len(n_out)}")
        if config.sparsity <= 0.0:
            raise ValueError("Random projection requires sparsity > 0.0")

    def build(self, n_in: List[int], n_out: List[int], config: ProjectionConfig) -> List[List[Matrix]]:
        return [utils.create_random_projection(n_in, n_out, sparsity=config.sparsity, seed=config.seed)]


# Mode registry: maps mode name to strategy instance
MODE_REGISTRY: Dict[ProjectionMode, IProjectionMode] = {
    "identity": IdentityMode(),
    "tiling": TilingMode(),
    "low_rank": LowRankMode(),
    "random": RandomMode(),
}


def get_mode_strategy(mode: ProjectionMode) -> IProjectionMode:
    """Get the strategy instance for a given mode."""
    return MODE_REGISTRY[mode]


# ======================================================================================
# PROJECTION MODULE
# ======================================================================================


class Projection(nn.Module):
    """Unified projection supporting 4 modes via strategy pattern.

    Modes:
    - identity: no transformation (n_in == n_out)
    - tiling: repeat-based expansion (init controls structured vs random)
    - low_rank: two-stage compression/expansion (init controls structured vs low-rank)
    - random: sparse/dense random projection

    The mode strategy determines pipeline structure (0/1/2 matrix steps).
    The init parameter controls initialization (structured vs random).
    Activation is applied once at the end.
    """

    def __init__(self, n_in: List[int], n_out: List[int], config: ProjectionConfig):
        super().__init__()
        self._config = config
        self._n_in = n_in
        self._n_out = n_out
        self._n_f = len(n_in)

        # Get mode strategy and validate
        strategy = get_mode_strategy(config.mode)
        strategy.validate(n_in, n_out, config)

        # Build projection pipeline (list of matrix steps)
        pipeline_steps = strategy.build(n_in, n_out, config)

        # Register matrices as parameters
        self._steps = nn.ModuleList()
        for step_matrices in pipeline_steps:
            self._steps.append(nn.ParameterList([nn.Parameter(W, requires_grad=config.learn) for W in step_matrices]))

        # Activation applied once at end
        self._activation = nn.Sigmoid() if config.activation == "sigmoid" else nn.Identity()

    @property
    def n_f(self) -> int:
        """Number of frequency channels."""
        return self._n_f

    @property
    def n_in(self) -> List[int]:
        """Input dimensions per frequency."""
        return self._n_in

    @property
    def n_out(self) -> List[int]:
        """Output dimensions per frequency."""
        return self._n_out

    @property
    def mode(self) -> ProjectionMode:
        """Projection mode."""
        return self._config.mode

    @property
    def W1(self) -> Optional[List[Matrix]]:
        """First-stage matrices (for 2-stage modes like low_rank)."""
        if len(self._steps) == 0:
            return None
        return [W.detach() for W in self._steps[0]]

    @property
    def W2(self) -> Optional[List[Matrix]]:
        """Second-stage matrices (for 2-stage modes like low_rank)."""
        if len(self._steps) < 2:
            return None
        return [W.detach() for W in self._steps[1]]

    def forward(self, x: MultiScaleCode) -> MultiScaleCode:
        """Apply projection pipeline then activation."""
        h: MultiScaleCode = x

        # Apply each step in sequence
        for step_matrices in self._steps:
            h = [h[f] @ W for f, W in enumerate(step_matrices)]

        # Apply final activation
        return [self._activation(h_f) for h_f in h]

    def inverse(self, y: MultiScaleCode) -> MultiScaleCode:
        """Apply inverse projection (transpose), reversing steps.

        Note: activation is not inverted.
        """
        h: MultiScaleCode = y

        # Apply transpose in reverse order
        for step_matrices in reversed(self._steps):
            h = [h[f] @ W.t() for f, W in enumerate(step_matrices)]

        return h
