"""LEC Projection: Sensory to hippocampal projection with tiling.

In TEM theory, the Lateral Entorhinal Cortex (LEC) projects filtered sensory
information to the hippocampus via outer product with grid cells. This module
implements the sensory projection pathway (x_f → x̃) using learned tiling matrices.

The projection creates hippocampal inputs by:
1. Normalizing filtered sensory per frequency: x_norm = f_n(x_f)
2. Tiling to hippocampal dimension: x_tiled = x_norm @ W_tile^T
3. Frequency-specific weighting: x̃ = sigmoid(w_p) * x_tiled

This allows the hippocampus to form conjunctive codes p that bind sensory
information (from LEC) with spatial location (from MEC via grid cells g).
"""

from typing import List

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.types import Matrix, MultiScaleCode


class ProjectionConfig(BaseModel):
    """Projection configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # Learning control
    learn_w_p: bool = Field(default=True, description="If True, frequency weights w_p are learnable; if False, frozen at 1.0")

    # Initialization
    w_p_init: float = Field(default=1.0, ge=0, frozen=True, description="Initial value for frequency weights w_p (before sigmoid)")


class Projection(nn.Module):
    """LEC sensory projection to hippocampal space via learned tiling.

    Projects multi-frequency filtered sensory (x_f) to hippocampal inputs (x̃) that
    can form conjunctive codes with grid cells. Each frequency has a learnable weight
    controlling its contribution to hippocampal representations.

    TEM Theory:
        The hippocampus forms place cells p via outer product: p = g ⊗ x̃
        where g comes from MEC (grid cells) and x̃ comes from LEC (sensory).
        This binding creates "where × what" conjunctive representations.

    Args:
        W_tile: Tiling matrices shared from parent LECModel
        config: Projection configuration parameters
    """

    def __init__(self, W_tile: List[Matrix], config: ProjectionConfig):
        """Initialize projection module.

        Args:
            W_tile: Tiling matrices for projection (managed by parent LECModel).
            config: Projection configuration (learning control, initialization).
        """
        super().__init__()
        self._config = config
        self._W_tile = W_tile

        # Learnable frequency-specific weights (initialized to w_p_init)
        p = [nn.Parameter(torch.tensor(config.w_p_init), requires_grad=config.learn_w_p) for _ in range(self.n_f)]
        self._w_p = nn.ParameterList(p)

    @property
    def n_f(self) -> int:
        """Number of frequency channels."""
        return len(self._W_tile)

    @property
    def n_p(self) -> List[int]:
        """Number of hippocampal place cells per frequency."""
        return [matrix.shape[1] for matrix in self._W_tile]

    def normalize(self, x_f: MultiScaleCode) -> MultiScaleCode:
        """Normalize sensory representations per frequency module."""
        return [self.normalize_fn(x_f_f) for x_f_f in x_f]

    @staticmethod
    def normalize_fn(x: Tensor) -> Tensor:
        """Normalize a single frequency tensor: demean, ReLU, L2 normalize.

        Args:
            x: Input tensor of shape (batch, n_x_f).

        Returns:
            Normalized tensor of shape (batch, n_x_f).
        """
        x_demeaned = x - x.mean(dim=-1, keepdim=True)
        x_relu = torch.relu(x_demeaned)
        x_normalized = torch.nn.functional.normalize(x_relu, p=2, dim=-1)
        return x_normalized

    def set_w_learning(self, learn: bool):
        """Set learning state for frequency weights w_p.

        Args:
            learn: If True, enable gradients; if False, freeze parameters.
        """
        self._config.learn_w_p = learn
        for param in self._w_p:
            param.requires_grad_(learn)

    def tiling(self, x_f: MultiScaleCode) -> MultiScaleCode:
        """Tile normalized sensory to hippocampal dimension with learned weighting.

        Expands the sensory representation from n_x_c to n_p dimensions using
        learned tiling matrices W_tile, then applies frequency-specific weights
        to modulate each frequency's contribution to hippocampal representations.

        Mathematical operation:
            x̃[f] = sigmoid(w_p[f]) * (x_f[f] @ W_tile[f])

        Args:
            x_f: Normalized filtered sensory List[n_f] of (batch, n_x_c).

        Returns:
            Tiled sensory List[n_f] of (batch, n_p[f]).
        """
        w_p, W_tile, n_f = self._w_p, self._W_tile, self.n_f
        return [torch.sigmoid(w_p[f]) * x_f[f] @ W_tile[f] for f in range(n_f)]

    def forward(self, x_f: MultiScaleCode) -> MultiScaleCode:
        """Project filtered sensory to hippocampal space: x_f → x̃.

        Pipeline: normalize(x_f) → tile → weight → x̃

        Args:
            x_f: Filtered sensory List[n_f] of (batch, n_x_c).

        Returns:
            Projected sensory List[n_f] of (batch, n_p[f]).
        """
        x_norm = self.normalize(x_f)
        return self.tiling(x_norm)


__all__ = ["ProjectionConfig", "Projection"]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Projection usage example: Sensory to hippocampal projection.

    Demonstrates how the projection module transforms filtered sensory
    representations into hippocampal space via tiling and weighting.
    """
    print("=" * 80)
    print("Projection Example - Sensory to Hippocampal Space")
    print("=" * 80)

    # Configuration
    n_x_c = 10  # Compressed sensory dimension
    n_p = [96, 80, 64]  # Place cells per frequency
    n_f = len(n_p)
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Compressed dimension: {n_x_c}")
    print(f"  Frequencies: {n_f}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Batch size: {batch_size}")

    # Create tiling matrices (normally from context)
    W_tile = [torch.randn(n_x_c, n_p_f) for n_p_f in n_p]
    print(f"\n✓ Tiling matrices: {[W.shape for W in W_tile]}")

    # Create projection
    config = ProjectionConfig(learn_w_p=True, w_p_init=1.0)
    projection = Projection(W_tile, config)
    print(f"✓ Projection initialized (n_f={projection.n_f})")
    print(f"  Place cells per frequency: {projection.n_p}")

    # Create filtered sensory input
    x_f = [torch.randn(batch_size, n_x_c) for _ in range(n_f)]
    print(f"\n✓ Filtered sensory: {[x.shape for x in x_f]}")

    # Project to hippocampal space
    with torch.no_grad():
        x_projected = projection(x_f)

    print(f"✓ Projected sensory: {[x.shape for x in x_projected]}")

    # Show statistics per frequency
    print(f"\nProjection statistics:")
    for f in range(n_f):
        print(f"  Frequency {f}:")
        print(f"    Input: mean={x_f[f].mean():.4f}, std={x_f[f].std():.4f}")
        print(f"    Output: mean={x_projected[f].mean():.4f}, std={x_projected[f].std():.4f}")
        print(f"    Dimension: {n_x_c} → {n_p[f]} (expansion={n_p[f]/n_x_c:.1f}x)")

    print("\n" + "=" * 80)
    print("TEM Pipeline: x → x_c (Encoder) → x_f (Processor) → x̃ (Projection) → p (HPC)")
    print("=" * 80)
