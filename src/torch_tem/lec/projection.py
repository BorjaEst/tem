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

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from .. import utils
from ..types import MultiScaleCode


class ProjectionParams(Protocol):
    """Architecture parameters needed by Projection.

    Attributes:
        n_f: Number of frequency modules
        n_x_f: Filtered sensory dimensions per frequency
        n_p: Hippocampal dimensions per frequency
    """

    n_f: int
    n_x_f: List[int]
    n_p: List[int]


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
        params: Configuration with n_f, n_x_f, n_g_subsampled_combined, n_p.

    Attributes:
        W_tile: Tiling matrices [f] expanding (batch, n_x_f) → (batch, n_p).
        w_p: Learnable frequency weights [f] controlling sensory contribution.
    """

    def __init__(self, params: ProjectionParams):
        """Initialize projection module.

        Note: W_tile matrices are managed by the parent LECModel and passed
        to forward() to ensure consistency and proper device management.

        Args:
            params: Configuration with n_f, n_x_f, n_p
        """
        super().__init__()
        self.n_f = params.n_f
        self.n_x_f = params.n_x_f
        self.n_p = params.n_p

        # Learnable frequency-specific weights (initialized to 1.0)
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def normalize(self, x_f: MultiScaleCode) -> MultiScaleCode:
        """Normalize sensory representations per frequency module.

        Applies a three-step normalization pipeline to prepare filtered sensory
        representations for stable outer product computation in the hippocampus:
        1. Demean: Center each frequency's representation around zero
        2. ReLU: Apply non-linear threshold to enforce non-negativity
        3. L2 normalize: Scale to unit norm for stable gradient flow

        TEM Theory:
            Normalization ensures that conjunctive codes (p = g ⊗ x̃) formed in
            the hippocampus have stable magnitudes regardless of input statistics.
            This prevents frequency modules with larger variance from dominating
            the place cell representations.

        Args:
            x_f: Filtered sensory representations, List[n_f] of (batch, n_x_f[f]).

        Returns:
            Normalized sensory representations, List[n_f] of (batch, n_x_f[f]).
                Each tensor has zero mean, non-negative values, and unit L2 norm.
        """
        return [torch.nn.functional.normalize(torch.relu(x - x.mean(dim=-1, keepdim=True)), p=2, dim=-1) for x in x_f]

    def tiling(self, x_f: MultiScaleCode, W_tile: List[Tensor]) -> MultiScaleCode:
        """Tile normalized sensory to hippocampal dimension with learned weighting.

        Expands the sensory representation from n_x_f to n_p dimensions using
        learned tiling matrices W_tile, then applies frequency-specific weights
        to modulate each frequency's contribution to hippocampal representations.

        TEM Theory:
            The tiling operation prepares sensory information for outer product
            with grid cells: p = g ⊗ x̃, where:
            - g: Grid cell activations from MEC (spatial "where")
            - x̃: Tiled sensory from LEC (sensory "what")
            - p: Place cell conjunctive codes ("where × what")

            The learnable weights w_p allow the model to learn which frequency
            scales are most informative for spatial navigation and memory tasks.

        Mathematical Operation:
            x̃[f] = sigmoid(w_p[f]) * (x_f[f] @ W_tile[f])
            where x̃[f] ∈ ℝ^(batch × n_p[f]) and x_f[f] ∈ ℝ^(batch × n_x_f[f])
            W_tile[f] ∈ ℝ^(n_x_f[f] × n_p[f]) created via Kronecker product

        Args:
            x_f: Normalized sensory representations, List[n_f] of (batch, n_x_f[f])
            W_tile: Tiling matrices, List[n_f] of (n_x_f[f], n_p[f])

        Returns:
            Tiled sensory representations, List[n_f] of (batch, n_p[f]).
                Ready for outer product with grid cells to form place cells.
        """
        return [torch.sigmoid(self.w_p[f]) * x_f[f] @ W_tile[f] for f in range(self.n_f)]

    def forward(self, x_f: MultiScaleCode, W_tile: List[Tensor]) -> MultiScaleCode:
        """Project filtered sensory to hippocampal space: x_f → x̃.

        Pipeline: normalize(x_f) → tile → weight → x̃

        Args:
            x_f: Filtered sensory List[n_f] of (batch, n_x_f[f])
            W_tile: Tiling matrices List[n_f] of (n_x_f[f], n_p[f])

        Returns:
            Projected sensory List[n_f] of (batch, n_p[f])
        """
        x_norm = self.normalize(x_f)
        return self.tiling(x_norm, W_tile)


__all__ = ["ProjectionParams", "Projection"]
