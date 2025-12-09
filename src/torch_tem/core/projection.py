"""Projection head for torch_tem package."""

from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..types import AbstractLocation, Matrix, MultiScaleCode


class ProjectionParams(Protocol):
    """Minimal interface for ProjectionHead.

    Dependencies: n_f, n_g, g_downsample, f_extended
    Complexity: Low (4 parameters)
    """

    n_f: int  # Total number of frequency modules (grid + optional OVC)
    n_g: List[int]  # Entorhinal abstract location neurons per frequency

    @property
    def f_extended(self) -> List[float]:
        """Extended frequency list including OVC modules when they are separate"""
        ...


class ProjectionHead(nn.Module):
    """Projects abstract locations for memory operations.

    Responsibilities:
    - Downsampling: g → g_downsampled (for memory efficiency)
    - Expansion: g_downsampled → place space (via W_repeat)
    - Inverse projection: p → g_downsampled (via W_repeat^T)
    - Complete transformation: g → g_ (downsample + expand)

    Note:
        The alpha parameter is stored for temporal filtering of sensory observations
        (used in x_prev2x), NOT for transforming abstract locations.
    """

    def __init__(self, params: ProjectionParams, g_downsample: List[Matrix], W_repeat: List[Matrix]):
        """Initialize projection head.

        Args:
            params: Configuration satisfying ProjectionParams protocol
            g_downsample: Downsampling matrices [n_f] of [n_g[f], n_g_subsampled[f]]
            W_repeat: Expansion matrices [n_f] of [n_g_subsampled[f], n_p[f]]
        """
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g
        self.g_downsample = g_downsample
        self.W_repeat = W_repeat

        # Learnable Laplacian scales (learned as inverse sigmoid)
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(np.log(params.f_extended[f] / (1 - params.f_extended[f])), dtype=torch.float)) for f in range(self.n_f)])

    def downsample(self, g: AbstractLocation) -> MultiScaleCode:
        """Downsample for memory indexing.

        Args:
            g: Abstract location [n_f] of [B, n_g[f]]

        Returns:
            g_downsampled: Downsampled to [n_f] of [B, n_g_subsampled[f]]
        """
        return [torch.matmul(g[f], self.g_downsample[f].to(g[f].device)) for f in range(self.n_f)]

    def expand(self, g_downsampled: MultiScaleCode) -> MultiScaleCode:
        """Expand downsampled abstract location to place cell space via W_repeat.

        This completes the g→g_ transformation by expanding the downsampled grid cell
        representation to hippocampal place cell dimensions through the W_repeat matrix.
        This matches the legacy g2g_ method: downsample then expand.

        Args:
            g_downsampled: Downsampled abstract location [n_f] of [B, n_g_subsampled[f]]

        Returns:
            g_: Abstract location in place cell space [n_f] of [B, n_p[f]]

        Example:
            >>> g_ = projection.downsample(g)  # [B, 36] → [B, 12]
            >>> g_ = projection.expand(g_)  # [B, 12] → [B, 96]
            >>> p = grounded_inference(g_, x_)  # Use expanded g_ for place cell inference

        Theory:
            The W_repeat matrix (Kronecker product: eye(n_g) ⊗ ones(1, n_x)) enables
            computing the outer product g ⊗ x via element-wise multiplication:
            p = (g @ W_repeat) * (x @ W_tile)
            This prepares g for combination with sensory input in place cell space.
        """
        return [torch.matmul(g_downsampled[f], self.W_repeat[f].to(g_downsampled[f].device)) for f in range(self.n_f)]

    def inverse_project(self, p: MultiScaleCode) -> MultiScaleCode:
        """Project from grounded location (p) back to abstract location space (g).

        This implements the reverse transformation p → g used in the memory inference path,
        where hippocampal place cell patterns are projected back to grid cell space via
        the transpose of the W_repeat matrix (sum over sensory preferences).

        Args:
            p: Grounded location [n_f] of [B, n_p[f]]

        Returns:
            g_downsampled: Projected abstract location [n_f] of [B, n_g_subsampled[f]]

        Note:
            This does NOT apply inverse Laplacian transform. The original TEM stores
            transformed g in memory (via g2g_), so the retrieval from memory already
            includes the transform implicitly. The transpose W_repeat^T provides the
            geometric inverse (sum over sensory dimensions).
        """
        return [torch.matmul(p[f], self.W_repeat[f].t().to(p[f].device)) for f in range(self.n_f)]

    def forward(self, g: AbstractLocation) -> MultiScaleCode:
        """Complete g→g_ transformation: downsample then expand to place cell space.

        This performs the full projection pipeline matching the legacy g2g_ behavior:
        1. Downsample: g → g_downsampled (for memory efficiency)
        2. Expand: g_downsampled → g_ (to place cell dimensions)

        Args:
            g: Raw abstract location [n_f] of [B, n_g[f]]

        Returns:
            g_: Abstract location in place cell space [n_f] of [B, n_p[f]]

        Note:
            The Laplacian transform (tanh(sigmoid(alpha) * g)) was removed as it's
            not part of the legacy g2g_ pathway. The alpha parameter is used for
            temporal filtering of sensory observations (x_prev2x), not for g projection.
        """
        g_downsampled = self.downsample(g)
        return self.expand(g_downsampled)
