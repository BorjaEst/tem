"""MEC Projection: Abstract location to hippocampal input.

Projects abstract location (grid cells) to hippocampal input space using
learned downsampling and expansion matrices. This enables the hippocampus
to form conjunctive place cell representations (p = g ⊗ x).

The projection pipeline:
1. Downsample: g → g_downsampled (dimensionality reduction)
2. Repeat/Expand: g_downsampled → g_ (hippocampal input space)

This module also provides inverse operations for memory-based inference.
"""

from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_tem import utils
from torch_tem.types import AbstractLocation, Matrix, MultiScaleCode


class ProjectionParams(Protocol):
    """Protocol for Projection initialization parameters.

    Attributes:
        n_f: Total number of frequency modules (grid + optional OVC).
        n_g: Entorhinal abstract location neurons per frequency.
        n_g_subsampled: Subsampled abstract location dimensions per frequency.
        n_x_f: Sensory dimensions per frequency (for hippocampal expansion).
        f_extended: Extended frequency list including OVC modules when separate.
    """

    n_f: int
    n_g: List[int]
    n_g_subsampled: List[int]
    n_x_f: List[int]
    f_extended: List[float]


class Projection(nn.Module):
    """MEC projection head for abstract location to hippocampal input.

    Projects grid cell representations to hippocampal input space through
    learned downsampling and expansion transformations. Each frequency module
    has its own projection pathway with learnable Laplacian scales.

    Args:
        params: Configuration with n_f, n_g, n_g_subsampled, n_x_f, f_extended.
    """

    def __init__(self, params: ProjectionParams):
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g

        # Register downsampling matrices as buffers for automatic device management
        W_down = utils.create_g_downsample(params.n_g, params.n_g_subsampled)
        for i, matrix in enumerate(W_down):
            self.register_buffer(f"W_down_{i}", matrix)

        # Register expansion matrices as buffer for automatic device management
        W_repeat = utils.create_repeat_matrices(params.n_g_subsampled, params.n_x_f)
        for i, matrix in enumerate(W_repeat):
            self.register_buffer(f"W_repeat_{i}", matrix)

        # Learnable Laplacian scales (learned as inverse sigmoid)
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(np.log(params.f_extended[f] / (1 - params.f_extended[f])), dtype=torch.float)) for f in range(self.n_f)])

    def downsample(self, g: AbstractLocation) -> MultiScaleCode:
        """Downsample abstract location for dimensionality reduction.

        Args:
            g: Abstract location List[n_f] of (batch, n_g[f]).

        Returns:
            Downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).
        """
        return [torch.matmul(g[f], getattr(self, f"W_down_{f}").to(g[f].device)) for f in range(self.n_f)]

    def down_inv(self, g_downsampled: MultiScaleCode) -> AbstractLocation:
        """Inverse downsample operation for memory-based inference.

        Args:
            g_downsampled: Downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).

        Returns:
            Reconstructed abstract location List[n_f] of (batch, n_g[f]).
        """
        batch_size = g_downsampled[0].shape[0]
        n_x_f = [getattr(self, f"W_down_{f}").shape[1] // self.n_g[f] for f in range(self.n_f)]
        return [g_downsampled[f].view(batch_size, self.n_g[f], n_x_f[f]).mean(dim=2) for f in range(self.n_f)]

    def repeat(self, g_downsampled: MultiScaleCode) -> MultiScaleCode:
        """Expand downsampled abstract location to hippocampal input space.

        Args:
            g_downsampled: Downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).

        Returns:
            Expanded hippocampal input List[n_f] of (batch, n_p[f]).
        """
        return [torch.matmul(g_downsampled[f], getattr(self, f"W_repeat_{f}").to(g_downsampled[f].device)) for f in range(self.n_f)]

    def repeat_inv(self, p: MultiScaleCode) -> MultiScaleCode:
        """Inverse expansion operation for memory-based inference.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Reconstructed downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).
        """
        batch_size = p[0].shape[0]
        n_x_f = [getattr(self, f"W_repeat_{f}").shape[1] // self.n_g[f] for f in range(self.n_f)]
        return [p[f].view(batch_size, self.n_g[f], n_x_f[f]).mean(dim=2) for f in range(self.n_f)]

    def inverse_project(self, p: MultiScaleCode) -> MultiScaleCode:
        """Full inverse projection from hippocampal place cells to abstract location.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Reconstructed abstract location List[n_f] of (batch, n_g[f]).
        """
        p_downsampled = self.repeat_inv(p)
        return self.down_inv(p_downsampled)

    def _get_n_g_subsampled(self, f: int) -> int:
        """Get downsampled dimension for frequency module f.

        Args:
            f: Frequency module index.

        Returns:
            Downsampled dimension for frequency f.
        """
        return getattr(self, f"W_repeat_{f}").shape[0]

    def forward(self, g: AbstractLocation) -> MultiScaleCode:
        """Project abstract location to hippocampal input space.

        Args:
            g: Abstract location List[n_f] of (batch, n_g[f]).

        Returns:
            Hippocampal input List[n_f] of (batch, n_p[f]).
        """
        g_downsampled = self.downsample(g)
        return self.repeat(g_downsampled)
