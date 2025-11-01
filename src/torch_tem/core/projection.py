"""Projection head for torch_tem package."""

from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..config.facets import ProjectionParams


class ProjectionHead(nn.Module):
    """Projects abstract locations to normalized codes and downsamples.

    Responsibilities:
    - Laplacian transform with learnable scales
    - Normalization (clamp, sigmoid)
    - Downsampling for memory indexing
    """

    def __init__(self, params: ProjectionParams):
        """Initialize projection head.

        Args:
            params: Configuration satisfying ProjectionParams protocol
        """
        super().__init__()
        self.n_f = params.n_f_calculated
        self.n_g = params.n_g_calculated
        self.g_downsample = params.g_downsample_calculated

        # Learnable Laplacian scales (learned as inverse sigmoid)
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(np.log(params.f_initial_extended[f] / (1 - params.f_initial_extended[f])), dtype=torch.float)) for f in range(self.n_f)]
        )

    def transform(self, g: List[Tensor]) -> List[Tensor]:
        """Apply Laplacian transform.

        Args:
            g: Abstract location [n_f] of [B, n_g[f]]

        Returns:
            g_transformed: Transformed abstract location
        """
        return [torch.tanh(torch.sigmoid(self.alpha[f]) * g[f]) for f in range(self.n_f)]

    def normalize_g(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp g to [-1, 1].

        Args:
            g: Abstract location

        Returns:
            g_normalized: Clamped to [-1, 1]
        """
        return [torch.clamp(g[f], -1, 1) for f in range(self.n_f)]

    def normalize_p(self, p: List[Tensor]) -> List[Tensor]:
        """Sigmoid normalization for grounded location.

        Args:
            p: Grounded location

        Returns:
            p_normalized: Sigmoid applied
        """
        return [torch.sigmoid(p[f]) for f in range(self.n_f)]

    def downsample(self, g: List[Tensor]) -> List[Tensor]:
        """Downsample for memory indexing.

        Args:
            g: Abstract location [n_f] of [B, n_g[f]]

        Returns:
            g_downsampled: Downsampled to [n_f] of [B, n_g_subsampled[f]]
        """
        return [torch.matmul(g[f], self.g_downsample[f].to(g[f].device)) for f in range(self.n_f)]

    def forward(self, g: List[Tensor]) -> List[Tensor]:
        """Full forward pass: transform and downsample.

        Args:
            g: Raw abstract location

        Returns:
            g_processed: Transformed and downsampled
        """
        g_transformed = self.transform(g)
        g_downsampled = self.downsample(g_transformed)
        return g_downsampled
