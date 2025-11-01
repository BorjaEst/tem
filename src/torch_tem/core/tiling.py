"""Sensory projection for torch_tem package."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ..config.facets import SensoryProjectionParams


class SensoryProjection(nn.Module):
    """Projects normalized sensory input to p-space for memory indexing.

    This component prepares sensory observations for outer product computation
    with abstract locations by applying W_tile transformation and learnable
    per-frequency weighting.

    Responsibilities:
    - Apply W_tile transformation to normalized sensory input
    - Per-frequency weighting with learnable sigmoid parameters
    - Prepare sensory input for Hebbian memory queries

    Mathematical Operation:
        For each frequency module f:
        x_[f] = sigmoid(w_p[f]) * (x_normalized[f] @ W_tile[f])

    Where:
        - x_normalized[f]: Temporally filtered sensory input [B, n_x_f[f]]
        - W_tile[f]: Tiling matrix for outer product computation [n_x_f[f], n_p[f]]
        - w_p[f]: Learnable frequency-specific weight
        - x_[f]: Projected sensory ready for memory indexing [B, n_p[f]]
    """

    def __init__(self, params: SensoryProjectionParams):
        """Initialize sensory projection.

        Args:
            params: Configuration satisfying SensoryProjectionParams protocol
        """
        super().__init__()
        self.n_f = params.n_f_calculated
        self.n_x_f = params.n_x_f_calculated
        self.W_tile = params.W_tile_calculated

        # Learnable frequency-specific weights (applied with sigmoid)
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def forward(self, x_normalized: List[Tensor]) -> List[Tensor]:
        """Project normalized sensory input to p-space.

        Args:
            x_normalized: Temporally filtered and normalized sensory input
                         [n_f] of [B, n_x_f[f]]

        Returns:
            x_: Projected sensory ready for memory indexing
                [n_f] of [B, n_p[f]]
        """
        # Apply W_tile transformation with learnable sigmoid-gated weights per frequency
        x_ = [torch.sigmoid(self.w_p[f]) * torch.matmul(x_normalized[f], self.W_tile[f].to(x_normalized[f].device)) for f in range(self.n_f)]

        return x_
