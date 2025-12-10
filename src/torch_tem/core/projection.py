"""Projection head for torch_tem package."""

from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..types import AbstractLocation, Matrix, MultiScaleCode


class ProjectionParams(Protocol):
    n_f: int  # Total number of frequency modules (grid + optional OVC)
    n_g: List[int]  # Entorhinal abstract location neurons per frequency
    f_extended: List[float]  # Extended frequency list including OVC modules when they are separate


class ProjectionHead(nn.Module):
    def __init__(self, params: ProjectionParams, W_repeat: List[Matrix], W_down: List[Matrix]):
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g

        # Validate matrix list lengths
        if len(W_down) != self.n_f:
            raise ValueError(f"W_down must have {self.n_f} matrices, got {len(W_down)}")
        if len(W_repeat) != self.n_f:
            raise ValueError(f"W_repeat must have {self.n_f} matrices, got {len(W_repeat)}")

        # Validate matrix shapes and compatibility
        for f in range(self.n_f):
            # Check W_down output matches W_repeat input
            if W_down[f].shape[1] != W_repeat[f].shape[0]:
                raise ValueError(f"W_down[{f}] output dimension ({W_down[f].shape[1]}) must match " f"W_repeat[{f}] input dimension ({W_repeat[f].shape[0]})")

            # Check W_down input matches n_g[f]
            if W_down[f].shape[0] != params.n_g[f]:
                raise ValueError(f"W_down[{f}] input dimension ({W_down[f].shape[0]}) must match " f"n_g[{f}] ({params.n_g[f]})")

        # Register downsampling matrices as buffers for automatic device management
        for i, matrix in enumerate(W_down):
            self.register_buffer(f"W_down_{i}", matrix)

        # Register expansion matrices as buffer for automatic device management
        for i, matrix in enumerate(W_repeat):
            self.register_buffer(f"W_repeat_{i}", matrix)

        # Learnable Laplacian scales (learned as inverse sigmoid)
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(np.log(params.f_extended[f] / (1 - params.f_extended[f])), dtype=torch.float)) for f in range(self.n_f)])

    def downsample(self, g: AbstractLocation) -> MultiScaleCode:
        return [torch.matmul(g[f], getattr(self, f"W_down_{f}").to(g[f].device)) for f in range(self.n_f)]

    def expand(self, g_downsampled: MultiScaleCode) -> MultiScaleCode:
        return [torch.matmul(g_downsampled[f], getattr(self, f"W_repeat_{f}").to(g_downsampled[f].device)) for f in range(self.n_f)]

    def inverse_project(self, p: MultiScaleCode) -> MultiScaleCode:
        batch_size = p[0].shape[0]
        n_x_f = [getattr(self, f"W_repeat_{f}").shape[1] // self._get_n_g_subsampled(f) for f in range(self.n_f)]
        return [p[f].view(batch_size, self._get_n_g_subsampled(f), n_x_f[f]).mean(dim=2) for f in range(self.n_f)]

    def _get_n_g_subsampled(self, f: int) -> int:
        return getattr(self, f"W_repeat_{f}").shape[0]

    def forward(self, g: AbstractLocation) -> MultiScaleCode:
        g_downsampled = self.downsample(g)
        return self.expand(g_downsampled)
