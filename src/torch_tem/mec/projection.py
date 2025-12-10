"""Projection head for torch_tem package."""

from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .. import utils
from ..types import AbstractLocation, Matrix, MultiScaleCode


class ProjectionParams(Protocol):
    n_f: int  # Total number of frequency modules (grid + optional OVC)
    n_g: List[int]  # Entorhinal abstract location neurons per frequency
    f_extended: List[float]  # Extended frequency list including OVC modules when they are separate


class Projection(nn.Module):
    def __init__(self, params: ProjectionParams):
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g

        # Register downsampling matrices as buffers for automatic device management
        W_down = utils.create_g_downsample(params.n_g)
        for i, matrix in enumerate(W_down):
            self.register_buffer(f"W_down_{i}", matrix)

        # Register expansion matrices as buffer for automatic device management
        W_repeat = utils.create_W_repeat(params.n_g, params.f_extended)
        for i, matrix in enumerate(W_repeat):
            self.register_buffer(f"W_repeat_{i}", matrix)

        # Learnable Laplacian scales (learned as inverse sigmoid)
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(np.log(params.f_extended[f] / (1 - params.f_extended[f])), dtype=torch.float)) for f in range(self.n_f)])

    def downsample(self, g: AbstractLocation) -> MultiScaleCode:
        return [torch.matmul(g[f], getattr(self, f"W_down_{f}").to(g[f].device)) for f in range(self.n_f)]

    def down_inv(self, g_downsampled: MultiScaleCode) -> AbstractLocation:
        batch_size = g_downsampled[0].shape[0]
        n_x_f = [getattr(self, f"W_down_{f}").shape[1] // self.n_g[f] for f in range(self.n_f)]
        return [g_downsampled[f].view(batch_size, self.n_g[f], n_x_f[f]).mean(dim=2) for f in range(self.n_f)]

    def repeat(self, g_downsampled: MultiScaleCode) -> MultiScaleCode:
        return [torch.matmul(g_downsampled[f], getattr(self, f"W_repeat_{f}").to(g_downsampled[f].device)) for f in range(self.n_f)]

    def repeat_inv(self, p: MultiScaleCode) -> MultiScaleCode:
        batch_size = p[0].shape[0]
        n_x_f = [getattr(self, f"W_repeat_{f}").shape[1] // self.n_g[f] for f in range(self.n_f)]
        return [p[f].view(batch_size, self.n_g[f], n_x_f[f]).mean(dim=2) for f in range(self.n_f)]

    def inverse_project(self, p: MultiScaleCode) -> MultiScaleCode:
        p_downsampled = self.repeat_inv(p)
        return self.down_inv(p_downsampled)

    def _get_n_g_subsampled(self, f: int) -> int:
        return getattr(self, f"W_repeat_{f}").shape[0]

    def forward(self, g: AbstractLocation) -> MultiScaleCode:
        g_downsampled = self.downsample(g)
        return self.repeat(g_downsampled)
