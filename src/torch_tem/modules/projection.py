from __future__ import annotations

import math
from typing import List, Literal, Optional, Protocol, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import ProjectionSettings


class TEMComponent(Protocol):
    shape: Sequence[int]


class ProjectionModule(nn.Module):

    def __init__(self, z_from: TEMComponent, z_to: TEMComponent, settings: ProjectionSettings):
        shape_from, shape_to = z_from.shape, z_to.shape
        if len(shape_from) != len(shape_to):
            raise ValueError("Components must have the same number of frequency modules.")

        super().__init__()
        self._settings = settings
        self._projection_module = _select_module(settings, shape_from, shape_to)
        self.set_learning(settings.learnable)

    @property
    def w(self):
        return self._projection_module.w

    def set_learning(self, enable: bool) -> None:
        """Enable or disable learning of projection weights."""
        for param in self._projection_module.parameters():
            param.requires_grad = enable
        # Note: settings is immutable, don't try to update it

    def set_w(self, w: List[Tensor]) -> None:
        """Set projection weights directly."""
        target = self._projection_module.w
        if isinstance(target, nn.ParameterList):
            if len(w) != len(target):
                raise ValueError(f"Expected {len(target)} tensors, got {len(w)}.")
            with torch.no_grad():
                for i, wi in enumerate(w):
                    target[i].copy_(wi)
        else:
            with torch.no_grad():
                target.copy_(w)

    def forward(self, z: List[Tensor]) -> List[Tensor]:
        return self._projection_module(z)

    def inverse(self, p: List[Tensor]) -> List[Tensor]:
        return self._projection_module.inverse(p)


class TileModule(nn.Module):

    def __init__(self, shape_from: List[int], shape_to: List[int], settings: ProjectionSettings):
        super(TileModule, self).__init__()
        self.settings = settings
        w_list = utils.create_tiling_matrices(shape_from, shape_to)
        self.w = nn.ParameterList([nn.Parameter(w) for w in w_list])

    def forward(self, z_from: List[Tensor]) -> List[Tensor]:
        return [torch.matmul(z_from[f], self.w[f]) for f in range(len(z_from))]

    def inverse(self, z_to: List[Tensor]) -> List[Tensor]:
        return [torch.matmul(z_to[f], self.w[f].T) for f in range(len(z_to))]


class LowRankModule(nn.Module):

    def __init__(self, shape_from: List[int], shape_to: List[int], settings: ProjectionSettings):
        super(LowRankModule, self).__init__()
        self.settings = settings
        self._shape_from = list(shape_from)
        self._shape_to = list(shape_to)

        rank_list = _coerce_rank_list(getattr(settings, "rank", None), shape_from, shape_to)

        if settings.init == "identity":
            # Legacy-equivalent: downsample to r dims, then repeat to n_out
            w_down = utils.create_downsample_matrix(shape_from, rank_list)
            w_repeat = utils.create_repeat_matrices(rank_list, shape_to)
        elif settings.init == "random":
            w_down = [torch.randn(n_in, r, dtype=torch.float) / math.sqrt(max(1, n_in)) for n_in, r in zip(shape_from, rank_list)]
            w_repeat = [torch.randn(r, n_out, dtype=torch.float) / math.sqrt(max(1, r)) for r, n_out in zip(rank_list, shape_to)]
        else:
            raise ValueError(f"Unknown init strategy: {settings.init}")

        self.w_down = nn.ParameterList([nn.Parameter(w, requires_grad=settings.learnable) for w in w_down])
        self.w_repeat = nn.ParameterList([nn.Parameter(w, requires_grad=settings.learnable) for w in w_repeat])

    @property
    def w(self) -> List[Tensor]:
        # Full matrices for inspection (not stored)
        return [self.w_down[f] @ self.w_repeat[f] for f in range(len(self.w_down))]

    def forward(self, z_from: List[Tensor]) -> List[Tensor]:
        # z[f]: [B, n_in] -> [B, r] -> [B, n_out]
        return [torch.matmul(torch.matmul(z_from[f], self.w_down[f]), self.w_repeat[f]) for f in range(len(z_from))]

    def inverse(self, z_to: List[Tensor]) -> List[Tensor]:
        # Approx reverse: p -> g_downsampled -> g
        return [torch.matmul(torch.matmul(z_to[f], self.w_repeat[f].T), self.w_down[f].T) for f in range(len(z_to))]


def _select_module(settings: ProjectionSettings, shape_from: List[int], shape_to: List[int]) -> nn.Module:
    if settings.mode == "tiling":
        return TileModule(shape_from, shape_to, settings)
    if settings.mode == "low_rank":
        return LowRankModule(shape_from, shape_to, settings)
    raise ValueError(f"Unknown projection mode: {settings.mode}")


def _coerce_rank_list(rank, shape_from: Sequence[int], shape_to: Sequence[int]) -> List[int]:
    """Normalize rank config into per-frequency list and validate."""
    n_freq = len(shape_from)
    if rank is None:
        rank_list = [math.gcd(int(n_in), int(n_out)) for n_in, n_out in zip(shape_from, shape_to)]
    elif isinstance(rank, int):
        rank_list = [int(rank) for _ in range(n_freq)]
    else:
        rank_list = [int(r) for r in list(rank)]
        if len(rank_list) != n_freq:
            raise ValueError(f"rank must have length {n_freq}, got {len(rank_list)}.")

    for f, (n_in, n_out, r) in enumerate(zip(shape_from, shape_to, rank_list)):
        if r <= 0:
            raise ValueError(f"rank[{f}] must be > 0, got {r}.")
        if r > n_in:
            raise ValueError(f"rank[{f}]={r} cannot exceed input dim {n_in}.")
        if n_out % r != 0:
            raise ValueError(f"rank[{f}]={r} must divide output dim {n_out}.")
    return rank_list
