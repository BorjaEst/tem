from __future__ import annotations

from typing import List, Literal, Optional, Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import ProjectionSettings


class ProjectionModule(nn.Module):

    def __init__(self, n_z: list[int], n_p: list[int], settings: ProjectionSettings):
        if len(n_z) != len(n_p):
            raise ValueError("n_z and n_p must have the same length.")

        super().__init__()
        self._settings = settings
        self._projection_module = _select_projection_module(settings, n_z, n_p)
        self.set_learning(settings.learnable)

    @property
    def w(self) -> Tensor:
        return self._projection_module.w

    def set_learning(self, enable: bool) -> None:
        """Enable or disable learning of projection weights."""
        for param in self._projection_module.parameters():
            param.requires_grad = enable
        # Note: settings is immutable, don't try to update it

    def set_w(self, w: List[Tensor]) -> None:
        """Set projection weights directly."""
        with torch.no_grad():
            self._projection_module.w.copy_(w)

    def forward(self, z: List[Tensor]) -> List[Tensor]:
        return self._projection_module(z)

    def inverse(self, p: List[Tensor]) -> List[Tensor]:
        return self._projection_module.inverse(p)


class TileModule(nn.Module):

    def __init__(self, n_z: list[int], n_p: list[int], settings: ProjectionSettings):
        super(TileModule, self).__init__()
        self.settings = settings
        self.w = nn.ParameterList([nn.Parameter(w) for w in utils.create_tiling_matrices(n_z, n_p)])

    def forward(self, z: List[Tensor]) -> List[Tensor]:
        return [torch.matmul(z[f], self.w[f]) for f in range(len(z))]

    def inverse(self, p: List[Tensor]) -> List[Tensor]:
        return [torch.matmul(p[f], self.w[f].T) for f in range(len(p))]


class LowRankModule(nn.Module):

    def __init__(self, n_z: list[int], n_p: list[int], settings: ProjectionSettings):
        super(LowRankModule, self).__init__()
        self.settings = settings
        # raise NotImplementedError("LowRankModule not yet implemented.")

    def forward(self, z: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("LowRankModule forward pass not yet implemented.")

    def inverse(self, p: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("LowRankModule inverse pass not yet implemented.")


def _select_projection_module(settings: ProjectionSettings, n_z: int, n_p: int) -> nn.Module:
    if settings.mode == "tiling":
        return TileModule(n_z, n_p, settings)
    if settings.mode == "low_rank":
        return LowRankModule(n_z, n_p, settings)
    raise ValueError(f"Unknown projection mode: {settings.projection_mode}")
