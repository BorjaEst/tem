from __future__ import annotations

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
        # raise NotImplementedError("LowRankModule not yet implemented.")

    def forward(self, z_from: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("LowRankModule forward pass not yet implemented.")

    def inverse(self, z_to: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("LowRankModule inverse pass not yet implemented.")


def _select_module(settings: ProjectionSettings, shape_from: List[int], shape_to: List[int]) -> nn.Module:
    if settings.mode == "tiling":
        return TileModule(shape_from, shape_to, settings)
    if settings.mode == "low_rank":
        return LowRankModule(shape_from, shape_to, settings)
    raise ValueError(f"Unknown projection mode: {settings.projection_mode}")
