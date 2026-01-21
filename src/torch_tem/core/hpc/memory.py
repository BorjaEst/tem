from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import HebbianUpdateSettings


@dataclass
class Runtime:
    eta: float = 0.5
    hebbian_decay: float = 0.9999


class HebbianUpdate(nn.Module):
    """Hebbian write/update logic for the grounded-location memory matrix."""

    def __init__(self, settings: HebbianUpdateSettings):
        super().__init__()
        self._settings = settings
        self._runtime = Runtime()

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def settings(self) -> HebbianUpdateSettings:
        return self._settings

    def forward(self, memory: Tensor, p_inf: Tensor, p_gen: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        update = torch.squeeze(torch.unsqueeze(p_inf + p_gen, 2) @ torch.unsqueeze(p_inf - p_gen, 1))
        update = update * mask.to(dtype=memory.dtype) if mask is not None else update
        return self.clamp_memory(hebbian_decay * memory + eta * update)

    def clamp_memory(self, m: Tensor) -> Tensor:
        return torch.clamp(m, min=self._settings.clamp_min, max=self._settings.clamp_max)
