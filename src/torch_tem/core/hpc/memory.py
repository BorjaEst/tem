from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import MemorySettings


@dataclass
class Runtime:
    eta: float = 0.5
    hebbian_decay: float = 0.9999


class MemorySystem(nn.Module):
    """Hebbian write/update logic for the grounded-location memory matrix."""

    def __init__(self, shape: List[int], grid_n_freq: int, f_init: List[float], settings: MemorySettings):
        super().__init__()
        self._settings = settings
        mask = utils.make_hebbian_write_mask(grid_n_freq, shape, f_init=f_init)
        self.register_buffer("p_update_mask", mask)
        self._runtime = Runtime()

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def settings(self) -> MemorySettings:
        return self._settings

    def forward(self, M_prev: Tensor, p_inf: Tensor, p_gen: Tensor, *, mask: Optional[Tensor] = None) -> Tensor:
        # Create new ground memory by outer product of learned vectors
        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inf + p_gen, 2), torch.unsqueeze(p_inf - p_gen, 1)))
        if mask is not None:
            M_new = M_new * mask.to(dtype=M_new.dtype)
        return torch.clamp(
            hebbian_decay * M_prev + eta * M_new,
            min=float(self._settings.clamp_min),
            max=float(self._settings.clamp_max),
        )
