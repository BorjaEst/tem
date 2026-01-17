from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch import Tensor, nn


def build_p_update_mask(*, shape: List[int], i_attractor: int, f_init: List[float]) -> Tensor:
    """Build the hierarchical connectivity mask for Hebbian memory updates."""

    n_p, n_f = list(shape), len(shape)
    mask = torch.zeros((np.sum(n_p), np.sum(n_p)), dtype=torch.float)
    n_p_cum = np.cumsum(np.concatenate(([0], n_p)))

    for f_from in range(n_f):
        for f_to in range(n_f):
            if f_from >= i_attractor or f_to >= i_attractor:
                # Connection between object vector modules: only allow from low to high frequency
                if f_from >= i_attractor and f_to >= i_attractor:
                    if f_init[f_from] <= f_init[f_to]:
                        mask[n_p_cum[f_from] : n_p_cum[f_from + 1], n_p_cum[f_to] : n_p_cum[f_to + 1]] = 1.0
                # Connection between object vector and normal modules: allow any connections
                else:
                    mask[n_p_cum[f_from] : n_p_cum[f_from + 1], n_p_cum[f_to] : n_p_cum[f_to + 1]] = 1.0
            else:
                # Connection between abstract location modules: only from low to high frequency
                if f_init[f_from] <= f_init[f_to]:
                    mask[n_p_cum[f_from] : n_p_cum[f_from + 1], n_p_cum[f_to] : n_p_cum[f_to + 1]] = 1.0

    return mask


@dataclass
class Runtime:
    eta: float = 0.5
    hebbian_decay: float = 0.9999


class HebbianUpdater(nn.Module):
    """Hebbian write/update logic for the grounded-location memory matrix."""

    def __init__(self, *, shape: List[int], i_attractor: int, f_init: List[float]):
        super().__init__()
        mask = build_p_update_mask(shape=shape, i_attractor=i_attractor, f_init=f_init)
        self.register_buffer("p_update_mask", mask)
        self._runtime = Runtime()

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    def forward(self, M_prev: Tensor, p_inf: Tensor, p_gen: Tensor, *, do_hierarchical_connections: bool = True) -> Tensor:
        # Create new ground memory by outer product of learned vectors
        eta, hebbian_decay = self.runtime.eta, self.runtime.hebbian_decay
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inf + p_gen, 2), torch.unsqueeze(p_inf - p_gen, 1)))
        if do_hierarchical_connections:
            M_new = M_new * self.p_update_mask
        return torch.clamp(hebbian_decay * M_prev + eta * M_new, min=-1, max=1)
