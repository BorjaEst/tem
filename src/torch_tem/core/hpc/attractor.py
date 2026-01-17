from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import AttractorSettings


class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics (pattern completion) over a memory matrix."""

    def __init__(self, shape: List[int], settings: AttractorSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._settings = settings

    @property
    def settings(self) -> AttractorSettings:
        return self._settings

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def n_freq(self) -> int:
        return len(self._shape)

    @property
    def n_iters(self) -> int:
        return self.settings.n_iters

    def forward(self, p_query: List[Tensor], M: Tensor, retrieve_it_mask: Optional[List[Tensor]] = None) -> List[Tensor]:
        # Start by flattening query grounded locations across frequency modules
        h_t = torch.cat(p_query, dim=1)
        h_t = self._activate(h_t)

        # If not specified: initialise mask as all 1s
        if retrieve_it_mask is None:
            retrieve_it_mask = [torch.ones(sum(self.shape), device=h_t.device, dtype=h_t.dtype) for _ in range(self.n_iters)]
        else:
            retrieve_it_mask = [m.to(device=h_t.device, dtype=h_t.dtype) for m in retrieve_it_mask]

        # Allow legacy callers to provide longer masks (e.g. per-frequency schedules).
        if len(retrieve_it_mask) < self.n_iters:
            pad = [torch.ones(sum(self.shape), device=h_t.device, dtype=h_t.dtype) for _ in range(self.n_iters - len(retrieve_it_mask))]
            retrieve_it_mask = list(retrieve_it_mask) + pad
        elif len(retrieve_it_mask) > self.n_iters:
            retrieve_it_mask = list(retrieve_it_mask[: self.n_iters])

        for tau in range(self.n_iters):
            # Apply one iteration of attractor dynamics only where mask==1
            update = self._activate(self.settings.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M)))
            h_t = (1 - retrieve_it_mask[tau]) * h_t + retrieve_it_mask[tau] * update

        # Re-cast the grounded location into different frequency modules
        n_p = np.cumsum(np.concatenate(([0], self.shape)))
        return [h_t[:, n_p[f] : n_p[f + 1]] for f in range(self.n_freq)]

    def _activate(self, p: Tensor) -> Tensor:
        return utils.leaky_relu(torch.clamp(p, min=-1, max=1))
