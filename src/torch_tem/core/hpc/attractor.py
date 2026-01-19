from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import AttractorSettings


class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics (pattern completion) over a memory matrix."""

    def __init__(self, shape: List[int], settings: Optional[AttractorSettings] = None):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._settings = settings = settings or AttractorSettings()
        self._activation_fn = utils.activation_from_str(settings.activation)

    @property
    def settings(self) -> AttractorSettings:
        return self._settings

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def forward(self, p_query: List[Tensor], M: Tensor, masks: Optional[List[Tensor]] = None) -> List[Tensor]:
        # Start by flattening query grounded locations across frequency modules
        p, kappa = torch.cat(p_query, dim=1), self.settings.kappa
        h = self.activation(p)

        # Ensure dtype consistency for numerical stability (device is automatic via buffers)
        masks = [m.to(dtype=h.dtype) for m in masks]
        M = M.to(dtype=h.dtype)

        for mask in masks:
            field = kappa * h + (h.unsqueeze(1) @ M).squeeze(1)
            h = (1 - mask) * h + mask * self.activation(field)

        # Re-cast the grounded location into different frequency modules
        return torch.split(h, split_size_or_sections=self.shape, dim=1)

    def activation(self, p: Tensor) -> Tensor:
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)
