from __future__ import annotations

"""HPC attractor retrieval.

This module implements iterative attractor dynamics used for pattern completion
over a Hebbian memory matrix.

The retrieval operates on a flattened grounded-location code `h` of size
`S = sum(shape)` and iteratively updates subsets of dimensions using a
stage mask schedule. Masks are typically produced by
`torch_tem.utils.update_to_masks` and represent which dimensions are updated at
each stage.
"""

from typing import List, Optional

import numpy as np
import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import AttractorSettings


class AttractorNetwork(nn.Module):
    """Attractor retrieval dynamics (pattern completion) over a memory matrix.

    The network flattens the multi-frequency query code, applies iterative
    updates of the form:

        field = kappa * h + h @ M

    and uses stage masks to control which dimensions update at each stage.
    """

    def __init__(self, shape: List[int], settings: Optional[AttractorSettings] = None):
        """Initialize the attractor.

        Args:
            shape: Feature sizes per frequency module.
            settings: Attractor settings. If `None`, defaults are used.
        """
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._settings = settings = settings or AttractorSettings()
        self._activation_fn = utils.activation_from_str(settings.activation)

    @property
    def settings(self) -> AttractorSettings:
        """Return attractor settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return per-frequency feature sizes."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return number of frequency modules."""
        return self._n_freq

    def forward(self, p_query: List[Tensor], M: Tensor, masks: Optional[List[Tensor]] = None) -> List[Tensor]:
        """Run attractor retrieval.

        Args:
            p_query: Query grounded-location code (multi-scale), one tensor per
                frequency module with shape `(B, shape[f])`.
            M: Hebbian memory matrix of shape `(B, S, S)` where `S = sum(shape)`.
            masks: Optional stage masks used to gate which dimensions update at
                each iteration stage. Each mask is expected to be broadcastable
                to `h` (shape `(B, S)`).

        Returns:
            Retrieved grounded-location code (multi-scale), where each returned
            tensor has shape `(B, shape[f])`.
        """
        # Flatten query grounded locations across frequency modules.
        p, kappa = torch.cat(p_query, dim=1), self.settings.kappa
        h = self.activation(p)

        # Ensure dtype consistency for numerical stability.
        masks = [m.to(dtype=h.dtype) for m in masks]
        M = M.to(dtype=h.dtype)

        for mask in masks:
            field = kappa * h + (h.unsqueeze(1) @ M).squeeze(1)
            h = (1 - mask) * h + mask * self.activation(field)

        # Re-split the grounded location into frequency modules.
        return torch.split(h, split_size_or_sections=self.shape, dim=1)

    def activation(self, p: Tensor) -> Tensor:
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)
