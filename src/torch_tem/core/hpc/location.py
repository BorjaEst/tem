"""HPC grounded-location distribution.

This module infers a distribution over grounded location codes (place-cell-like)
from projected sensory features and projected abstract location.

Notes:
    The uncertainty head is conceptually reusable across modules (e.g. MEC).
    If refactoring towards a shared "LocationBelief uncertainty" component, this
    module is a likely consumer.
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import GroundLocSettings
from torch_tem.types import LocationBelief


class GroundLocation(nn.Module):
    def __init__(self, shape: List[int], settings: GroundLocSettings):
        """Initialize grounded-location inference.

        Args:
            shape: Grounded-location feature sizes per frequency module.
            settings: Grounded-location inference settings.
        """
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(settings.activation)
        self._settings = settings

        # Uncertainty from predicted grounded location
        self.uncertainty_mlp = MLP(shape, shape, [torch.tanh, torch.exp], [2 * n for n in shape])

    @property
    def settings(self) -> GroundLocSettings:
        """Return grounded-location inference settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return per-frequency feature sizes."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return number of frequency modules."""
        return self._n_freq

    def forward(self, x_: List[Tensor], g_: List[Tensor]) -> LocationBelief:
        """Infer grounded-location mean and uncertainty.

        Args:
            x_: Projected sensory features per frequency module.
            g_: Projected abstract location per frequency module.

        Returns:
            A `LocationBelief` with:

            - `mean`: inferred grounded-location mean per frequency
            - `uncertainty`: inferred grounded-location uncertainty per frequency
        """
        mu_p = [self.activation(g_[f] * x_[f]) for f in range(self.n_freq)]
        sigma_p = self.uncertainty_mlp(mu_p)
        return LocationBelief(mean=mu_p, uncertainty=sigma_p)

    def activation(self, p: Tensor) -> Tensor:
        """Apply the configured activation with clamping."""
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)
