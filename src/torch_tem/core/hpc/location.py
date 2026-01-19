# Probably better to save as a module for transition uncertainty and share with MEC

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import GroundLocSettings
from torch_tem.types import Transition


class GroundLocation(nn.Module):
    """Distribution over grounded locations p (mean + learned sigma)."""

    def __init__(self, shape: List[int], settings: GroundLocSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(settings.activation)
        self._settings = settings

        if settings.do_sample:  # Uncertainty MLP module
            self.uncertainty_fn = MLP(shape, shape, [torch.tanh, torch.exp], [2 * n for n in shape])
        else:  # If we do not sample, no uncertainty
            self.uncertainty_fn = lambda p: None  # type: ignore

    @property
    def settings(self) -> GroundLocSettings:
        return self._settings

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def forward(self, x_: List[Tensor], g_: List[Tensor]) -> Transition:
        mu_p = [self.activation(g_[f] * x_[f]) for f in range(self.n_freq)]
        sigma_p = self.uncertainty_fn(mu_p)
        return Transition(mean=mu_p, uncertainty=sigma_p)

    def activation(self, p: Tensor) -> Tensor:
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)
