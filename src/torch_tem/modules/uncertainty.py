# Probably better to save as a module for transition uncertainty and share with MEC

from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import GroundLocSettings


class Uncertainty(nn.Module):
    def __init__(self, shape: List[int], settings: GroundLocSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(settings.activation)
        self._settings = settings

        # MLP to predict sigma from mu
        self.MLP_sigma_p = MLP(shape, shape, activation=[torch.tanh, None])

    @property
    def settings(self) -> GroundLocSettings:
        return self._settings

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def forward(self, mu: List[Tensor]) -> List[Tensor]:
        sigma = self.MLP_sigma_p(mu)
        return [self.activation(s) for s in sigma]

    def activation(self, p: Tensor) -> Tensor:
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)

    def sample(self, mu: List[Tensor], scale: float = 1.0) -> List[Tensor]:
        transition = utils.Transition(mean=mu, uncertainty=self(mu))
        return utils.sample_diag_gaussian(transition, scale=scale)
