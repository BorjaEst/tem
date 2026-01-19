from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.settings import LocDistributionSettings


class GroundedLocationDistribution(nn.Module):
    """Distribution over grounded locations p (mean + learned sigma)."""

    def __init__(self, shape: List[int], settings: LocDistributionSettings):
        super().__init__()
        self._shape = list(shape)
        self._n_freq = len(shape)
        self._settings = settings
        self.MLP_sigma_p = MLP(shape, shape, activation=[torch.tanh, torch.exp])

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def sigma(self, mu_p: List[Tensor]) -> List[Tensor]:
        sigma_p = self.MLP_sigma_p(mu_p)

        if self._settings.sigma_activation == "softplus":
            sigma_p = [torch.nn.functional.softplus(s) for s in sigma_p]

        if self._settings.sigma_min > 0:
            sigma_p = [torch.clamp(s, min=self._settings.sigma_min) for s in sigma_p]

        return sigma_p

    def sample(self, mu_p: List[Tensor]) -> List[Tensor]:
        sigma_p = self.sigma(mu_p)
        scale = float(self._settings.noise_scale)
        return [mu_p[f] + scale * sigma_p[f] * torch.randn_like(sigma_p[f]) for f in range(self.n_freq)]
