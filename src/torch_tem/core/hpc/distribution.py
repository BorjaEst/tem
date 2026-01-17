from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem.modules import MLP


class GroundedLocationDistribution(nn.Module):
    """Distribution over grounded locations p (mean + learned sigma)."""

    def __init__(self, *, shape: List[int]):
        super().__init__()
        self._shape = list(shape)
        self._n_freq = len(shape)
        self.MLP_sigma_p = MLP(shape, shape, activation=[torch.tanh, torch.exp])

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def sigma(self, mu_p: List[Tensor]) -> List[Tensor]:
        return self.MLP_sigma_p(mu_p)

    def sample(self, mu_p: List[Tensor]) -> List[Tensor]:
        sigma_p = self.sigma(mu_p)
        return [mu_p[f] + sigma_p[f] * torch.randn_like(sigma_p[f]) for f in range(self.n_freq)]
