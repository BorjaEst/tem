from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import LocDistributionSettings
from torch_tem.types import Transition


class LocationDistribution(nn.Module):
    """Distribution over grounded locations p (mean + learned sigma)."""

    def __init__(self, shape: List[int], settings: LocDistributionSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._activation_fn = utils.activation_from_str(settings.activation)
        self._settings = settings

        # MLP to predict sigma from mu
        self.MLP_sigma_p = MLP(shape, shape, activation=[torch.tanh, torch.exp])

    @property
    def settings(self) -> LocDistributionSettings:
        return self._settings

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def forward(self, x_: List[Tensor], g_: List[Tensor]) -> List[Tensor]:
        return [self.activation(g_[f] * x_[f]) for f in range(self.n_freq)]

    def sample(self, mu_p: List[Tensor]) -> List[Tensor]:
        """Alias for sampling grounded locations.

        Args:
            mu_p: Per-frequency grounded location means.

        Returns:
            Sampled grounded locations.
        """
        sigma_p = self.uncertanty(mu_p)
        transition = Transition(mean=mu_p, uncertainty=sigma_p)
        return utils.sample_diag_gaussian(transition, enabled=True, scale=self.settings.noise_scale)

    def uncertanty(self, mu_p: List[Tensor]) -> List[Tensor]:
        """Compute grounded location uncertainty.

        Args:
            mu_p: Per-frequency grounded location means.

        Returns:
            Per-frequency standard deviations.
        """
        sigma_p = self.MLP_sigma_p(mu_p)
        if self.settings.sigma_activation == "softplus":
            sigma_p = [torch.nn.functional.softplus(s) for s in sigma_p]
        if self.settings.sigma_min > 0:
            sigma_p = [torch.clamp(s, min=self.settings.sigma_min) for s in sigma_p]
        return sigma_p

    def activation(self, p: Tensor) -> Tensor:
        p = torch.clamp(p, min=self.settings.clamp_min, max=self.settings.clamp_max)
        return self._activation_fn(p)
