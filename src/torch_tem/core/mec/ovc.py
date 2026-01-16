"""MEC OVC (Object Vector Cell) correction from shiny landmark cues."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings

from .utils import resolve_ovc_slice


class OVCCorrection(nn.Module):
    """OVC landmark correction: fuse shiny object cues into selected frequency modules.

    Processes scalar shiny cues to predict grid cell corrections, then fuses them
    into only the designated OVC modules using precision weighting.
    """

    def __init__(self, n_g: List[int], settings: OVCSettings):
        super().__init__()
        self._settings = settings
        self._ovc_start, self._n_ovc = resolve_ovc_slice(len(n_g), settings.n_freq)
        ovc_out_sizes = n_g[self._ovc_start : self._ovc_start + self._n_ovc]
        hidden_dim = [settings.hidden_dim] * self._n_ovc

        # Shiny cue → mean and uncertainty
        self.MLP_mu_g_shiny = MLP([1] * self._n_ovc, ovc_out_sizes, [torch.relu, None], hidden_dim)
        self.MLP_sigma_g_shiny = MLP([1] * self._n_ovc, ovc_out_sizes, [torch.relu, torch.exp], hidden_dim)

    @property
    def enabled(self) -> bool:
        """Whether OVC correction is enabled."""
        return self._n_ovc > 0

    @property
    def settings(self) -> OVCSettings:
        """OVC correction settings."""
        return self._settings

    def forward(self, locations: list[dict], mu: List[Tensor], sigma: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply OVC correction to shiny environments.

        Args:
            locations: Per-env metadata (shiny key indicates landmark presence)
            mu: Grid cell means per frequency (will be modified in-place for shiny envs)
            sigma: Grid cell uncertainties per frequency

        Returns:
            mu_corrected: Means with OVC correction applied
            sigma_corrected: Uncertainties with OVC correction applied
        """
        if not self.enabled:
            return mu, sigma

        # Identify shiny environments
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return mu, sigma

        device = mu[0].device
        shiny_mask = torch.tensor(shiny_envs, dtype=torch.bool, device=device)

        # Extract shiny cue values (scalar per env in legacy)
        shiny_vals = [loc["shiny"] for loc in locations if loc.get("shiny") is not None]
        shiny_tensor = torch.as_tensor(shiny_vals, dtype=torch.float32, device=device).unsqueeze(-1)

        # Predict OVC correction from shiny cues
        mu_shiny, sigma_shiny = self._estimate_shiny([shiny_tensor] * self._n_ovc)

        # Fuse into OVC modules only (in-place modification)
        mu_out, sigma_out = list(mu), list(sigma)
        for f in range(self._ovc_start, self._ovc_start + self._n_ovc):
            f_ovc = f - self._ovc_start
            mu_fused, sigma_fused = utils.inv_var_weight(
                [mu[f][shiny_mask], mu_shiny[f_ovc]],
                [sigma[f][shiny_mask], sigma_shiny[f_ovc]],
            )
            # Write back fused values (only for shiny environments)
            mu_out[f] = mu[f].clone()
            sigma_out[f] = sigma[f].clone()
            mu_out[f][shiny_mask] = mu_fused
            sigma_out[f][shiny_mask] = sigma_fused

        return mu_out, sigma_out

    def _estimate_shiny(self, shiny_input: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Estimate OVC correction from shiny landmark cues.

        Args:
            shiny_input: List of shiny cue tensors (one per OVC module)

        Returns:
            mu_g_shiny: Mean predictions for OVC modules
            sigma_g_shiny: Uncertainty predictions for OVC modules
        """
        # Predict mean with legacy nonlinearity (abs → leaky_relu)
        mu_g = [torch.abs(mu) for mu in self.MLP_mu_g_shiny(shiny_input)]
        mu_g_shiny = [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in mu_g]

        # Predict uncertainty
        sigma_g_shiny = self.MLP_sigma_g_shiny(shiny_input)

        return mu_g_shiny, sigma_g_shiny
