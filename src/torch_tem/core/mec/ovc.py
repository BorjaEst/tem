"""MEC OVC (Object Vector Cell) correction from shiny landmark cues."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings

from .utils import fuse_inv_var, resolve_ovc_slice


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
    def settings(self) -> OVCSettings:
        """OVC correction settings."""
        return self._settings

    @property
    def n_ovc(self) -> int:
        """Number of OVC modules."""
        return self._n_ovc

    def forward(self, locations: list[dict], mu: List[Tensor], sigma: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Apply OVC correction to shiny environments.

        Args:
            locations: Per-env metadata (shiny key indicates landmark presence)
            mu: Grid cell means per frequency
            sigma: Grid cell uncertainties per frequency

        Returns:
            mu_corrected: Means with OVC correction applied (sampled if do_sample=True)
            sigma_corrected: Uncertainties with OVC correction applied
        """
        shiny_mask = self._identify_shiny_envs(locations, mu[0].device)
        if shiny_mask is None:  # No shiny envs present
            return mu, sigma

        shiny_input = self._extract_shiny_cues(locations, shiny_mask, mu[0].device)
        mu_shiny, sigma_shiny = self._predict_correction(shiny_input)
        mu_fused, sigma_fused = fuse_inv_var(
            mu,
            sigma,
            mu_shiny,
            sigma_shiny,
            mask=shiny_mask,
            freqs=range(self._ovc_start, self._ovc_start + self._n_ovc),
        )

        if self.settings.do_sample:  # Sample if enabled (legacy behavior)
            mu_corrected = [mu_f + sigma_f * torch.randn_like(mu_f) for mu_f, sigma_f in zip(mu_fused, sigma_fused)]
        else:
            mu_corrected = mu_fused

        return mu_corrected, sigma_fused

    def _identify_shiny_envs(self, locations: list[dict], device: torch.device) -> Tensor | None:
        """Identify which environments have shiny landmarks.

        Args:
            locations: Per-env metadata (shiny key indicates landmark presence)
            device: Device for tensor allocation

        Returns:
            Boolean mask (batch,) for shiny envs, or None if no shiny envs present
        """
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return None
        return torch.tensor(shiny_envs, dtype=torch.bool, device=device)

    def _extract_shiny_cues(self, locations: list[dict], shiny_mask: Tensor, device: torch.device) -> List[Tensor]:
        """Extract shiny cue values as tensor inputs for OVC modules.

        Args:
            locations: Per-env metadata
            shiny_mask: Boolean mask indicating shiny environments
            device: Device for tensor allocation

        Returns:
            List of shiny cue tensors (one per OVC module)
        """
        shiny_vals = [loc["shiny"] for loc in locations if loc.get("shiny") is not None]
        shiny_tensor = torch.as_tensor(shiny_vals, dtype=torch.float32, device=device).unsqueeze(-1)
        return [shiny_tensor] * self._n_ovc

    def _predict_correction(self, shiny_input: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Predict OVC correction mean and uncertainty from shiny landmark cues.

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
