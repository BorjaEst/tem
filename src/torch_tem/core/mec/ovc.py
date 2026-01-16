"""MEC OVC (Object Vector Cell) correction from shiny landmark cues."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings
from torch_tem.types import Transition


class OVCCorrection(nn.Module):
    """OVC landmark correction: fuse shiny object cues into selected frequency modules.

    Processes scalar shiny cues to predict grid cell corrections, then fuses them
    into only the designated OVC modules using precision weighting.
    """

    def __init__(self, mec_shape: List[int], settings: OVCSettings):
        super().__init__()
        self._ovc_start, self._ovc_count = utils.resolve_ovc_slice(len(mec_shape), settings.n_freq)
        self._ovc_shape = mec_shape[self._ovc_start : self._ovc_start + self._ovc_count]
        self._settings = settings

        # Shiny cue → mean and uncertainty
        hidden_dim = [settings.hidden_dim] * self.n_freq
        self.MLP_mu_g_shiny = MLP([1] * self.n_freq, self.shape, [torch.relu, None], hidden_dim)
        self.MLP_sigma_g_shiny = MLP([1] * self.n_freq, self.shape, [torch.relu, torch.exp], hidden_dim)

    @property
    def settings(self) -> OVCSettings:
        """OVC correction settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Shape of OVC modules."""
        return self._ovc_shape

    @property
    def start(self) -> int:
        """Starting frequency index of OVC modules."""
        return self._ovc_start

    @property
    def n_freq(self) -> int:
        """Number of OVC modules."""
        return self._ovc_count

    def forward(self, locations: list[dict], transition: Transition) -> Transition:
        """Apply OVC correction to shiny environments.

        Args:
            locations: Per-env metadata (shiny key indicates landmark presence)
            mu: Grid cell means per frequency
            sigma: Grid cell uncertainties per frequency

        Returns:
            transition: Corrected grid cell distribution (mean, uncertainty)
        """
        shiny_mask = self._identify_shiny_envs(locations, transition.mean[0].device)
        if shiny_mask is None:  # No shiny envs present
            return transition

        shiny_input = self._extract_shiny_cues(locations, shiny_mask, transition.mean[0].device)
        freqs = range(self.start, self.start + self.n_freq)

        correction = self._predict_correction(shiny_input)
        return utils.inv_var_trans(transition, correction, shiny_mask, freqs)

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
        return [shiny_tensor] * self.n_freq

    def _predict_correction(self, shiny_input: List[Tensor]) -> Transition:
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

        return Transition(mean=mu_g_shiny, uncertainty=sigma_g_shiny)
