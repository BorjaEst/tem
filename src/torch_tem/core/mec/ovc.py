"""MEC OVC correction.

This module uses shiny landmark cues to compute corrections for a subset of
frequency modules (OVC modules) and fuses them with a reference transition.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings
from torch_tem.types import Transition


class OVCCorrection(nn.Module):
    """Fuse shiny landmark cues into selected frequency modules.

    The correction is applied only to the configured OVC frequency slice and
    combined using inverse-variance weighting.
    """

    def __init__(self, mec_shape: List[int], settings: OVCSettings):
        super().__init__()
        self._ovc_start, self._ovc_count = utils.resolve_ovc_slice(len(mec_shape), settings.n_freq)
        self._ovc_shape = mec_shape[self._ovc_start : self._ovc_start + self._ovc_count]
        self._settings = settings

        # Shiny cue → mean and uncertainty
        hidden_dim = [settings.hidden_dim] * self.n_freq
        self.g_shiny_mlp = MLP([1] * self.n_freq, self.shape, [torch.relu, None], hidden_dim)
        self.uncertainty_mlp = MLP([1] * self.n_freq, self.shape, [torch.relu, torch.exp], hidden_dim)

    @property
    def settings(self) -> OVCSettings:
        """Return the OVC settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return OVC module sizes per frequency."""
        return self._ovc_shape

    @property
    def start(self) -> int:
        """Return the starting frequency index of OVC modules."""
        return self._ovc_start

    @property
    def n_freq(self) -> int:
        """Return the number of OVC frequency modules."""
        return self._ovc_count

    def forward(self, locations: list[dict], transition: Transition) -> Transition:
        """Apply OVC correction to environments with shiny cues.

        Args:
            locations: Per-environment metadata. If `loc.get("shiny")` is not
                `None`, the value is treated as a scalar cue for that batch
                element.
            transition: Reference transition to correct.

        Returns:
            A corrected `Transition`. If no shiny cues are present, returns the
            input transition unchanged.
        """
        shiny_mask = self._identify_shiny_envs(locations, transition.mean[0].device)
        if shiny_mask is None:  # No shiny envs present
            return transition

        shiny_input = self._extract_shiny_cues(locations, shiny_mask, transition.mean[0].device)
        freqs = range(self.start, self.start + self.n_freq)

        correction = self._predict_correction(shiny_input)
        return utils.inv_var_trans(transition, correction, shiny_mask, freqs)

    def _identify_shiny_envs(self, locations: list[dict], device: torch.device) -> Tensor | None:
        """Return a mask selecting environments with shiny cues.

        Args:
            locations: Per-environment metadata.
            device: Device for the returned tensor.

        Returns:
            A boolean mask of shape `(batch,)`, or `None` if no shiny cues are
            present.
        """
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return None
        return torch.tensor(shiny_envs, dtype=torch.bool, device=device)

    def _extract_shiny_cues(self, locations: list[dict], shiny_mask: Tensor, device: torch.device) -> List[Tensor]:
        """Extract shiny cue values as inputs for the OVC MLPs.

        Args:
            locations: Per-environment metadata.
            shiny_mask: Boolean mask indicating which batch items have cues.
            device: Device for returned tensors.

        Returns:
            A list of cue tensors (one per OVC module). Each tensor has shape
            `(n_shiny, 1)`.
        """
        shiny_vals = [loc["shiny"] for loc in locations if loc.get("shiny") is not None]
        shiny_tensor = torch.as_tensor(shiny_vals, dtype=torch.float32, device=device).unsqueeze(-1)
        return [shiny_tensor] * self.n_freq

    def _predict_correction(self, shiny_input: List[Tensor]) -> Transition:
        """Predict mean and uncertainty for the OVC correction.

        Args:
            shiny_input: List of cue tensors (one per OVC module).

        Returns:
            A `Transition` containing OVC correction mean and uncertainty.
        """
        # Predict mean with legacy nonlinearity (abs → leaky_relu)
        mu_g = [torch.abs(mu) for mu in self.g_shiny_mlp(shiny_input)]
        mu_g_shiny = [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in mu_g]

        # Predict uncertainty
        sigma_g_shiny = self.uncertainty_mlp(shiny_input)

        return Transition(mean=mu_g_shiny, uncertainty=sigma_g_shiny)
