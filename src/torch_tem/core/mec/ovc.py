"""OVC (Object Vector Cell) module: shiny landmark -> abstract location heads.

Design goal:
- Keep behavior equivalent to legacy.py's MLP_mu_g_shiny / MLP_sigma_g_shiny
- Keep API compatible with core/model.py via MECModel exposing the MLPs
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings


class OVCModel(nn.Module):
    def __init__(self, shape: List[int], f_init: List[float], settings: OVCSettings):
        super().__init__()
        self._settings = settings

        # Select how many OVC frequency modules to instantiate.
        # Convention: OVC modules occupy the *tail* of the MEC module list.
        # - If settings.n_freq is None: all modules have shiny heads.
        # - If settings.n_freq == 0: no OVC heads.
        # - If settings.n_freq == k: only last k modules have shiny heads.
        n_total = len(shape)
        n_ovc_freq = n_total if settings.n_freq is None else int(settings.n_freq)
        if n_ovc_freq < 0 or n_ovc_freq > n_total:
            raise ValueError(f"OVCSettings.n_freq must be in [0, {n_total}] or None; got {settings.n_freq}")

        self.n_f = n_ovc_freq
        self.n_ovc = shape[-n_ovc_freq:] if n_ovc_freq > 0 else []
        f_init_ovc = f_init[-n_ovc_freq:] if n_ovc_freq > 0 else []

        # OVC modules can also have hierarchical connections (optional)
        # Legacy: if separate_ovc=True, OVC block has its own hierarchy
        self.ovc_connections = ovc_connections(f_init_ovc) if self.n_f > 0 else []

        # Initialize shiny → abstract location MLPs (only if OVC modules exist)
        if self.n_f > 0:
            # Shiny input dimension (legacy: 1 per module, receives stacked shiny coords)
            # The shiny tensor has shape (n_shiny_envs, n_shiny_features, 1)
            # where n_shiny_features = 2 * n_shiny_objects (x,y coords per object)
            n_shiny_in = 1  # Legacy: each module gets scalar input after unsqueeze(-1)
            self.MLP_mu_g_shiny = MLP(
                in_dim=[n_shiny_in] * self.n_f,
                out_dim=self.n_ovc,
                hidden_dim=[20] * self.n_f,
                activation=[torch.relu, None],
            )
            self.MLP_sigma_g_shiny = MLP(
                in_dim=[n_shiny_in] * self.n_f,
                out_dim=self.n_ovc,
                hidden_dim=[20] * self.n_f,
                activation=[torch.relu, torch.exp],
            )

    def shiny_mean(self, shiny: Tensor) -> List[Tensor]:
        """Compute mean of abstract location from shiny landmarks (legacy behavior)."""
        if self.n_f == 0:
            return []
        mu_g = self.MLP_mu_g_shiny(shiny)
        mu_g = [torch.abs(mu) for mu in mu_g]
        return self.clamp_ovc(mu_g)

    def shiny_uncertainty(self, shiny: Tensor) -> List[Tensor]:
        """Compute uncertainty of abstract location from shiny landmarks."""
        if self.n_f == 0:
            return []
        return self.MLP_sigma_g_shiny(shiny)

    def fuse_shiny(
        self,
        mu_g: List[Tensor],
        sigma_g: List[Tensor],
        locations: list[dict],
        n_total_freq: int,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Fuse shiny landmark information into OVC modules.

        Args:
            mu_g: Current mean abstract location (all frequencies)
            sigma_g: Current uncertainty (all frequencies)
            locations: Per-environment location dicts (with 'shiny' key if present)
            n_total_freq: Total number of MEC frequency modules

        Returns:
            Updated (mu_g, sigma_g) with shiny fusion applied to OVC modules
        """
        if self.n_f == 0:
            return mu_g, sigma_g

        # Detect shiny environments
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return mu_g, sigma_g

        # Extract shiny coordinates for environments with shiny objects
        # Shape: (n_shiny_envs, 4) where 4 = 2 objects × 2 coords
        device = mu_g[0].device
        shiny_tensor = torch.stack([torch.tensor(loc["shiny"], dtype=torch.float, device=device) for loc in locations if loc["shiny"] is not None])
        # Legacy format: add extra dimension at the end
        shiny_locations = torch.unsqueeze(shiny_tensor, dim=-1)

        # Compute shiny-derived abstract location
        shiny_input = [shiny_locations for _ in range(self.n_f)]
        mu_g_shiny = self.shiny_mean(shiny_input)
        sigma_g_shiny = self.shiny_uncertainty(shiny_input)

        # Determine which modules are OVC (last self.n_f modules)
        module_start = n_total_freq - self.n_f

        # Fuse shiny information into OVC modules only
        # Import here to avoid circular dependency
        from torch_tem import utils

        shiny_mask = torch.tensor(shiny_envs, dtype=torch.bool, device=device)
        for f in range(module_start, n_total_freq):
            f_ovc = f - module_start
            # Fuse only for shiny environments
            mu_fused, sigma_fused = utils.inv_var_weight(
                [mu_g[f][shiny_mask, :], mu_g_shiny[f_ovc]],
                [sigma_g[f][shiny_mask, :], sigma_g_shiny[f_ovc]],
            )
            # Scatter fused values back into full batch
            mask_expanded = shiny_mask.unsqueeze(-1).expand_as(mu_g[f])
            mu_g[f] = mu_g[f].masked_scatter(mask_expanded, mu_fused)
            sigma_g[f] = sigma_g[f].masked_scatter(mask_expanded, sigma_fused)

        return mu_g, sigma_g

    def clamp_ovc(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp + leaky ReLU (matches legacy f_p-like behavior for OVC)."""
        g = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in g]


def ovc_connections(f_init: List[float]) -> List[List[bool]]:
    """Build hierarchical connections for OVC modules (same logic as grid)."""
    n = len(f_init)
    return [[f_init[f1] <= f_init[f2] for f1 in range(n)] for f2 in range(n)]
