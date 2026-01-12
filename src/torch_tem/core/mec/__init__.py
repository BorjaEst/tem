"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.core.mec.grid import GridModel
from torch_tem.core.mec.ovc import OVCModel
from torch_tem.settings import MECSettings


@dataclass
class MECState:
    """State container for MEC dynamics."""

    g: List[Tensor]  # Grid cell activations
    ovc: Optional[List[Tensor]] = None  # OVC activations
    uncertainty: Optional[List[Tensor]] = None  # Grid cell uncertainty

    def detach(self) -> "MECState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return MECState(
            g=[v.detach() for v in self.g] if self.g is not None else None,
            ovc=[v.detach() for v in self.ovc] if self.ovc is not None else None,
            uncertainty=[v.detach() for v in self.uncertainty] if self.uncertainty is not None else None,
        )


@dataclass
class MECRuntime:
    """Runtime values injected by training (not architectural parameters)."""

    p2g_scale_offset: float = 1.0  # Variance offset scaling for p->g inference


class MECModel(nn.Module):
    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._settings = settings

        # Store for backward compatibility with methods that reference self.n_g
        self._n_a = n_a
        self._shape = shape

        # Store runtime values (injected by training loop)
        self.runtime = MECRuntime()

        # Initialize GridModel (path integration for ALL modules)
        self.grid = GridModel(n_a, n_p, shape, f_init, settings=settings.grid_cells)

        # Initialize OVCModel (shiny landmark heads).
        # OVCModel is responsible for selecting which modules are OVC based on settings.ovc_cells.
        self.ovc = OVCModel(shape, f_init, settings=settings.ovc_cells)

    def init_state(self, batch_size: int, device: torch.device) -> MECState:
        """Initialize MEC state with prior grid cell activations."""
        g_init = self.grid.g_init(batch_size, device)
        ovc = None  # TODO: Initialize OVC state if needed
        return MECState(g=g_init.mean, uncertainty=g_init.uncertainty, ovc=ovc)

    def set_runtime(self, *, p2g_scale_offset: float):
        """Update runtime hyperparameters for MEC module."""
        self.runtime.p2g_scale_offset = p2g_scale_offset

    @property
    def n_in(self) -> int:
        """Dimensionality of action input."""
        return self._n_a

    @property
    def shape(self) -> List[int]:
        """Shape of grid cell modules."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Number of grid cell frequency modules."""
        return len(self.shape)

    def forward(self, *, _) -> Tuple[List[Tensor], MECState]:
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(self, a: Tensor, locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Compute next MEC state from action-driven transition.

        Args:
            a: One-hot encoded actions (B, n_a). With has_static_action=True,
               action 0 (stand still) is encoded as all-zeros.
            locations: Per-env location dicts (for shiny detection)
            state: Current MEC state

        Returns:
            New MECState with updated g_gen and g_path.

        Note:
            Caller is responsible for resetting state.g to g_init at episode boundaries.
            This module always applies transition dynamics from the provided state.
        """
        # Shiny envs use no_direc=True (no action-driven transitions)
        no_direc = [loc.get("shiny") is not None for loc in locations]
        g_gen, transition = self.grid(a, state.g, no_direc=no_direc)

        return g_gen, MECState(g=transition.mean, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        mu_g_mem = self.grid.MLP_mu_g_mem(p_x)
        err = sum(utils.squared_error(mu_g_mem, state.g))

        # Prepare uncertainty input: [vector norm, reconstruction error]
        sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]
        # Clamp for stability (recommended by original authors)
        mu_g_mem = self.grid.g_clamp(mu_g_mem)
        # Infer abstract location uncertainty from memory quality
        sigma = self.grid.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma[f] + self.runtime.p2g_scale_offset * self._settings.p2g_sig_val for f in range(self.n_freq)]

        mu_g_path = state.g
        sigma_g_path = state.uncertainty

        # Fuse path integration with memory cues (if provided)
        mu_g, sigma_g = [], []
        for f in range(self.n_freq):
            if mu_g_mem is not None and sigma_g_mem is not None:
                # Precision-weighted fusion
                mu, sigma = utils.inv_var_weight(
                    [mu_g_path[f], mu_g_mem[f]],
                    [sigma_g_path[f], sigma_g_mem[f]],
                )
            else:
                # Path integration only
                mu, sigma = mu_g_path[f], sigma_g_path[f]
            mu_g.append(mu)
            sigma_g.append(sigma)

        # Apply shiny correction (OVC-only modules)
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if any(shiny_envs) and self.ovc.n_f > 0:
            # Extract shiny coordinates for environments with shiny objects
            # Shape: (n_shiny_envs, 4) where 4 = 2 objects × 2 coords
            shiny_tensor = torch.stack([torch.tensor(loc["shiny"], dtype=torch.float, device=mu_g[0].device) for loc in locations if loc["shiny"] is not None])
            # Legacy format: add extra dimension at the end
            shiny_locations = torch.unsqueeze(shiny_tensor, dim=-1)

            # Compute shiny-derived abstract location
            shiny_input = [shiny_locations for _ in range(self.ovc.n_f)]
            mu_g_shiny = self.ovc.shiny_mean(shiny_input)
            sigma_g_shiny = self.ovc.shiny_uncertainty(shiny_input)

            # Determine which modules are OVC (last ovc.n_f modules)
            module_start = self.n_freq - self.ovc.n_f

            # Fuse shiny information into OVC modules only
            shiny_mask = torch.tensor(shiny_envs, dtype=torch.bool, device=mu_g[0].device)
            for f in range(module_start, self.n_freq):
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

        # Sample or take mean (depending on settings)
        if self._settings.grid_cells.do_sample:
            g_inf = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_g, sigma_g)]
        else:
            g_inf = mu_g

        return g_inf, MECState(g=g_inf, ovc=state.ovc, uncertainty=sigma_g)


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
