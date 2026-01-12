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


class MECModel(nn.Module):
    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._settings = settings

        # Store for backward compatibility with methods that reference self.n_g
        self._n_a = n_a
        self._shape = shape

        # Initialize GridModel (path integration for ALL modules)
        self.grid = GridModel(n_a, n_p, shape, f_init, settings=settings.grid_cells)

        # Initialize OVCModel (shiny landmark heads).
        # OVCModel is responsible for selecting which modules are OVC based on settings.ovc_cells.
        self.ovc = OVCModel(shape, f_init, settings=settings.ovc_cells)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MECState:
        """Initialize MEC state with prior grid cell activations."""
        g_init = self.grid.g_init(batch_size, device)
        ovc = None  # TODO: Initialize OVC state if needed
        return MECState(g=g_init.mean, uncertainty=g_init.uncertainty, ovc=ovc)

    def set_runtime(self, *, p2g_scale_offset: float):
        """Update runtime hyperparameters for MEC module."""
        self.grid.set_runtime(p2g_scale_offset=p2g_scale_offset)

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
        g_gen, transition = self.grid.path_integrate(a, state.g, no_direc=no_direc)

        return g_gen, MECState(g=transition.mean, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Infer abstract location from grounded location and path integration.

        Orchestrates:
        1. GridModel.infer_from_memory(): memory-cued abstract location
        2. Precision-weighted fusion of path integration + memory cues
        3. OVCModel.fuse_shiny(): shiny landmark correction for OVC modules
        4. Sampling or mean extraction

        Args:
            p_x: Grounded location (place cells)
            locations: Per-environment location dicts
            state: Current MEC state (g_path, uncertainty)

        Returns:
            Tuple of (g_inf, updated MECState)
        """
        # Step 1: Infer from memory (Grid responsibility)
        mu_g_mem, sigma_g_mem = self.grid.infer_from_memory(p_x, state.g)

        # Step 2: Fuse path integration with memory cues
        mu_g, sigma_g = [], []
        for f in range(self.n_freq):
            mu, sigma = utils.inv_var_weight([state.g[f], mu_g_mem[f]], [state.uncertainty[f], sigma_g_mem[f]])
            mu_g.append(mu)
            sigma_g.append(sigma)

        # Step 3: Apply shiny correction (OVC responsibility)
        mu_g, sigma_g = self.ovc.fuse_shiny(mu_g, sigma_g, locations, self.n_freq)

        # Step 4: Sample or take mean
        if self._settings.grid_cells.do_sample:
            g_inf = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_g, sigma_g)]
        else:
            g_inf = mu_g

        return g_inf, MECState(g=g_inf, ovc=state.ovc, uncertainty=sigma_g)


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
