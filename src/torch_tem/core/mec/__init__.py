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

from torch_tem.core.mec.grid import GridModel
from torch_tem.core.mec.ovc import OVCModel
from torch_tem.settings import MECSettings
from torch_tem.types import Transition


@dataclass
class MECState:
    """State container for MEC dynamics."""

    # Recurrent carry state (posterior grid from previous step)
    g: Optional[List[Tensor]] = None
    uncertainty: Optional[List[Tensor]] = None

    # Optional object-vector cell state (not yet used)
    ovc: Optional[List[Tensor]] = None


class MECModel(nn.Module):
    def __init__(self, n_a: int, shape: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._settings = settings

        # Store for backward compatibility with methods that reference self.n_g
        self._n_a = n_a
        self._shape = shape

        # Initialize GridModel (path integration for ALL modules)
        self.grid = GridModel(n_a, shape, f_init, settings=settings.grid_cells)

        # Initialize OVCModel (shiny landmark heads).
        # OVCModel is responsible for selecting which modules are OVC based on settings.ovc_cells.
        self.ovc = OVCModel(shape, f_init, settings=settings.ovc_cells)

    def init_state(self, batch_size: int, device: torch.device) -> MECState:
        """Initialize MEC state with zeros."""
        gt_0 = self.grid.g_init(batch_size, device)
        return MECState(g=gt_0.mean, uncertainty=gt_0.uncertainty)

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

    def forward(self, a: Optional[Tensor], p_x: Optional[List[Tensor]], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        if p_x is None:
            return self.generative(a, locations, state)
        else:
            return self.inference(p_x, locations, state)

    def generative(self, a: Tensor, locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Compute next MEC state from action-driven transition.

        Args:
            a: One-hot encoded actions (B, n_a). With has_static_action=True,
               action 0 (stand still) is encoded as all-zeros.
            locations: Per-env location dicts (for shiny detection)
            state: Current MEC state

        Returns:
            Tuple of:
            - g_gen: Generated grid cell activations (before memory retrieval)
            - Updated MECState with

        Note:
            Caller is responsible for resetting state.g to g_init at episode boundaries.
            This module always applies transition dynamics from the provided state.
        """
        # Shiny envs use no_direc=True (no action-driven transitions)
        # Only trigger when shiny is explicitly True (not just when key exists)
        no_direc = [loc.get("shiny") is True for loc in locations]
        transition = self.grid(a, state.g, no_direc=no_direc)

        return transition.mean, MECState(g=transition.mean, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        raise NotImplementedError("MEC inference not implemented yet.")


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
