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
    """State container for MEC dynamics.

    Attributes:
        g_gen: Ancestral prediction for the generative pathway.
        g_path: Path integration prior (mean, uncertainty) for inference.
        g: Infered abstract location.
    """

    g_gen: List[Tensor]
    g_path: Transition
    g: Optional[List[Tensor]] = None
    ovc: Optional[List[Tensor]] = None

    def __post_init__(self):
        """Ensure g is always defined (defaults to g_path.mean)."""
        if self.g is None:
            self.g = list(self.g_path.mean)


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
        """Initialize MEC state with prior grid cell activations."""
        g_init = self.grid.g_init(batch_size, device)
        return MECState(g_gen=list(g_init.mean), g_path=g_init)

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
            New MECState with updated g_gen and g_path.

        Note:
            Caller is responsible for resetting state.g to g_init at episode boundaries.
            This module always applies transition dynamics from the provided state.
        """
        # Shiny envs use no_direc=True (no action-driven transitions)
        no_direc = [loc.get("shiny") is not None for loc in locations]
        g_gen, transition = self.grid(a, state.g, no_direc=no_direc)

        return g_gen, MECState(g_gen=g_gen, g_path=transition)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        raise NotImplementedError("MEC inference not implemented yet.")


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
