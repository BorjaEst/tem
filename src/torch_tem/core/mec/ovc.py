"""OVC (Object Vector Cell) module: shiny landmark -> abstract location heads.

Design goal:
- Keep behavior equivalent to legacy.py's MLP_mu_g_shiny / MLP_sigma_g_shiny
- Keep API compatible with core/model.py via MECModel exposing the MLPs
"""

from __future__ import annotations

from typing import List, Optional

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.settings import OVCSettings


class OVCModel(nn.Module):
    def __init__(self, shape: List[int], f_init: List[float], settings: OVCSettings):
        super().__init__()
        self._settings = settings

        # Store dimensions
        self.n_ovc = shape
        self.n_f = len(shape)

        # OVC modules can also have hierarchical connections (optional)
        # Legacy: if separate_ovc=True, OVC block has its own hierarchy
        self.ovc_connections = ovc_connections(f_init) if self.n_f > 0 else []

        # Initialize shiny → abstract location MLPs (only if OVC modules exist)
        if self.n_f > 0:
            # Shiny input dimension (legacy: typically 2 * n_shiny for x,y coords)
            # This is hardcoded in legacy; we'll make it configurable via settings if needed
            n_shiny_in = 4  # Legacy default: 2 shiny objects × 2 coords
            self.MLP_mu_g_shiny = MLP(n_in=n_shiny_in, n_out=self.n_ovc, n_h=20, initialisation="xavier", activation="relu")
            self.MLP_sigma_g_shiny = MLP(n_in=n_shiny_in, n_out=self.n_ovc, n_h=20, initialisation="xavier", activation="relu")

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

    def clamp_ovc(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp + leaky ReLU (matches legacy f_p-like behavior for OVC)."""
        g = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in g]


def ovc_connections(self, f_init: List[float]) -> List[List[bool]]:
    """Build hierarchical connections for OVC modules (same logic as grid)."""
    n = len(f_init)
    return [[f_init[f1] <= f_init[f2] for f1 in range(n)] for f2 in range(n)]
