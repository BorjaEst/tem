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
    """OVC heads that map shiny cues -> g mean/sigma for selected modules."""

    def __init__(self, n_ovc: List[int], settings: OVCSettings):
        super().__init__()
        self._settings = settings
        self.n_ovc = list(n_ovc)
        self.n_f = len(self.n_ovc)

        self.MLP_mu_g_shiny = MLP(
            in_dim=[1 for _ in range(self.n_f)],
            out_dim=[dim for dim in self.n_ovc],
            hidden_dim=[2 * dim for dim in self.n_ovc],
        )
        self.MLP_sigma_g_shiny = MLP(
            in_dim=[1 for _ in range(self.n_f)],
            out_dim=[dim for dim in self.n_ovc],
            hidden_dim=[2 * dim for dim in self.n_ovc],
            activation=[torch.tanh, torch.exp],
        )

    def shiny_mean(self, shiny: Tensor) -> List[Tensor]:
        """Compute mean of abstract location from shiny landmarks (legacy behavior)."""
        mu_g = self.MLP_mu_g_shiny(shiny)
        mu_g = [torch.abs(mu) for mu in mu_g]
        return self._clamp_ovc(mu_g)

    def shiny_uncertainty(self, shiny: Tensor) -> List[Tensor]:
        """Compute uncertainty of abstract location from shiny landmarks."""
        return self.MLP_sigma_g_shiny(shiny)

    def _clamp_ovc(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp + leaky ReLU (matches legacy f_p-like behavior for OVC)."""
        g = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in g]
