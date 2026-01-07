"""Abstract location inference: fusing path integration with memory.

Combines transition predictions with memory-based corrections using
inverse-variance weighting (Bayesian fusion).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import MECSettings
from torch_tem.types import AbstractLocation, GroundedLocation, Transition


class AbstractLocModel(nn.Module):
    """Abstract location inference via multi-cue fusion.

    Combines:
    1. Path integration (from transition model) - always available
    2. Memory-based correction (from grounded location) - when available

    Uses inverse-variance weighting to combine estimates based on uncertainty.
    """

    def __init__(
        self,
        *,
        n_p: List[int],
        n_g: List[int],
        settings: Optional[MECSettings] = None,
    ):
        """Initialize abstract location inference.

        Args:
            n_p: Grounded location (place cell) dimensions per frequency.
            n_g: Abstract location (grid cell) dimensions per frequency.
            settings: MECSettings (uses defaults if None).
        """
        super().__init__()
        self.settings = settings or MECSettings()
        self._n_f = len(n_g)

        # Memory → Grid MLP (p_x → mu_g_mem)
        self.mlp_mu_g_mem = MLP(
            in_dim=n_p,
            out_dim=n_g,
            activation=(torch.nn.functional.elu, None),
        )

        # Initialize with small weights (legacy parity)
        self.mlp_mu_g_mem.set_weights(from_layer=1, value=[torch.randn(n_g[f], n_g[f]) * self.settings.g_mem_std for f in range(self._n_f)])

        # Memory uncertainty MLP ([||g||^2, error] → sigma_g_mem)
        self.mlp_sigma_g_mem = MLP(
            in_dim=[2] * self._n_f,
            out_dim=n_g,
            activation=(torch.nn.functional.elu, torch.nn.functional.softplus),
        )

    def forward(
        self,
        g_path: Transition,
        p_x: Optional[GroundedLocation] = None,
        sensory_error: Optional[Tensor] = None,
    ) -> AbstractLocation:
        """Infer abstract location by fusing path integration with memory.

        Args:
            g_path: Path integration prediction (mean + uncertainty).
            p_x: Retrieved grounded location from memory (None = generative mode).
            sensory_error: Reconstruction error per batch element (for uncertainty).

        Returns:
            Fused abstract location (list of tensors per frequency).
        """
        # Generative mode: only path integration
        if p_x is None:
            if self.settings.do_sample:
                return [g_path.mean[f] + g_path.uncertainty[f] * torch.randn_like(g_path.mean[f]) for f in range(self._n_f)]
            return g_path.mean

        # Inference mode: compute memory estimate
        mu_g_mem = self.mlp_mu_g_mem(p_x)
        mu_g_mem = [torch.clamp(g, -1.0, 1.0) for g in mu_g_mem]  # Legacy clamp

        # Compute uncertainty from memory quality
        if sensory_error is None:
            sensory_error = torch.zeros(p_x[0].shape[0], device=p_x[0].device)

        quality_indicators = [
            torch.stack(
                [
                    torch.sum(mu_g_mem[f] ** 2, dim=1),  # ||g||^2
                    sensory_error,
                ],
                dim=1,
            )
            for f in range(self._n_f)
        ]
        sigma_g_mem = self.mlp_sigma_g_mem(quality_indicators)

        # Fuse path and memory estimates via inverse-variance weighting
        mu_g, sigma_g = [], []
        for f in range(self._n_f):
            mu, sigma = utils.inv_var_weight(
                [g_path.mean[f], mu_g_mem[f]],
                [g_path.uncertainty[f], sigma_g_mem[f]],
            )
            mu_g.append(mu)
            sigma_g.append(sigma)

        # Sample if configured
        if self.settings.do_sample:
            return [mu_g[f] + sigma_g[f] * torch.randn_like(mu_g[f]) for f in range(self._n_f)]
        return mu_g
