"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.settings import GridSettings

# from torch_tem.settings import GridSettings
from torch_tem.types import Transition


class GridModel(nn.Module):
    """Grid cell path integration module.

    Handles action-driven transition dynamics for grid cells only.
    Does not include OVC logic or full MEC state management.
    """

    def __init__(self, n_a: int, n_g: List[int], settings: GridSettings, f_init: Optional[List[float]] = None):
        super().__init__()
        self._settings = settings  # Protected to avoid modification
        self.n_f = n_f = len(n_g)  # Number of grid cell frequencies
        self.n_g = n_g

        alpha_freq = f_init if f_init is not None else _alpha_init(settings, len(n_g))
        self.g_connections = g_conn = grid_connections(alpha_freq)

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=settings.g_init_std)
        self.g_init_mean = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n_g[f]), dtype=torch.float32)) for f in range(n_f)])
        self.g_init_logstd = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n_g[f]), dtype=torch.float32)) for f in range(n_f)])

        # Transition weights (action-conditioned)
        self.MLP_D_a = MLP(
            in_dim=[n_a for _ in range(n_f)],  # Multiplex through all frequencies
            out_dim=[sum(n_g[fb] for fb in range(n_f) if g_conn[fa][fb]) * n_g[fa] for fa in range(n_f)],
            activation=[torch.tanh, None],
            hidden_dim=[settings.n_hidden for _ in range(n_f)],
            bias=[True, False],
        )
        self.MLP_D_a.set_weights(1, 0.0)

        # Non-directional transition weights (used for shiny generative branch)
        f_no_a = lambda f_to: torch.zeros(sum(n_g[f_from] for f_from in range(n_f) if g_conn[f_to][f_from]) * n_g[f_to])
        self.D_no_a = nn.ParameterList([nn.Parameter(f_no_a(f_to)) for f_to in range(n_f)])

        # Transition uncertainty model
        self.MLP_sigma_g_path = MLP(n_g, n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    # ---------------------------------------------------------------------
    # Public API (compatible with model.py)
    # ---------------------------------------------------------------------

    def get_g_init(self, batch_size: int, device: torch.device) -> Transition:
        """Return initial grid cell activations as (mean, uncertainty) Transition."""
        mean = [self.g_init_mean[f].unsqueeze(0).expand(batch_size, -1).to(device) for f in range(self.n_f)]
        uncertainty = [torch.exp(self.g_init_logstd[f]).unsqueeze(0).expand(batch_size, -1).to(device) for f in range(self.n_f)]
        return Transition(mean=mean, uncertainty=uncertainty)

    def forward(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> Tuple[List[Tensor], Transition]:
        """Return the transition distribution (mu, sigma) before sampling.

        Args:
            a: One-hot encoded actions (B, n_a)
            g: Current grid cell activations (previous g)
            no_direc: Per-batch boolean mask for non-directional transitions

        Returns:
            Transition with mean and uncertainty
        """
        mu = self.g_mean(a, g, no_direc=no_direc)
        sigma = self.g_uncertainty(g)

        # Sample g from (mu, sigma) if enabled (legacy behavior)
        g_path = self.sample_g(mu, sigma)

        # if ANY shiny env exists, recompute g_gen for ALL envs from transitioned state
        shiny_envs = no_direc if no_direc is not None else [False] * a.size(0)
        g_gen = self.generate_g(a, g_path.mean, shiny_envs)

        return g_gen, g_path

    # ---------------------------------------------------------------------
    # Mean / uncertainty (legacy f_mu_g_path / f_sigma_g_path)
    # ---------------------------------------------------------------------

    def g_mean(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> List[Tensor]:
        """Compute transition mean: g_next = g + action_delta."""

        mats = self.transition_matrices(a, no_direc)

        g_in = [torch.cat([g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1).unsqueeze(1) for f_to in range(self.n_f)]
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        g_next = [g_f + delta_f for g_f, delta_f in zip(g, delta)]

        # Clamp activations for stability
        return self.g_clamp(g_next)

    def g_uncertainty(self, g: List[Tensor]) -> List[Tensor]:
        """Compute transition uncertainty from current state."""
        return self.MLP_sigma_g_path(g)

    # ---------------------------------------------------------------------
    # Helper functions (small + explicit)
    # ---------------------------------------------------------------------

    def sample_g(self, mu, sigma) -> Transition:
        """Sample g from (mu, sigma) if enabled (legacy behavior)."""
        if not self._settings.do_sample:
            return Transition(mean=mu, uncertainty=sigma)
        mu = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu, sigma)]
        return Transition(mean=mu, uncertainty=sigma)

    def generate_g(self, a: Tensor, g_transitioned: List[Tensor], shiny_envs: list[bool]) -> List[Tensor]:
        """Compute g_gen for generative pathway (legacy shiny behavior).

        Legacy rule: if ANY env has shiny objects, recompute g_gen for ALL envs
        using non-directional transition weights (per-env mask applied via no_direc).
        Otherwise, g_gen = g_transitioned (the sampled/mean path integration result).
        """
        if any(shiny_envs):
            return self.g_mean(a, g_transitioned, no_direc=shiny_envs)
        return g_transitioned

    def transition_matrices(self, a: Tensor, no_direc: list[bool]) -> List[Tensor]:
        """Compute per-frequency transition matrices, applying no-direction rows."""
        d_flat = self.MLP_D_a([a for _ in range(self.n_f)])
        if no_direc is None:
            no_direc = [False] * a.shape[0]  # batch_size

        no_direc_mask = torch.tensor(no_direc, device=a.device, dtype=torch.bool)
        if torch.any(no_direc_mask):
            for f in range(self.n_f):
                d_no_a = self.D_no_a[f].unsqueeze(0).expand_as(d_flat[f])
                d_flat[f] = torch.where(no_direc_mask.unsqueeze(1), d_no_a, d_flat[f])

        mats: List[Tensor] = []
        for f_to in range(self.n_f):
            in_dim = sum(self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from])
            mats.append(d_flat[f_to].reshape(-1, in_dim, self.n_g[f_to]))
        return mats

    def g_clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations to [-1, 1] for stability."""
        return [torch.clamp(g_f, min=-1, max=1) for g_f in g]


def _alpha_init(settings: GridSettings, n_f: int) -> List[float]:
    """Initialize temporal filtering factors based on desired time constants.

    Returns frequencies in [0, 1] range (NOT logit-transformed).
    The calling code will apply the logit transform.
    """
    if settings.frequencies_init == "linear":  # Linearly spaced time constants between min and max
        return np.linspace(0.9, 0.1, n_f).tolist()
    raise ValueError(f"Unknown frequencies_init method: {settings.frequencies_init}")


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
