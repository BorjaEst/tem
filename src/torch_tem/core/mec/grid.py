"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import GridSettings

# from torch_tem.settings import GridSettings
from torch_tem.types import Transition


class GridModel(nn.Module):
    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: GridSettings):
        super().__init__()
        self._settings = settings  # Protected to avoid modification

        # Store hyperparameters
        self._n_a = n_a
        self._n_g = n_g = shape
        self.g_connections = g_conn = grid_connections(f_init)
        n_f = len(shape)

        # Runtime values (injected by training loop)
        self.p2g_scale_offset: float = 1.0  # Variance offset scaling for p->g inference

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

        # Generative memory models
        self.MLP_mu_g_mem = MLP(n_p, shape, hidden_dim=[2 * g for g in shape])
        init_w = lambda f: truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=self._settings.g_mem_std)
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(init_w(f), dtype=torch.float32) for f in range(n_f)])
        self.MLP_sigma_g_mem = MLP([2 for _ in n_p], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def g_init(self, batch_size: int, device: torch.device) -> Transition:
        """Return initial grid cell activations as (mean, uncertainty) Transition."""
        mean = [self.g_init_mean[f].unsqueeze(0).expand(batch_size, -1).to(device) for f in range(self.n_freq)]
        uncertainty = [torch.exp(self.g_init_logstd[f]).unsqueeze(0).expand(batch_size, -1).to(device) for f in range(self.n_freq)]
        return Transition(mean=mean, uncertainty=uncertainty)

    def set_runtime(self, *, p2g_scale_offset: float):
        """Update runtime hyperparameters for MEC module."""
        self.p2g_scale_offset = p2g_scale_offset

    @property
    def n_in(self) -> int:
        """Dimensionality of grid cell activations."""
        return self._n_a

    @property
    def shape(self) -> List[int]:
        """Shape of grid cell modules."""
        return self._n_g

    @property
    def n_freq(self) -> int:
        """Number of grid cell frequency modules."""
        return len(self.shape)

    def forward(self, *, _):
        raise NotImplementedError("GridModel forward not implemented. Use path_integrate() or infer_from_memory().")

    def path_integrate(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> Tuple[List[Tensor], Transition]:
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

        # if ANY shiny env exists, recompute g_gen for ALL envs from PREVIOUS g (not transitioned)
        shiny_envs = no_direc if no_direc is not None else [False] * a.size(0)
        if any(shiny_envs):
            g_gen = self.g_mean(a, g, no_direc=shiny_envs)  # from previous g
        else:
            g_gen = g_path.mean  # transitioned

        return g_gen, g_path

    def infer_from_memory(self, p_x: List[Tensor], g: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Infer abstract location from memory-cued grounded location.

        Args:
            p_x: Grounded location (place cells)
            g: Current grid cell state (for error computation)
            p2g_scale_offset: Runtime scaling for variance offset
            p2g_sig_val: Base variance offset value

        Returns:
            Tuple of (mu_g_mem, sigma_g_mem)
        """
        # Compute mean from memory
        mu_g_mem = self.MLP_mu_g_mem(p_x)
        err = utils.squared_error(mu_g_mem, g)

        # Prepare uncertainty input: [vector norm, reconstruction error]
        sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err[f], dim=1)), dim=1) for f, g in enumerate(mu_g_mem)]

        # Clamp for stability
        mu_g_mem = self.g_clamp(mu_g_mem)

        # Infer uncertainty from memory quality
        sigma = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma[f] + self.p2g_scale_offset * self._settings.p2g_sig_val for f in range(self.n_freq)]

        return mu_g_mem, sigma_g_mem

    # ---------------------------------------------------------------------
    # Mean / uncertainty (legacy f_mu_g_path / f_sigma_g_path)
    # ---------------------------------------------------------------------

    def g_mean(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> List[Tensor]:
        """Compute transition mean: g_next = g + action_delta."""

        mats = self.transition_matrices(a, no_direc)

        g_in = [torch.cat([g[f_from] for f_from in range(self.n_freq) if self.g_connections[f_to][f_from]], dim=1).unsqueeze(1) for f_to in range(self.n_freq)]
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

    def transition_matrices(self, a: Tensor, no_direc: list[bool]) -> List[Tensor]:
        """Compute per-frequency transition matrices, applying no-direction rows."""
        d_flat = self.MLP_D_a([a for _ in range(self.n_freq)])
        if no_direc is None:
            no_direc = [False] * a.shape[0]  # batch_size

        no_direc_mask = torch.tensor(no_direc, device=a.device, dtype=torch.bool)
        if torch.any(no_direc_mask):
            for f in range(self.n_freq):
                d_no_a = self.D_no_a[f].unsqueeze(0).expand_as(d_flat[f])
                d_flat[f] = torch.where(no_direc_mask.unsqueeze(1), d_no_a, d_flat[f])

        mats: List[Tensor] = []
        for f_to in range(self.n_freq):
            in_dim = sum(self.shape[f_from] for f_from in range(self.n_freq) if self.g_connections[f_to][f_from])
            mats.append(d_flat[f_to].reshape(-1, in_dim, self.shape[f_to]))
        return mats

    def g_clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations to [-1, 1] for stability."""
        return [torch.clamp(g_f, min=-1, max=1) for g_f in g]


def grid_connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
