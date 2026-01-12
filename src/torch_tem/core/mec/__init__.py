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
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import GridSettings, MECSettings, OVCSettings
from torch_tem.types import Transition


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


class GridModel(nn.Module):
    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: GridSettings):
        super().__init__()
        self._settings = settings  # Protected to avoid modification

        # Store hyperparameters
        self._n_a = n_a
        self._n_g = n_g = shape
        self.g_connections = g_conn = connections(f_init)
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
        self.ovc_connections = ovc_conn = connections(f_init_ovc) if self.n_f > 0 else []

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


def connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]
