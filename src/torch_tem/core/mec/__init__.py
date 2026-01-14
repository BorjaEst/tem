"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import MECSettings
from torch_tem.types import Transition

__all__ = ["MECModel", "MECState"]


@dataclass
class MECState:
    """State container for MEC dynamics."""

    cells: List[Tensor]  # MEC cell activations (grid and OVC)
    uncertainty: Optional[List[Tensor]] = None  # Grid cell uncertainty

    def detach(self) -> "MECState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return MECState(
            cells=[v.detach() for v in self.cells] if self.cells is not None else None,
            uncertainty=[v.detach() for v in self.uncertainty] if self.uncertainty is not None else None,
        )


class OVCCueModelBase(nn.Module, ABC):
    """OVC modules that process shiny landmark cues for location correction."""

    def __init__(self, ovc_out_sizes: List[int], settings: MECSettings):
        self.__ovc_sizes = ovc_out_sizes
        n_ovc = len(ovc_out_sizes)

        hidden_dim = [settings.hidden_dim_ovc] * n_ovc
        self.MLP_mu_g_shiny = MLP([1] * n_ovc, ovc_out_sizes, [torch.relu, None], hidden_dim)
        self.MLP_sigma_g_shiny = MLP([1] * n_ovc, ovc_out_sizes, [torch.relu, torch.exp], hidden_dim)

    @property
    def n_ovc_modules(self) -> int:
        """Number of OVC modules (receive shiny correction)."""
        return len(self.__ovc_sizes)

    def shiny_mean(self, shiny: List[Tensor]) -> List[Tensor]:
        """Compute mean of abstract location from shiny landmarks (legacy behavior)."""
        mu_g = [torch.abs(mu) for mu in self.MLP_mu_g_shiny(shiny)]
        return [torch.nn.functional.leaky_relu(g_f, negative_slope=0.1) for g_f in self.clamp(mu_g)]

    def shiny_uncertainty(self, shiny: List[Tensor]) -> List[Tensor]:
        """Compute uncertainty of abstract location from shiny landmarks."""
        return self.MLP_sigma_g_shiny(shiny)

    def estimate_shiny(self, shiny_input: List[Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Estimate OVC correction from shiny landmark cues.

        Args:
            shiny_input: List of shiny cue tensors (one per OVC module)

        Returns:
            (mu_g_shiny, sigma_g_shiny): Mean and uncertainty for OVC modules
        """
        mu_g_shiny = self.shiny_mean(shiny_input)
        sigma_g_shiny = self.shiny_uncertainty(shiny_input)
        return mu_g_shiny, sigma_g_shiny

    @abstractmethod
    def clamp(self, g: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("OVCCueModelBase.clamp must be implemented in derived classes.")


class PathInegratorBase(nn.Module, ABC):
    def __init__(self, n_a: int, n_g: List[int], f_init: List[float], settings: MECSettings):
        # Store hyperparameters
        self.__n_g, n_freq = n_g, len(n_g)
        self.__do_sample = settings.do_sample
        self.g_connections = g_conn = connections(f_init)

        # Transition weights (action-conditioned)
        out_dim = [sum(nb for fb, nb in enumerate(n_g) if g_conn[fa][fb]) * na for fa, na in enumerate(n_g)]
        self.MLP_D_a = MLP([n_a] * n_freq, out_dim, activation=[torch.tanh, None], hidden_dim=[settings.hidden_dim_grid] * n_freq, bias=[True, False])
        self.MLP_D_a.set_weights(1, 0.0)

        # Non-directional transition weights (used for shiny generative branch)
        f_no_a = lambda f_to: torch.zeros(sum(n_g[f_from] for f_from in range(n_freq) if g_conn[f_to][f_from]) * n_g[f_to])
        self.D_no_a = nn.ParameterList([nn.Parameter(f_no_a(f_to)) for f_to in range(n_freq)])
        self.MLP_sigma_g_path = MLP(n_g, n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def path_integrate(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> Tuple[List[Tensor], Transition]:
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
        return g_gen, Transition(mean=g_path.mean, uncertainty=g_path.uncertainty)

    def g_mean(self, a: Tensor, g: List[Tensor], no_direc: list[bool] | None = None) -> List[Tensor]:
        mats, n_freq = self.transition_matrices(a, no_direc), len(self.__n_g)
        g_in = [torch.cat([g[f_from] for f_from in range(n_freq) if self.g_connections[f_to][f_from]], dim=1).unsqueeze(1) for f_to in range(n_freq)]
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        g_next = [g_f + delta_f for g_f, delta_f in zip(g, delta)]
        return self.clamp(g_next)

    def g_uncertainty(self, g: List[Tensor]) -> List[Tensor]:
        return self.MLP_sigma_g_path(g)

    def sample_g(self, mu, sigma) -> Transition:
        if not self.__do_sample:
            return Transition(mean=mu, uncertainty=sigma)
        mu = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu, sigma)]
        return Transition(mean=mu, uncertainty=sigma)

    def transition_matrices(self, a: Tensor, no_direc: list[bool]) -> List[Tensor]:
        n_freq = len(self.__n_g)
        d_flat = self.MLP_D_a([a for _ in range(n_freq)])
        if no_direc is None:
            no_direc = [False] * a.shape[0]  # batch_size

        no_direc_mask = torch.tensor(no_direc, device=a.device, dtype=torch.bool)
        if torch.any(no_direc_mask):
            for f in range(n_freq):
                d_no_a = self.D_no_a[f].unsqueeze(0).expand_as(d_flat[f])
                d_flat[f] = torch.where(no_direc_mask.unsqueeze(1), d_no_a, d_flat[f])

        mats: List[Tensor] = []
        for f_to in range(n_freq):
            in_dim = sum(self.__n_g[f] for f in range(n_freq) if self.g_connections[f_to][f])
            mats.append(d_flat[f_to].reshape(-1, in_dim, self.__n_g[f_to]))
        return mats

    @abstractmethod
    def clamp(self, g: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("PathInegratorBase.clamp must be implemented in derived classes.")


class PathMemoryBase(nn.Module, ABC):
    def __init__(self, n_p: List[int], n_g: List[int], settings: MECSettings):
        # Store hyperparameters
        self.__n_g, n_freq = n_g, len(n_g)
        self.__p2g_sig_val = settings.p2g_sig_val
        self.p2g_scale_offset: float = 1.0  # Variance offset scaling for p->g inference

        # Generative memory models
        self.MLP_mu_g_mem = MLP(n_p, n_g, hidden_dim=[2 * g for g in n_g])
        init_w = lambda f: truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=settings.std_grid_mem)
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(init_w(f), dtype=torch.float32) for f in range(n_freq)])
        self.MLP_sigma_g_mem = MLP([2 for _ in n_p], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def set_runtime(self, *, p2g_scale_offset: float):
        self.p2g_scale_offset = p2g_scale_offset

    def infer_from_memory(self, p_x: List[Tensor], g: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        # Compute mean from memory
        mu_g_mem = self.MLP_mu_g_mem(p_x)
        err = utils.squared_error(mu_g_mem, g)
        # Prepare uncertainty input: [vector norm, reconstruction error]
        sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err[f], dim=1)), dim=1) for f, g in enumerate(mu_g_mem)]
        # Clamp for stability
        mu_g_mem = self.clamp(mu_g_mem)
        # Infer uncertainty from memory quality
        sigma = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma[f] + self.p2g_scale_offset * self.__p2g_sig_val for f, _ in enumerate(self.__n_g)]

        return mu_g_mem, sigma_g_mem

    @abstractmethod
    def clamp(self, g: List[Tensor]) -> List[Tensor]:
        raise NotImplementedError("PathMemoryBase.clamp must be implemented in derived classes.")


class MECModel(PathInegratorBase, PathMemoryBase, OVCCueModelBase):
    def __init__(self, n_a: int, n_p: List[int], n_cells: List[int], f_init: List[float], settings: MECSettings):
        nn.Module.__init__(self)
        self.__n_a, self.__n_cells = n_a, n_cells

        # Determine which modules are OVC (receive shiny landmark correction)
        self._ovc_start, self._n_ovc = resolve_ovc_slice(len(n_cells), settings.n_freq_ovc)
        ovc_out_sizes = n_cells[self._ovc_start : self._ovc_start + self._n_ovc]

        # Initialize all parent classes using cooperative multiple inheritance
        PathInegratorBase.__init__(self, n_a, n_cells, f_init, settings)
        PathMemoryBase.__init__(self, n_p, n_cells, settings)
        OVCCueModelBase.__init__(self, ovc_out_sizes, settings)
        self._settings = settings

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=settings.std_grid_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in n_cells])
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in n_cells])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MECState:
        return MECState(
            cells=[g.unsqueeze(0).expand(batch_size, -1).to(device) for g in self.cells_init],
            uncertainty=[torch.exp(std).unsqueeze(0).expand(batch_size, -1).to(device) for std in self.uncertainty_init],
        )

    @property
    def n_in(self) -> int:
        return self.__n_a

    @property
    def shape(self) -> List[int]:
        return self.__n_cells

    @property
    def n_freq(self) -> int:
        return len(self.shape)

    def forward(self, *, _) -> Tuple[List[Tensor], MECState]:
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(self, a: Tensor, locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        # Shiny envs use no_direc=True (no action-driven transitions)
        no_direc = [loc.get("shiny") is not None for loc in locations]
        g_gen, transition = self.path_integrate(a, state.cells, no_direc=no_direc)
        return g_gen, MECState(cells=transition.mean, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        # Step 1: Infer from memory (Grid responsibility)
        mu_g_mem, sigma_g_mem = self.infer_from_memory(p_x, state.cells)

        # Step 2: Fuse path integration with memory cues
        mu_mec, sigma_mec = [], []
        for f in range(self.n_freq):
            mu, sigma = utils.inv_var_weight([state.cells[f], mu_g_mem[f]], [state.uncertainty[f], sigma_g_mem[f]])
            mu_mec.append(mu)
            sigma_mec.append(sigma)

        # Step 3: Apply OVC correction from shiny landmarks (if enabled and shiny envs present)
        if self._n_ovc > 0:
            # Identify which environments have shiny landmarks
            shiny_envs = [loc.get("shiny") is not None for loc in locations]
            if any(shiny_envs):
                device = mu_mec[0].device
                shiny_mask = torch.tensor(shiny_envs, dtype=torch.bool, device=device)

                # Extract shiny values (scalar per env in legacy)
                shiny_vals = [loc["shiny"] for loc in locations if loc.get("shiny") is not None]
                shiny_tensor = torch.as_tensor(shiny_vals, dtype=torch.float32, device=device).unsqueeze(-1)  # (N_shiny, 1)

                # Prepare input for OVC cue processing (list of tensors, one per OVC module)
                shiny_input = [shiny_tensor for _ in range(self._n_ovc)]

                # Predict OVC correction from shiny cues
                mu_shiny, sigma_shiny = self.estimate_shiny(shiny_input)

                # Fuse shiny correction into OVC modules only
                for f in range(self._ovc_start, self._ovc_start + self._n_ovc):
                    f_ovc = f - self._ovc_start
                    mu_fused, sigma_fused = utils.inv_var_weight(
                        [mu_mec[f][shiny_mask], mu_shiny[f_ovc]],
                        [sigma_mec[f][shiny_mask], sigma_shiny[f_ovc]],
                    )
                    # Write back fused values (only for shiny environments)
                    mu_mec[f][shiny_mask] = mu_fused
                    sigma_mec[f][shiny_mask] = sigma_fused

        # Step 4: Sample or take mean
        if self._settings.do_sample:
            mec_inf = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_mec, sigma_mec)]
        else:
            mec_inf = mu_mec

        return mec_inf, MECState(cells=mec_inf, uncertainty=sigma_mec)

    def clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations to [-1, 1] for stability."""
        return [torch.clamp(g_f, min=self._settings.clamp_min, max=self._settings.clamp_max) for g_f in g]


def connections(f_grid: list[float]) -> list[list[bool]]:
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]


def resolve_ovc_slice(n_freq_total: int, n_freq_ovc: Optional[int]) -> Tuple[int, int]:
    """Determine which MEC modules are OVC (receive shiny landmark correction).

    Args:
        n_freq_total: Total number of MEC modules
        n_freq_ovc: User setting for OVC module count:
            - None: all modules are OVC (legacy separate_ovc=False)
            - 0: no OVC modules (disable shiny correction)
            - k>0: last k modules are OVC (legacy separate_ovc=True, n_f_ovc=k)

    Returns:
        (ovc_start, ovc_count): slice range [ovc_start:ovc_start+ovc_count] for OVC modules

    Examples:
        >>> resolve_ovc_slice(5, None)  # All modules are OVC
        (0, 5)
        >>> resolve_ovc_slice(5, 0)     # No OVC modules
        (5, 0)
        >>> resolve_ovc_slice(5, 2)     # Last 2 modules are OVC
        (3, 2)
    """
    if n_freq_ovc is None:
        # Apply to all modules (legacy separate_ovc=False)
        return 0, n_freq_total

    n_freq_ovc = int(n_freq_ovc)
    if n_freq_ovc < 0 or n_freq_ovc > n_freq_total:
        raise ValueError(f"MECSettings.n_freq_ovc must be in [0, {n_freq_total}] or None; got {n_freq_ovc}")

    if n_freq_ovc == 0:
        # No OVC correction
        return n_freq_total, 0

    # Apply to last k modules (legacy separate_ovc=True)
    return n_freq_total - n_freq_ovc, n_freq_ovc
