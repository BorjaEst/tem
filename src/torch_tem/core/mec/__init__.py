"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

This module is now composed of specialized submodules:
- MECPathIntegrator: Action-driven transitions
- MECMemoryInference: p→g correction from memory
- MECOVCCorrection: Shiny landmark cue processing

Design goals:
- Explicit state (MECState passed in/out, never mutated internally)
- Composable submodules (each with single responsibility)
- TEM-compatible API (init_state, generative, inference)
- Export-ready (prepare for ONNX via optional tensor-only interfaces)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils as tem_utils
from torch_tem.settings import MECSettings

from .memory import P2GMemoryModel
from .ovc import OVCCorrection
from .path import PathIntegrator
from .utils import connections, resolve_ovc_slice

__all__ = [
    "MECModel",
    "MECState",
    "MECPathIntegrator",
    "MECMemoryInference",
    "MECOVCCorrection",
    "connections",
    "resolve_ovc_slice",
]


@dataclass
class MECState:
    """State container for MEC dynamics.

    Represents the current belief about abstract location as grid cell activations
    with associated uncertainty. State is explicit and immutable from the module's
    perspective (returned as new copies on each forward pass).
    """

    cells: List[Tensor]  # MEC cell activations (grid and OVC) per frequency
    uncertainty: Optional[List[Tensor]] = None  # Grid cell uncertainty per frequency

    def detach(self) -> "MECState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return MECState(
            cells=[v.detach() for v in self.cells] if self.cells is not None else None,
            uncertainty=[v.detach() for v in self.uncertainty] if self.uncertainty is not None else None,
        )


class MECModel(nn.Module):
    """MEC orchestrator: composes path integration, memory inference, and OVC correction.

    Maintains TEM-compatible API while using composable submodules internally.
    State is explicit (passed in, returned as new MECState).
    """

    def __init__(self, n_a: int, n_p: List[int], n_cells: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._n_a = n_a
        self._n_cells, self._n_freq = n_cells, len(n_cells)
        self._settings = settings

        # Composable submodules (single responsibility each)
        self.path = PathIntegrator(n_a, n_cells, f_init, settings.path)
        self.memory = P2GMemoryModel(n_p, n_cells, settings.p2g)
        self.ovc = OVCCorrection(n_cells, settings.ovc)

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=settings.sigma_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in n_cells])
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in n_cells])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MECState:
        """Initialize MEC state from learned priors."""
        return MECState(
            cells=[g.unsqueeze(0).expand(batch_size, -1).to(device) for g in self.cells_init],
            uncertainty=[torch.exp(std).unsqueeze(0).expand(batch_size, -1).to(device) for std in self.uncertainty_init],
        )

    def set_runtime(self, *, p2g_scale_offset: float):
        """Set runtime hyperparameters (for curriculum training)."""
        self.memory.set_runtime(p2g_scale_offset=p2g_scale_offset)

    @property
    def settings(self) -> MECSettings:
        """MEC module settings."""
        return self._settings

    @property
    def n_in(self) -> int:
        """Number of action inputs."""
        return self._n_a

    @property
    def shape(self) -> List[int]:
        """Grid cell counts per frequency module."""
        return self._n_cells

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return self._n_freq

    def forward(self, *, _) -> Tuple[List[Tensor], MECState]:
        """Not implemented. Use generative() or inference()."""
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(self, a: Tensor, locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Execute generative path integration step.

        Args:
            a: Action tensor (batch, n_a) one-hot encoded
            locations: Per-env metadata (shiny key indicates landmark presence)
            state: Current MEC state

        Returns:
            g_gen: Grid code for generative branch (query to HPC)
            new_state: Updated MEC state after transition
        """
        # Build no-direction mask for shiny environments
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        no_direc_mask = torch.tensor(shiny_envs, device=a.device, dtype=torch.bool) if any(shiny_envs) else None

        # Path integrate
        g_gen, transition = self.path(a, state.cells, no_direc_mask)

        return g_gen, MECState(cells=transition.mean, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Execute inference step: fuse path integration with memory and OVC cues.

        Args:
            p_x: Retrieved place cell activations per frequency (from HPC)
            locations: Per-env metadata (for OVC shiny correction)
            state: Current MEC state (contains path-integrated belief)

        Returns:
            g_inf: Inferred grid code (fused estimate)
            new_state: Updated MEC state
        """
        # Step 1: Infer from memory (p_x → g correction)
        mu_g_mem, sigma_g_mem = self.memory(p_x, state.cells)

        # Step 2: Fuse path integration with memory cues (precision weighting)
        mu_mec, sigma_mec = [], []
        for f in range(self.n_freq):
            mu, sigma = tem_utils.inv_var_weight([state.cells[f], mu_g_mem[f]], [state.uncertainty[f], sigma_g_mem[f]])
            mu_mec.append(mu)
            sigma_mec.append(sigma)

        # Step 3: Apply OVC correction from shiny landmarks (if enabled)
        mu_mec, sigma_mec = self.ovc(locations, mu_mec, sigma_mec)

        # Step 4: Sample or take mean (controlled by settings)
        if self._settings.do_sample:
            g_inf = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_mec, sigma_mec)]
        else:
            g_inf = mu_mec

        # Clamp for stability
        g_inf = self._clamp(g_inf)

        return g_inf, MECState(cells=g_inf, uncertainty=sigma_mec)

    def _clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations for stability."""
        return [torch.clamp(g_f, min=self._settings.clamp_min, max=self._settings.clamp_max) for g_f in g]
