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

from torch_tem.core.mec.memory import P2GMemoryModel
from torch_tem.core.mec.ovc import OVCCorrection
from torch_tem.core.mec.path import PathIntegrator
from torch_tem.settings import MECSettings
from torch_tem.types import Transition

__all__ = ["MECModel", "MECState", "MECPathIntegrator", "MECMemoryInference", "MECOVCCorrection"]


@dataclass
class MECState:
    """State container for MEC dynamics.

    Represents the current belief about abstract location as grid cell activations
    with associated uncertainty. State is explicit and immutable from the module's
    perspective (returned as new copies on each forward pass).
    """

    cells: List[Tensor]  # MEC cell activations (grid and OVC) per frequency
    uncertainty: Optional[List[Tensor]] = None  # Grid cell uncertainty per frequency
    _ovc_start: Optional[int] = None  # Cached OVC start index for transition property

    @property
    def grid_cells(self) -> List[Tensor]:
        """Return only the grid cell activations (exclude OVCs)."""
        if self._ovc_start is None:
            return self.cells
        return self.cells[: self._ovc_start]

    @property
    def ovc_cells(self) -> Optional[List[Tensor]]:
        """Return only the OVC cell activations, or None if no OVCs present."""
        if self._ovc_start is None:
            return None
        return self.cells[self._ovc_start :]

    def new(self, **kwargs) -> List[Tensor]:
        copy = self.__dict__.copy()
        copy.update(kwargs)
        return MECState(**copy)

    @property
    def transition(self) -> Transition:
        """Return current state as Transition (mean, uncertainty)."""
        return Transition(mean=self.cells, uncertainty=self.uncertainty)

    def detach(self) -> "MECState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return MECState(
            cells=[v.detach() for v in self.cells] if self.cells is not None else None,
            uncertainty=[v.detach() for v in self.uncertainty] if self.uncertainty is not None else None,
            _ovc_start=self._ovc_start,
        )


class MECModel(nn.Module):
    """MEC orchestrator: composes path integration, memory inference, and OVC correction.

    Maintains TEM-compatible API while using composable submodules internally.
    State is explicit (passed in, returned as new MECState).
    """

    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._n_a = n_a
        self._shape, self._n_freq = shape, len(shape)
        self._settings = settings

        # Composable submodules (single responsibility each)
        self.path = PathIntegrator(n_a, shape, f_init, settings.path)
        self.memory = P2GMemoryModel(n_p, shape, settings.p2g)
        self.ovc = OVCCorrection(shape, settings.ovc)

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=settings.sigma_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in shape])
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in shape])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MECState:
        """Initialize MEC state from learned priors."""
        return MECState(
            cells=[g.unsqueeze(0).expand(batch_size, -1).to(device) for g in self.cells_init],
            uncertainty=[torch.exp(std).unsqueeze(0).expand(batch_size, -1).to(device) for std in self.uncertainty_init],
            _ovc_start=self.ovc.start if self.ovc.n_freq > 0 else None,
        )

    def set_runtime(self, *, p2g_scale_offset: float):
        """Set runtime hyperparameters (for curriculum training)."""
        self.memory.scale_curriculum_sigma(p2g_scale_offset)

    @property
    def settings(self) -> MECSettings:
        """MEC module settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Grid cell counts per frequency module."""
        return self._shape

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
            g_gen: Grid code for generative branch (sampled if do_sample=True)
            new_state: Updated MEC state after transition
        """
        # Build no-direction mask for shiny environments
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        any_shiny = any(shiny_envs)
        no_direc_mask = torch.tensor(shiny_envs, device=a.device, dtype=torch.bool) if any_shiny else None

        # 1) Action-driven transition for the state (legacy g_path)
        transition = self.path(a, state.cells, no_direc_mask=None)
        cells_next = self._sample(transition.mean, transition.uncertainty)

        # 2) g_gen: reuse mu when possible, only compute no_direc when needed
        if any_shiny:
            g_gen = self._clamp(self.path.mean(a, state.cells, no_direc_mask))
        elif self.settings.do_sample:
            g_gen = cells_next  # legacy: g_gen == sampled g when no shiny
        else:
            g_gen = self._clamp(transition.mean)

        return g_gen, state.new(cells=cells_next, uncertainty=transition.uncertainty)

    def inference(self, p_x: List[Tensor], locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Execute inference step: fuse path integration with memory and OVC cues.

        Args:
            p_x: Retrieved place cell activations per frequency (from HPC)
            locations: Per-env metadata (for OVC shiny correction)
            state: Current MEC state (contains path-integrated belief)

        Returns:
            g_inf: Inferred grid code (sampled if do_sample=True, legacy parity)
            new_state: Updated MEC state
        """
        # Step 1: Correct path integration with memory-based inference
        transition = self.memory(p_x, state.transition)

        # Step 2: Apply OVC correction from shiny landmarks (if enabled)
        transition = self.ovc(locations, transition) if self.ovc.n_freq > 0 else transition

        # Apply central sampling policy (legacy parity: g_inf is sampled when do_sample=True)
        cells_next = self._sample(transition.mean, transition.uncertainty)
        g_inf = self._clamp(cells_next)

        return g_inf, state.new(cells=cells_next, uncertainty=transition.uncertainty)

    def _clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations for stability."""
        return [torch.clamp(g_f, min=self._settings.clamp_min, max=self._settings.clamp_max) for g_f in g]

    def _sample(self, mu: List[Tensor], sigma: List[Tensor]) -> List[Tensor]:
        """Apply sampling policy: mu + sigma * eps if do_sample, else mu.

        Args:
            mu: Distribution means per frequency
            sigma: Distribution uncertainties per frequency

        Returns:
            Sampled values if do_sample=True, else means
        """
        if self._settings.do_sample:
            return [mu_f + sigma_f * torch.randn_like(mu_f) for mu_f, sigma_f in zip(mu, sigma)]
        return mu
