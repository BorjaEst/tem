"""MEC (medial entorhinal cortex) dynamics.

This package implements grid-cell path integration with optional corrections
from hippocampal memory (p→g) and shiny landmark cues (OVC).

The public entry point is `MECModel`, which exposes a TEM-compatible API via
`init_state`, `generative`, and `inference`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem.core.mec.ovc import OVCCorrection
from torch_tem.core.mec.p2g import P2GMemory
from torch_tem.core.mec.path import PathIntegrator
from torch_tem.settings import MECSettings
from torch_tem.types import Transition

__all__ = ["MECModel", "MECState", "OVCCorrection", "PathIntegrator", "P2GMemory"]


@dataclass
class MECState:
    """Container for MEC state.

    The state represents a belief over abstract location encoded as grid (and
    optionally OVC) activations with per-frequency uncertainty.

    Attributes:
        cells: List of per-frequency activations. If OVC modules are enabled,
            their activations are appended after the grid modules.
        uncertainty: Optional list of per-frequency uncertainties aligned with
            `cells`.
        _ovc_start: Internal start index of OVC modules within `cells`, or
            `None` when OVC is disabled.
    """

    cells: List[Tensor]  # MEC cell activations (grid and OVC) per frequency
    uncertainty: Optional[List[Tensor]] = None  # Grid cell uncertainty per frequency
    _ovc_start: Optional[int] = None  # Cached OVC start index for transition property

    @property
    def grid_cells(self) -> List[Tensor]:
        """Return grid-cell activations (excluding any OVC modules).

        Returns:
            Per-frequency grid-cell activations.
        """
        if self._ovc_start is None:
            return self.cells
        return self.cells[: self._ovc_start]

    @property
    def ovc_cells(self) -> Optional[List[Tensor]]:
        """Return OVC activations if present.

        Returns:
            Per-frequency OVC activations, or `None` if OVC modules are disabled.
        """
        if self._ovc_start is None:
            return None
        return self.cells[self._ovc_start :]

    def new(self, **kwargs) -> "MECState":
        """Return a new state with updated fields.

        This is a convenience helper used to keep state updates explicit while
        avoiding in-place mutation.

        Args:
            **kwargs: Field overrides for the new state.

        Returns:
            A new `MECState` instance.
        """
        copy = self.__dict__.copy()  # TODO: Should we use detach here?
        copy.update(kwargs)
        return MECState(**copy)

    @property
    def transition(self) -> Transition:
        """Convert the state to a `Transition`.

        Returns:
            A `Transition(mean=cells, uncertainty=uncertainty)`.
        """
        return Transition(mean=self.cells, uncertainty=self.uncertainty)

    def detach(self) -> "MECState":
        """Return a detached copy.

        This is typically used when caching a previous iteration state without
        keeping autograd history.

        Returns:
            A detached copy of the current state.
        """
        return MECState(
            cells=[v.detach() for v in self.cells] if self.cells is not None else None,
            uncertainty=[v.detach() for v in self.uncertainty] if self.uncertainty is not None else None,
            _ovc_start=self._ovc_start,
        )


class MECModel(nn.Module):
    """Compose MEC submodules into a TEM-compatible interface.

    The model exposes a stateful interface using explicit `MECState` objects.
    Internally it composes:

    - `PathIntegrator` for action-driven transitions
    - `P2GMemory` for memory-based correction (p→g)
    - `OVCCorrection` for shiny landmark cue fusion
    """

    def __init__(self, n_a: int, n_p: List[int], shape: List[int], f_init: List[float], settings: MECSettings):
        super().__init__()
        self._n_a = n_a
        self._shape, self._n_freq = shape, len(shape)
        self._settings = settings

        # Composable submodules (single responsibility each)
        self.path = PathIntegrator(n_a, shape, f_init, settings.path)
        self.p2g = P2GMemory(n_p, shape, settings.p2g)
        self.ovc = OVCCorrection(shape, settings.ovc)

        # Prior: learned "default phase" of the grid code at reset
        init_fn = lambda size: truncnorm.rvs(-2, 2, size=size, loc=0, scale=settings.sigma_init)
        self.cells_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in shape])
        self.uncertainty_init = nn.ParameterList([nn.Parameter(torch.tensor(init_fn(n), dtype=torch.float32)) for n in shape])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> MECState:
        """Create an initial MEC state from learned priors.

        Args:
            batch_size: Batch size for the returned state tensors.
            device: Optional device to place the returned tensors on.

        Returns:
            An initialized `MECState`.
        """
        g0 = [g.unsqueeze(0).expand(batch_size, -1).to(device) for g in self.cells_init]
        sigma_0 = [std.unsqueeze(0).expand(batch_size, -1).to(device) for std in self.uncertainty_init]
        return MECState(cells=g0, uncertainty=sigma_0, _ovc_start=self.ovc.start if self.ovc.n_freq > 0 else None)

    def set_runtime(self, *, p2g_scale_offset: float):
        """Set runtime hyperparameters.

        Args:
            p2g_scale_offset: Scale factor for the P2G uncertainty curriculum.
        """
        self.p2g.scale_curriculum_sigma(p2g_scale_offset)

    @property
    def settings(self) -> MECSettings:
        """Return the MEC settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Return grid-cell counts per frequency module."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Return the number of frequency modules."""
        return self._n_freq

    @property
    def grid_n_freq(self) -> int:
        """Return the number of grid (spatial) frequency modules."""
        if self.settings.ovc.n_freq is None:
            return self._n_freq
        return self._n_freq - self.ovc.n_freq

    def forward(self, *, _) -> Tuple[List[Tensor], MECState]:
        """Not implemented.

        Raises:
            NotImplementedError: Always. Use `generative` or `inference`.
        """
        raise NotImplementedError("MEC forward not implemented. Use generative() or inference().")

    def generative(self, a: Tensor, locations: list[dict], state: MECState) -> Tuple[List[Tensor], MECState]:
        """Run the generative (path integration) update.

        Args:
            a: One-hot action tensor of shape `(batch, n_a)`.
            locations: Per-environment metadata. A non-`None` `"shiny"` value
                indicates a landmark cue is present.
            state: Current MEC state.

        Returns:
            A tuple `(g_gen, new_state)` where `g_gen` is the generative grid
            code and `new_state` is the updated MEC state.
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
        """Run inference by fusing memory and OVC cues into the state.

        Args:
            p_x: Retrieved place-cell activations per frequency (from HPC).
            locations: Per-environment metadata used for OVC correction.
            state: Current MEC state (typically after path integration).

        Returns:
            A tuple `(g_inf, new_state)` where `g_inf` is the inferred grid code
            and `new_state` is the updated MEC state.
        """
        # Step 1: Correct path integration with memory-based inference
        transition = self.p2g(p_x, state.transition)

        # Step 2: Apply OVC correction from shiny landmarks (if enabled)
        transition = self.ovc(locations, transition) if self.ovc.n_freq > 0 else transition

        # Apply central sampling policy (legacy parity: g_inf is sampled when do_sample=True)
        cells_next = self._sample(transition.mean, transition.uncertainty)
        g_inf = self._clamp(cells_next)

        return g_inf, state.new(cells=cells_next, uncertainty=transition.uncertainty)

    def _clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp activations for numerical stability.

        Args:
            g: Per-frequency activations.

        Returns:
            Clamped activations.
        """
        return [torch.clamp(g_f, min=self._settings.clamp_min, max=self._settings.clamp_max) for g_f in g]

    def _sample(self, mu: List[Tensor], sigma: List[Tensor]) -> List[Tensor]:
        """Sample from a diagonal Gaussian if enabled.

        When `settings.do_sample` is true, returns `mu + sigma * eps` with
        `eps ~ N(0, I)`.

        Args:
            mu: Per-frequency distribution means.
            sigma: Per-frequency distribution uncertainties.

        Returns:
            Sampled activations if sampling is enabled, otherwise `mu`.
        """
        if self._settings.do_sample:
            return [mu_f + sigma_f * torch.randn_like(mu_f) for mu_f, sigma_f in zip(mu, sigma)]
        return mu
