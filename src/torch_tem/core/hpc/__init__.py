from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.core.hpc.attractor import AttractorNetwork
from torch_tem.core.hpc.distribution import GroundedLocationDistribution
from torch_tem.core.hpc.hebbian import HebbianUpdater
from torch_tem.settings import HPCSettings
from torch_tem.types import Matrix

__all__ = ["HPCModel", "HPCState", "p_retrieve_mask_inf", "p_retrieve_mask_gen"]


@dataclass
class HPCState:
    """State container for HPC dynamics."""

    p: List[Tensor]  # Multi-frequency grounded location features
    memory: List[Matrix]  # Memory matrices

    def detach(self) -> "HPCState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return HPCState(
            p=[v.detach() for v in self.p] if self.p is not None else None,
            memory=[m.detach() for m in self.memory] if self.memory is not None else None,
        )


@dataclass
class HPCRuntime:
    """Runtime values injected by training (not architectural parameters)."""

    eta: float = 0.0
    hebbian_decay: float = 0.9999


class HPCModel(nn.Module):
    """HPC façade that preserves legacy API while delegating to submodules."""

    def __init__(self, grid_n_freq: int, shape: List[int], f_init: List[float], settings: HPCSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._grid_n_freq = int(grid_n_freq)
        self._settings = settings
        self._i_attractor = grid_n_freq

        # Store masks as buffers for device management
        masks = torch.stack(gen_masks_full(shape, n_stages=grid_n_freq))
        self.register_buffer("masks_full", masks, persistent=False)
        masks = torch.stack(gen_masks_hierarchical(shape, n_stages=grid_n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)

        self.runtime = HPCRuntime()

        # Instantiate submodules
        self.attractor = AttractorNetwork(shape, settings.attractor)
        self.hebbian_updater = HebbianUpdater(shape=self._shape, i_attractor=self._i_attractor, f_init=f_init)
        self.distribution = GroundedLocationDistribution(shape=self._shape)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> HPCState:
        if device is None:
            # Fall back to module device when not explicitly provided
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        return HPCState(p=p_init, memory=self._init_memory(batch_size=batch_size, device=device))

    def _init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
        m0 = torch.zeros((batch_size, sum(self.shape), sum(self.shape)), dtype=torch.float, device=device)
        memory = [m0]
        memory.append(m0 if self._settings.common_memory else m0.clone())
        return memory

    def set_runtime(self, *, eta: float, hebbian_decay: float) -> None:
        self.runtime.eta = float(eta)
        self.runtime.hebbian_decay = float(hebbian_decay)

    @property
    def settings(self) -> HPCSettings:
        """HPC module settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return self._n_freq

    @property
    def i_attractor(self) -> int:
        """Number of attractor iterations."""
        return self._i_attractor

    @property
    def i_attractor_max_freq_inf(self) -> list[int]:
        return [self.i_attractor for _ in range(self.n_freq)]

    @property
    def i_attractor_max_freq_gen(self) -> list[int]:
        return [self.i_attractor - freq_nr for freq_nr in range(self.i_attractor)] + [self.i_attractor for _ in range(self.n_freq - self.i_attractor)]

    def forward(self, *, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        raise NotImplementedError("HPC forward not implemented. Use generative() or inference().")

    def generative(self, p_g: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        if not self._settings.do_sample:
            return p_g, HPCState(p=p_g, memory=state.memory)
        p = self.distribution.sample(p_g)
        return p, HPCState(p=p, memory=state.memory)

    def inference(self, x_: List[Tensor], g_: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        mu_p = self.f_p([g_[f] * x_[f] for f in range(self.n_freq)])
        if not self._settings.do_sample:
            return mu_p, HPCState(p=mu_p, memory=state.memory)
        p = self.distribution.sample(mu_p)
        return p, HPCState(p=p, memory=state.memory)

    def hebbian(self, M_prev: Tensor, p_inf: Tensor, p_gen_gi: Tensor, do_hierarchical_connections: bool = True) -> Tensor:
        return self.hebbian_updater(
            M_prev,
            p_inf,
            p_gen_gi,
            do_hierarchical_connections=do_hierarchical_connections,
            eta=self.runtime.eta,
            hebbian_decay=self.runtime.hebbian_decay,
        )

    def recall(self, p_query: List[Tensor], state: HPCState, *, mode: Literal["full", "hierarchical"]) -> List[Tensor]:
        if mode == "full":
            return self.attractor(p_query, state.memory[1], masks=self.masks_full)
        elif mode == "hierarchical":
            return self.attractor(p_query, state.memory[0], masks=self.masks_hierarchical)
        raise ValueError(f"Invalid mode '{mode}'. Expected 'full' or 'hierarchical'.")

    def f_p(self, p):
        return [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1))


def gen_masks_full(hpc_shape: List[int], n_stages: int) -> List[torch.Tensor]:
    masks = [torch.zeros(sum(hpc_shape)) for _ in range(n_stages)]
    i_attractor_max_freq_inf = [n_stages for _ in range(n_stages)]
    n_p = np.cumsum([0] + hpc_shape)

    # For each frequency, insert ones in the mask for those iterations
    for f, max_i in enumerate(i_attractor_max_freq_inf):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0
    return masks


def gen_masks_hierarchical(hpc_shape: List[int], n_stages: int) -> List[torch.Tensor]:
    masks = [torch.zeros(sum(hpc_shape)) for _ in range(n_stages)]
    i_attractor_max_freq_gen = [n_stages - f for f in range(n_stages)] + [n_stages for _ in range(len(hpc_shape) - n_stages)]
    n_p = np.cumsum([0] + hpc_shape)

    # For each frequency, insert ones in the mask for those iterations
    for f, max_i in enumerate(i_attractor_max_freq_gen):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0
    return masks
