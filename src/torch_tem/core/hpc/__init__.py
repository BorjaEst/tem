from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

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


class HPCModel(nn.Module):
    """HPC façade that preserves legacy API while delegating to submodules."""

    def __init__(self, i_attractor: int, shape: List[int], f_init: List[float], settings: HPCSettings):
        super().__init__()
        self._shape = list(shape)
        self._i_attractor = int(i_attractor)
        self._settings = settings

        self.attractor = AttractorNetwork(shape, settings.attractor)
        self.hebbian_updater = HebbianUpdater(shape, self._i_attractor, f_init, self._settings.hebbian_update)
        self.distribution = GroundedLocationDistribution(shape, self._settings.distribution)

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
        self.hebbian_updater.runtime.eta = float(eta)
        self.hebbian_updater.runtime.hebbian_decay = float(hebbian_decay)

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return len(self._shape)

    @property
    def i_attractor(self) -> int:
        """Legacy split/iteration parameter (architecture-derived).

        Kept for compatibility with legacy code and mask construction.
        Attractor iterations are controlled by `HPCSettings.attractor.n_iters`.
        """
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

    def f_p(self, p):
        return [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1))


def p_retrieve_mask_inf(hpc: HPCModel) -> List[torch.Tensor]:
    """Hierarchical memory retrieval masks for inference model (legacy helper)."""
    n_p, i_attractor = hpc.shape, hpc.i_attractor
    masks = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]
    n_p = np.cumsum(np.concatenate(([0], n_p)))

    for f, max_i in enumerate(hpc.i_attractor_max_freq_inf):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0
    return masks


def p_retrieve_mask_gen(hpc: HPCModel) -> List[torch.Tensor]:
    """Hierarchical memory retrieval masks for generative model (legacy helper)."""
    n_p, i_attractor = hpc.shape, hpc.i_attractor
    masks = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]
    n_p = np.cumsum(np.concatenate(([0], n_p)))

    for f, max_i in enumerate(hpc.i_attractor_max_freq_gen):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0
    return masks
