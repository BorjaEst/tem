from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

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

    def __init__(self, grid_n_freq: int, shape: List[int], f_init: List[float], settings: HPCSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._grid_n_freq = int(grid_n_freq)
        self._settings = settings
        masks = _build_retrieve_masks(shape, n_stages=grid_n_freq)

        # Store masks as buffers for device management
        self.register_buffer("p_retrieve_mask_full", torch.stack(masks["full"]), persistent=False)
        self.register_buffer("p_retrieve_mask_hierarchical", torch.stack(masks["hierarchical"]), persistent=False)

        # Instantiate submodules
        self.attractor = AttractorNetwork(shape, settings.attractor)
        self.hebbian_updater = HebbianUpdater(shape, self._grid_n_freq, f_init, self._settings.hebbian_update)
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
    def grid_n_freq(self) -> int:
        """Number of grid (spatial) modules."""
        return self._grid_n_freq

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

    def recall(self, p_query: List[Tensor], state: HPCState, *, mode: Literal["inference", "generative"]) -> List[Tensor]:
        if mode == "inference":
            return self.attractor(p_query, state.memory[1], masks=self.p_retrieve_mask_full)
        elif mode == "generative":
            return self.attractor(p_query, state.memory[0], masks=self.p_retrieve_mask_hierarchical)
        raise ValueError(f"Invalid mode '{mode}'. Expected 'inference' or 'generative'.")

    def f_p(self, p):
        return [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1))


def _build_retrieve_masks(hpc_shape: List[int], n_stages: int) -> Dict[str, List[Tensor]]:
    n_freq = len(hpc_shape)
    n_stages = max(1, min(int(n_stages), n_freq))
    n_units = sum(hpc_shape)
    n_p = torch.tensor([0] + hpc_shape, dtype=torch.long).cumsum(dim=0).tolist()

    mask_full = [torch.zeros(n_units, dtype=torch.float32) for _ in range(n_stages)]
    mask_hierarchical = [torch.zeros(n_units, dtype=torch.float32) for _ in range(n_stages)]

    for f in range(n_freq):
        start, end = n_p[f], n_p[f + 1]
        for i in range(n_stages):
            mask_full[i][start:end] = 1.0
        max_i_gen = n_stages - f if f < n_stages else n_stages
        for i in range(max_i_gen):
            mask_hierarchical[i][start:end] = 1.0

    return {"full": mask_full, "hierarchical": mask_hierarchical}
