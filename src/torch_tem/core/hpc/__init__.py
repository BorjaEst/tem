from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.core.hpc.attractor import AttractorNetwork
from torch_tem.core.hpc.hebbian import HebbianUpdater
from torch_tem.core.hpc.location import GroundLocation
from torch_tem.settings import HPCSettings
from torch_tem.types import GroundedLocation, Matrix, MultiScaleCode, Transition

__all__ = ["HPCModel", "HPCState", "AttractorNetwork", "GroundLocation", "HebbianUpdater"]


@dataclass
class HPCState:
    """State container for HPC dynamics."""

    transition: Transition  # State and uncertainty over grounded locations
    memory: List[Matrix]  # Memory matrices

    def new(self, cells: GroundedLocation, uncertanty: MultiScaleCode) -> "HPCState":
        copy = self.__dict__.copy()  # TODO: Should we use detach here?
        copy.update({"transition": Transition(mean=cells, uncertainty=uncertanty)})
        return HPCState(**copy)

    def detach(self) -> "HPCState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        mean = [v.detach() for v in self.cells]
        uncertainty = [v.detach() for v in self.uncertainty] if self.uncertainty else None
        memory = [m.detach() for m in self.memory] if self.memory is not None else None
        return HPCState(Transition(mean, uncertainty), memory)

    @property
    def cells(self) -> List[Tensor]:
        """Return grounded location features."""
        return self.transition.mean

    @property
    def uncertainty(self) -> Optional[List[Tensor]]:
        """Return grounded location uncertainty."""
        return self.transition.uncertainty


class HPCModel(nn.Module):
    """HPC façade that preserves legacy API while delegating to submodules."""

    def __init__(self, n_stages: int, shape: List[int], f_init: List[float], settings: HPCSettings):
        super().__init__()
        self._shape, self._n_freq = list(shape), len(shape)
        self._n_stages = n_stages
        self._settings = settings

        # Store masks as buffers for device management
        masks = utils.update_to_masks(shape, update=utils.make_update_full(n_stages, self.n_freq))
        self.register_buffer("masks_full", masks, persistent=False)
        masks = utils.update_to_masks(shape, update=utils.make_update_hierarchical(n_stages, self.n_freq))
        self.register_buffer("masks_hierarchical", masks, persistent=False)

        # Instantiate submodules
        self.attractor = AttractorNetwork(shape, settings.attractor)
        self.hebbian_updater = HebbianUpdater(shape, n_stages, f_init, settings.hebbian_update)
        self.location = GroundLocation(shape, settings.location)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> HPCState:
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        transition = Transition(mean=p_init, uncertainty=None)
        return HPCState(transition, memory=self._init_memory(batch_size=batch_size, device=device))

    def _init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
        # TODO: merge rename HebbianUpdated by memory and mv this there
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
    def n_stages(self) -> int:
        """Number of attractor stages."""
        return self._n_stages

    def forward(self, *, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        raise NotImplementedError("HPC forward not implemented. Use generative() or inference().")

    def generative(self, p_g: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        transition = Transition(mean=p_g, uncertainty=state.uncertainty)
        p = utils.sample_diag_gaussian(transition, scale=1.0)
        return p, state.new(p, state.uncertainty)

    def inference(self, x_: List[Tensor], g_: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        transition = self.location(x_, g_)
        p = utils.sample_diag_gaussian(transition, scale=1.0)
        return p, state.new(p, transition.uncertainty)

    def recall(self, p_query: List[Tensor], state: HPCState, *, mode: Literal["full", "hierarchical"]) -> List[Tensor]:
        if mode == "full":
            return self.attractor(p_query, state.memory[1], masks=self.masks_full)
        elif mode == "hierarchical":
            return self.attractor(p_query, state.memory[0], masks=self.masks_hierarchical)
        raise ValueError(f"Invalid mode '{mode}'. Expected 'full' or 'hierarchical'.")
