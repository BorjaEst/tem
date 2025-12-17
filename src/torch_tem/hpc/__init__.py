"""Hippocampus (HPC) module: Memory storage and retrieval for TEM.

This module implements the hippocampal memory system responsible for storing
and retrieving grounded location representations through Hebbian plasticity
and attractor dynamics.

Components:
    Memory: High-level interface combining storage and retrieval
    MemoryStorage: Hebbian memory matrices with plasticity updates
    AttractorDynamics: Iterative pattern completion for memory retrieval

Theory:
    The hippocampus maintains associative memory between abstract locations (g)
    and grounded locations (p) through Hebbian learning. Attractor dynamics
    enable pattern completion, allowing partial cues to retrieve full memories.

    The dual-memory architecture (M_gen/M_inf) supports bidirectional inference:
    - M_gen: Grid → Place mapping for generative pathway
    - M_inf: Sensory → Place mapping for inference pathway (optional)
"""

from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor, nn

from .. import utils
from ..types import BatchedMemory, GroundedLocation, MultiScaleCode
from . import attractor, grounded, storage
from .attractor import AttractorDynamics, AttractorParams
from .grounded import GroundedLocInference, GroundedLocParams
from .storage import MemoryStorage, StorageParams


class HPCParams(attractor.AttractorParams, grounded.GroundedLocParams, storage.StorageParams):
    """Protocol for HPC model initialization parameters.

    Combines parameters from all HPC submodules and adds HPC-specific config.
    """

    batch_size: int
    eta: float  # Learning rate for memory updates
    n_p: List[int]  # Total place cell dimensions per frequency
    lambda_: float  # Memory retention factor
    common_memory: bool  # Whether to use shared memory for inference/generation
    n_f: int  # Number of frequency modules
    n_f_g: int  # Number of frequency modules used for grounded location
    n_f_ovc: int  # Number of frequency modules used for overcomplete representation
    f_extended: bool  # Whether to use extended frequency representation
    i_attractor: List[int]  # Indices of frequencies used in attractor dynamics
    max_freq_inf: int  # Max frequency index for inference attractor
    max_freq_gen: int  # Max frequency index for generative attractor


@dataclass(frozen=True)
class HPCState:
    """HPC state containing grounded location and memory matrices.

    Attributes:
        grounded_location: Inferred place cell activations (conjunctive code)
        memory: List [M_gen, M_inf] where M_inf may be None if common_memory=True
    """

    grounded_location: GroundedLocation
    memory: List[BatchedMemory]  # [M_gen, M_inf] or [M_gen, None]

    def detach(self) -> None:
        """Detach all tensors in the state from the computation graph."""
        self.grounded_location = [x.detach() for x in self.grounded_location]
        self.memory = [mem.detach() if mem is not None else None for mem in self.memory]


class HPCModel(nn.Module):
    def __init__(self, params: HPCParams):
        super().__init__()
        update_maks = utils.create_p_update_mask(params.n_p, params.n_f, params.n_f_g, params.n_f_ovc, params.f_extended)
        inference_mask = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_inf)
        generative_mask = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_gen)

        # Initialize components
        self.storage = storage.MemoryStorage(params, update_maks)
        self.attractor = attractor.AttractorDynamics(params, inference_mask, generative_mask)
        self.grounded = grounded.GroundedLocInference(params)
        self.batch_size = params.batch_size
        self.eta = params.eta
        self.use_dual_memory = not params.common_memory

    @property
    def n_p_total(self) -> int:
        """Total number of place cells across all frequencies."""
        return sum(self.storage.n_p)

    def init_state(self, device: torch.device) -> HPCState:
        p = torch.zeros([self.batch_size, sum(self.storage.n_p)], dtype=torch.float, device=device)
        M_gen = torch.zeros(self.batch_size, self.n_p_total, self.n_p_total)
        if not self.use_dual_memory:
            return HPCState(grounded_location=p, memory=[M_gen])
        M_inf = torch.zeros(self.batch_size, self.n_p_total, self.n_p_total) if self.use_dual_memory else None
        return HPCState(grounded_location=p, memory=[M_gen, M_inf])

    def forward(self, g_: MultiScaleCode, x_: MultiScaleCode, p_generated: GroundedLocation, state: HPCState) -> HPCState:
        p = self.grounded(g_, x_)  # Infer grounded location from grid cells and sensory input
        memory = self.update(p, p_generated, state)  # Update memory with Hebbian plasticity
        return HPCState(grounded_location=p, memory=memory)

    def retrieve(self, query: MultiScaleCode, for_inference: bool, state: HPCState) -> MultiScaleCode:
        """Retrieve grounded location from memory via attractor dynamics.

        Args:
            query: Initial query pattern (sensory or abstract projection)
            for_inference: If True, use M_inf; if False, use M_gen
            state: Current HPC state containing memory matrices

        Returns:
            Refined grounded location after attractor convergence
        """
        M = state.memory[1] if (for_inference and self.use_dual_memory) else state.memory[0]
        return self.attractor(query, M, for_inference=for_inference)

    def update(self, p_inferred: MultiScaleCode, p_generated: MultiScaleCode, state: HPCState) -> List[BatchedMemory]:
        """Update memory matrices using Hebbian plasticity.

        Args:
            p_inferred: Inferred grounded locations as List[n_f] of [B, n_p[f]]
            p_generated: Generated grounded locations as List[n_f] of [B, n_p[f]]
            state: Current HPC state with memory matrices

        Returns:
            Updated memory matrices [M_gen] or [M_gen, M_inf]
        """
        # Transform multi-scale code to flat vectors for storage
        p_inferred_flat = torch.cat(p_inferred, dim=1)  # [B, sum(n_p)]
        p_generated_flat = torch.cat(p_generated, dim=1)  # [B, sum(n_p)]

        # Update generative memory
        M_gen = self.storage.update(p_inferred_flat, p_generated_flat, state.memory[0])

        if not self.use_dual_memory:
            return [M_gen, None]

        # Update inference memory (without mask for full connectivity)
        M_inf = self.storage.update(p_inferred_flat, p_generated_flat, state.memory[1])
        return [M_gen, M_inf]


__all__ = ["HPCParams", "HPCState", "HPCModel"]
