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
from typing import List, Optional, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from torch_tem.core.hpc.attractor import AttractorConfig, AttractorDynamics
from torch_tem.core.hpc.grounded import GroundedLocConfig, GroundedLocInference
from torch_tem.core.hpc.storage import MemoryStorage, StorageConfig
from torch_tem.types import BatchedMemory, GroundedLocation, MultiScaleCode


class HPCConfig(BaseModel):
    """Hippocampus configuration: memory storage, retrieval, and grounded location inference."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # General HPC settings
    common_memory: bool = Field(default=False, description="Share a single memory between generative and inference networks")

    # Submodule configurations
    attractor: AttractorConfig = Field(default_factory=AttractorConfig, description="Attractor dynamics configuration")
    grounded: GroundedLocConfig = Field(default_factory=GroundedLocConfig, description="Grounded location inference configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Memory storage configuration")


class HPCContext(Protocol):
    """Protocol for HPC context providing architecture parameters.

    Attributes:
        mask_inference: Hierarchical masks for inference retrieval [i_attractor] of (sum(n_p),).
        mask_generative: Hierarchical masks for generative retrieval [i_attractor] of (sum(n_p),).
        update_mask: Mask for Hebbian memory updates (sum(n_p), sum(n_p)).
    """

    mask_inference: List[torch.Tensor]
    mask_generative: List[torch.Tensor]
    update_mask: torch.Tensor


@dataclass(frozen=True)
class HPCState:
    """HPC state containing grounded location and memory matrices.

    Attributes:
        grounded_location: Inferred place cell activations (conjunctive code).
            None on initialization, set by first forward pass.
        memory: List [M_gen, M_inf] where M_inf may be None if common_memory=True.
    """

    grounded_location: Optional[GroundedLocation]
    memory: List[BatchedMemory]  # [M_gen, M_inf] or [M_gen, None]

    def detach(self) -> "HPCState":
        """Detach all tensors in the state from the computation graph.

        Returns:
            HPCState: New state with detached tensors.
        """
        return HPCState(
            grounded_location=[x.detach() for x in self.grounded_location] if self.grounded_location is not None else None,
            memory=[mem.detach() if mem is not None else None for mem in self.memory],
        )


class HPCModel(nn.Module):
    """Hippocampal memory model combining storage, retrieval, and grounded location inference.

    This module integrates three core components:
    - AttractorDynamics: Iterative pattern completion for memory retrieval
    - GroundedLocInference: Conjunctive coding of grid cells and sensory input
    - MemoryStorage: Hebbian plasticity for associative learning
    """

    def __init__(self, context: HPCContext, config: HPCConfig):
        """Initialize HPC model with context and configuration.

        Args:
            context (HPCContext): Architectural parameters (dimensions, masks).
            config (HPCConfig): Hyperparameters (learning rates, iterations).
        """
        super().__init__()
        self._config = config

        self.attractor = AttractorDynamics(context.mask_inference, context.mask_generative, config.attractor)
        self.grounded = GroundedLocInference(config.grounded)
        self.storage = MemoryStorage(context.update_mask, config.storage)

    @property
    def common_memory(self) -> bool:
        """Whether generative and inference networks share a common memory."""
        return self._config.common_memory

    @property
    def size(self) -> int:
        """Total size of grounded location representation (sum of place cell dimensions)."""
        return self.storage._update_mask.shape[0]

    def init_state(self, batch_size: int, device: torch.device) -> HPCState:
        """Initialize HPC state with None grounded location and zero memory matrices.

        Args:
            batch_size (int): Number of parallel sequences.
            device (torch.device): Device for tensor allocation.

        Returns:
            HPCState: Initial state with None grounded_location (will be set by first forward)
                     and zero-initialized memory matrices.
        """
        M_gen = torch.zeros(batch_size, self.size, self.size, device=device)
        if self.common_memory:
            return HPCState(grounded_location=None, memory=[M_gen])
        M_inf = torch.zeros(batch_size, self.size, self.size, device=device)
        return HPCState(grounded_location=None, memory=[M_gen, M_inf])

    def forward(self, g_: MultiScaleCode, x_: MultiScaleCode, p_generated: GroundedLocation, state: HPCState) -> HPCState:
        """Forward pass: infer grounded location and update memory.

        Args:
            g_ (MultiScaleCode): Expanded abstract location (grid cells).
            x_ (MultiScaleCode): Expanded sensory input.
            p_generated (GroundedLocation): Generated grounded location from previous step.
            state (HPCState): Current HPC state.

        Returns:
            HPCState: Updated state with new grounded location and memories.
        """
        p = self.grounded(g_, x_)  # Infer grounded location from grid cells and sensory input
        memory = self.update(p, p_generated, state)  # Update memory with Hebbian plasticity
        return HPCState(grounded_location=p, memory=memory)

    def retrieve(self, query: MultiScaleCode, for_inference: bool, state: HPCState) -> MultiScaleCode:
        """Retrieve grounded location from memory via attractor dynamics.

        Args:
            query (MultiScaleCode): Initial query pattern (sensory or abstract projection).
            for_inference (bool): If True, use M_inf; if False, use M_gen.
            state (HPCState): Current HPC state containing memory matrices.

        Returns:
            MultiScaleCode: Refined grounded location after attractor convergence.
        """
        M = state.memory[1] if (for_inference and not self.common_memory) else state.memory[0]
        return self.attractor(query, M, for_inference=for_inference)

    def update(self, p_inferred: MultiScaleCode, p_generated: MultiScaleCode, state: HPCState) -> List[BatchedMemory]:
        """Update memory matrices using Hebbian plasticity.

        Args:
            p_inferred (MultiScaleCode): Inferred grounded locations as List[n_f] of [B, n_p[f]].
            p_generated (MultiScaleCode): Generated grounded locations as List[n_f] of [B, n_p[f]].
            state (HPCState): Current HPC state with memory matrices.

        Returns:
            List[BatchedMemory]: Updated memory matrices [M_gen] or [M_gen, M_inf].
        """
        # Transform multi-scale code to flat vectors for storage
        p_inferred_flat = torch.cat(p_inferred, dim=1)  # [B, sum(n_p)]
        p_generated_flat = torch.cat(p_generated, dim=1)  # [B, sum(n_p)]

        # Update generative memory
        M_gen = self.storage.update(p_inferred_flat, p_generated_flat, state.memory[0])

        if self.common_memory:
            return [M_gen, None]

        # Update inference memory (without mask for full connectivity)
        M_inf = self.storage.update(p_inferred_flat, p_generated_flat, state.memory[1])
        return [M_gen, M_inf]


__all__ = ["HPCConfig", "HPCState", "HPCModel", "AttractorConfig", "GroundedLocConfig", "StorageConfig"]
