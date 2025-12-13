"""Unified memory system combining storage and retrieval for TEM.

This module provides a high-level Memory class that encapsulates both
Hebbian memory storage and attractor-based retrieval, managing the
dual-memory architecture (M_gen/M_inf) internally.
"""

from typing import List, Protocol

from .. import utils
from ..types import Matrix, MultiScaleCode
from . import attractor, storage


class MemoryParams(Protocol):
    """Protocol defining parameters required for memory system initialization."""

    n_p: List[int]
    n_f: int
    n_f_g: int
    n_f_ovc: int
    f_extended: List[float]
    lambda_: float
    kappa: float
    common_memory: bool
    batch_size: int
    i_attractor: int
    max_freq_inf: List[int]
    max_freq_gen: List[int]


class Memory:
    """Unified hippocampal memory system combining Hebbian storage and attractor retrieval.

    Manages the dual-memory architecture (M_gen/M_inf) for bidirectional inference
    between abstract and grounded location representations. Connectivity masks for
    hierarchical learning and retrieval are computed internally from model parameters.

    Architecture:
        - MemoryStorage: Hebbian plasticity with hierarchical update masks
        - AttractorDynamics: Iterative pattern completion with early-stopping
        - Dual Memory: Separate M_gen (generation) and M_inf (inference) matrices

    Data Flow:
        Inference pathway:  x → x_ → M_inf → p_x  (sensory to grounded)
        Generative pathway: g → g_ → M_gen → p_g  (abstract to grounded)
        Memory update:      M ← λ·M + η·(p_inf + p_gen) ⊗ (p_inf - p_gen)

    Attributes:
        storage: MemoryStorage instance managing Hebbian matrices
        attractor: AttractorDynamics instance for pattern completion

    Example:
        >>> from torch_tem.hpc import Memory
        >>> memory = Memory(model_config)
        >>>
        >>> # Retrieve grounded location from sensory input (inference)
        >>> p_x = memory.retrieve(x_, for_inference=True)
        >>>
        >>> # Retrieve grounded location from grid cells (generation)
        >>> p_g = memory.retrieve(g_, for_inference=False)
        >>>
        >>> # Update memory with Hebbian plasticity
        >>> memory.update(p_inferred, p_generated, eta=0.3)
        >>>
        >>> # Access underlying storage for advanced operations
        >>> M_gen = memory.storage.M_gen
        >>> all_memories = memory.get_all_memories()
    """

    def __init__(self, params: MemoryParams):
        """Initialize memory system with storage and attractor dynamics.

        Connectivity masks are computed internally based on the model parameters:
        - p_update_mask: Hierarchical mask for Hebbian learning
        - mask_inf: Conservative retrieval masks for stable inference
        - mask_gen: Flexible retrieval masks for generation

        Args:
            params: Memory system configuration including:
                   - n_p, n_f, n_f_g, n_f_ovc, f_extended (for mask computation)
                   - lambda_, kappa, common_memory, batch_size, i_attractor
                   - max_freq_inf, max_freq_gen (for attractor dynamics)
        """
        # Compute connectivity masks internally
        p_update_mask = utils.create_p_update_mask(params.n_p, params.n_f, params.n_f_g, params.n_f_ovc, params.f_extended)
        mask_inf = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_inf)
        mask_gen = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_gen)

        self.storage = storage.MemoryStorage(params, p_update_mask)
        self.attractor = attractor.AttractorDynamics(params, mask_inf, mask_gen)

    def retrieve(self, query: MultiScaleCode, for_inference: bool) -> MultiScaleCode:
        """Retrieve grounded location from memory via attractor dynamics.

        Performs pattern completion on the query using the appropriate memory
        matrix and retrieval masks based on the operation mode.

        Args:
            query: Initial query pattern as list of per-frequency tensors [B, n_p[f]]
                  Can be from sensory input (x_) or grid cells (g_)
            for_inference: If True, use inference memory (M_inf) and conservative masks
                          If False, use generative memory (M_gen) and flexible masks

        Returns:
            Refined grounded location after attractor convergence
            List of per-frequency tensors [B, n_p[f]]

        Note:
            - Inference mode: Conservative early-stopping for stable retrieval
            - Generative mode: More iterations for flexible pattern completion
        """
        M = self.storage.get_memory(for_inference=for_inference)
        return self.attractor(query, M, for_inference=for_inference)

    def update(self, p_inferred: Matrix, p_generated: Matrix, eta: float) -> None:
        """Update memory matrices using Hebbian plasticity.

        Implements the Hebbian update rule:
        M_gen = λ*M + η*(p_inf + p_gen) ⊗ (p_inf - p_gen)

        Args:
            p_inferred: Inferred grounded locations [B, sum(n_p)]
                       Typically from direct inference (g ⊗ x)
            p_generated: Generated grounded locations [B, sum(n_p)]
                        Typically from pattern completion on grid cells
            eta: Learning rate (remembering strength), typically 0.1-0.5
        """
        self.storage.update(p_inferred, p_generated, eta)

    def get_memory(self, for_inference: bool) -> Matrix:
        """Get the appropriate memory matrix for inference or generation.

        Args:
            for_inference: If True, return inference memory (M_inf or shared M_gen)
                          If False, return generative memory (M_gen)

        Returns:
            Memory matrix [B, sum(n_p), sum(n_p)]
        """
        return self.storage.get_memory(for_inference)

    def get_all_memories(self) -> List[Matrix]:
        """Get all memory matrices for checkpointing.

        Returns:
            List containing [M_gen] or [M_gen, M_inf] if dual-memory is enabled
        """
        return self.storage.get_all_memories()

    def set_memories(self, memories: List[Matrix]) -> None:
        """Restore memory matrices from checkpoint.

        Args:
            memories: List containing [M_gen] or [M_gen, M_inf]
        """
        self.storage.set_memories(memories)
