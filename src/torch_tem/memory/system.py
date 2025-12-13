"""Unified memory system combining storage and retrieval for TEM.

This module provides a high-level Memory class that encapsulates both
Hebbian memory storage and attractor-based retrieval, managing the
dual-memory architecture (M_gen/M_inf) internally.
"""

from typing import List, Protocol

from .. import utils
from ..types import Matrix, MultiScaleCode
from .attractor import AttractorDynamics
from .storage import MemoryStorage


class MemoryParams(Protocol):
    """Combined parameters for memory system."""

    n_p: List[int]  # Dimensions of grounded location per frequency
    n_f: int  # Total number of frequency modules
    n_f_g: int  # Number of grid cell frequency modules
    n_f_ovc: int  # Number of OVC frequency modules
    f_extended: List[float]  # Extended frequency list including OVC modules
    lambda_: float  # Memory retention factor
    kappa: float  # Attractor stability parameter
    common_memory: bool  # Whether to use a common memory for inference and generation
    batch_size: int  # Number of parallel environments / memory instances
    i_attractor: int  # Number of attractor iterations
    max_freq_inf: List[int]  # Max iterations per frequency for inference
    max_freq_gen: List[int]  # Max iterations per frequency for generation


class Memory:
    """Unified memory system combining storage and retrieval.

    Handles both Hebbian memory storage and attractor-based retrieval,
    managing dual-memory architecture (M_gen/M_inf) internally. This class
    provides a clean, high-level API for memory operations in TEM.

    The memory system bridges inference and generation:
    - Inference: Retrieving grounded locations from sensory input (x → p)
    - Generation: Predicting grounded locations from grid cells (g → p)

    Connectivity masks are computed internally from the model configuration,
    simplifying the initialization and ensuring consistency.

    Example:
        >>> memory = Memory(params)
        >>> # Retrieve from sensory input
        >>> p_x = memory.retrieve(x_, for_inference=True)
        >>> # Retrieve from grid cells
        >>> p_g = memory.retrieve(g_, for_inference=False)
        >>> # Update memory
        >>> memory.update(p_inferred, p_generated, eta=0.3)
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

        self.storage = MemoryStorage(params, p_update_mask)
        self.attractor = AttractorDynamics(params, mask_inf, mask_gen)

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
