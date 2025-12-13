"""Memory matrix generation for TEM examples.

This module provides utilities for generating pre-trained memory matrices
that can be used in generative and inference examples without requiring
online learning. These matrices approximate the Hebbian associations that
would form during training.

The generative pathway in the TEM manuscript uses a pre-learned memory
matrix M_gen without online updates during generation. This module provides
functions to create such matrices from walk data.
"""

from typing import List, Protocol, Tuple

import torch
from torch import Tensor

from ..types import HebbianMemory, Vector


class MemoryParams(Protocol):
    """Protocol for memory matrix generation configuration.

    Components implementing this protocol provide the necessary parameters
    for generating memory matrices.
    """

    n_p: List[int]
    lambda_: float
    eta: float


def generate_hebbian_memory(
    patterns_i: Tensor,
    patterns_j: Tensor,
    lambda_: float = 0.95,
    eta: float = 0.3,
) -> HebbianMemory:
    """Generate Hebbian memory matrix from paired patterns.

    Creates memory matrix through Hebbian learning rule:
        M = Σ_t η·(p_i^t ⊗ p_j^t)

    This simulates the memory that would form from observing the pattern
    pairs during training. For generative memory, both patterns are typically
    from the same source (M_gen uses p_g ⊗ p_g).

    Args:
        patterns_i: Pre-synaptic patterns [T, n_p] or [T, B, n_p]
        patterns_j: Post-synaptic patterns [T, n_p] or [T, B, n_p]
        lambda_: Memory decay rate (not used in batch processing)
        eta: Hebbian learning rate

    Returns:
        Memory matrix [n_p, n_p]

    Example:
        >>> # Generate memory from grid cell patterns
        >>> g_patterns = generate_grid_patterns(walk, model_config)
        >>> M_gen = generate_hebbian_memory(g_patterns, g_patterns, eta=0.3)
    """
    # Flatten batch dimension if present
    if patterns_i.dim() == 3:
        T, B, n_p = patterns_i.shape
        patterns_i = patterns_i.view(T * B, n_p)
        patterns_j = patterns_j.view(T * B, n_p)

    # Compute outer product sum: M = Σ_t η·(p_i ⊗ p_j)
    M = torch.zeros(patterns_i.shape[1], patterns_j.shape[1])
    for t in range(patterns_i.shape[0]):
        M += eta * torch.outer(patterns_i[t], patterns_j[t])

    # Average over timesteps to prevent unbounded growth
    M = M / patterns_i.shape[0]

    return M


def generate_memory_from_walk(
    walk_data: List[Tensor],
    projection_fn,
    lambda_: float = 0.95,
    eta: float = 0.3,
    for_inference: bool = False,
) -> HebbianMemory:
    """Generate memory matrix from walk trajectory.

    Processes walk data through projection function and builds Hebbian
    associations. This is the recommended way to create memory matrices
    for examples.

    Args:
        walk_data: List of patterns at each timestep (e.g., grid cells)
        projection_fn: Function to project patterns to hippocampus
        lambda_: Memory decay rate
        eta: Hebbian learning rate
        for_inference: If True, create inference memory (M_inf), else generative (M_gen)

    Returns:
        Memory matrix [sum(n_p), sum(n_p)]

    Example:
        >>> # Create generative memory
        >>> M_gen = generate_memory_from_walk(
        ...     g_history,
        ...     lambda g: mec_projection.repeat(mec_projection.downsample(g)),
        ...     eta=0.3,
        ...     for_inference=False
        ... )
    """
    # Project all patterns
    projected_patterns = []
    for pattern in walk_data:
        projected = projection_fn(pattern)
        if isinstance(projected, list):
            projected = torch.cat(projected, dim=-1)
        projected_patterns.append(projected)

    patterns = torch.stack(projected_patterns)  # [T, n_p]

    # Generate memory from self-associations
    return generate_hebbian_memory(patterns, patterns, lambda_, eta)


def generate_random_memory(
    n_p: int,
    sparsity: float = 0.1,
    scale: float = 0.01,
) -> HebbianMemory:
    """Generate random sparse memory matrix.

    Creates a random memory matrix with sparse structure, useful for
    quick prototyping or when walk data is not available.

    Args:
        n_p: Total number of hippocampal units
        sparsity: Fraction of non-zero elements
        scale: Scale of random weights

    Returns:
        Random sparse memory matrix [n_p, n_p]
    """
    M = torch.randn(n_p, n_p) * scale

    # Apply sparsity
    mask = torch.rand(n_p, n_p) < sparsity
    M = M * mask

    # Symmetrize for stability
    M = (M + M.T) / 2

    return M


def generate_block_diagonal_memory(
    n_p_list: List[int],
    scale: float = 0.01,
    inter_block_scale: float = 0.001,
) -> HebbianMemory:
    """Generate block-diagonal memory matrix.

    Creates memory with strong within-frequency associations and weak
    cross-frequency associations, matching the hierarchical structure
    of TEM.

    Args:
        n_p_list: Number of units per frequency [n_p[0], n_p[1], ...]
        scale: Scale of within-block weights
        inter_block_scale: Scale of between-block weights

    Returns:
        Block-diagonal memory matrix [sum(n_p), sum(n_p)]

    Example:
        >>> # For model with n_p = [96, 80, 64]
        >>> M = generate_block_diagonal_memory([96, 80, 64], scale=0.01)
    """
    n_p_total = sum(n_p_list)
    M = torch.randn(n_p_total, n_p_total) * inter_block_scale

    # Add strong diagonal blocks
    offset = 0
    for n_p in n_p_list:
        block = torch.randn(n_p, n_p) * scale
        block = (block + block.T) / 2  # Symmetrize
        M[offset : offset + n_p, offset : offset + n_p] = block
        offset += n_p

    return M


class MemoryMatrixGenerator:
    """Generates pre-trained memory matrices for TEM examples.

    This class provides a high-level interface for creating memory matrices
    from various sources, suitable for use in generative and inference examples.
    """

    def __init__(self, params: MemoryParams):
        """Initialize memory matrix generator.

        Args:
            params: Configuration with n_p, lambda_, eta
        """
        self.params = params
        self.n_p_total = sum(params.n_p)

    def from_patterns(
        self,
        patterns: Tensor,
        for_inference: bool = False,
    ) -> Tuple[HebbianMemory, HebbianMemory]:
        """Generate both M_gen and M_inf from patterns.

        Args:
            patterns: Pattern sequence [T, n_p_total]
            for_inference: Unused, kept for API compatibility

        Returns:
            Tuple of (M_gen, M_inf) memory matrices
        """
        M_gen = generate_hebbian_memory(patterns, patterns, self.params.lambda_, self.params.eta)
        M_inf = M_gen.clone()  # In simple case, same memory

        return M_gen, M_inf

    def from_walk(
        self,
        walk_patterns: List[List[Tensor]],
        projection_fn,
    ) -> Tuple[HebbianMemory, HebbianMemory]:
        """Generate memory matrices from walk data.

        Args:
            walk_patterns: List of pattern lists at each timestep
            projection_fn: Function to project to hippocampus

        Returns:
            Tuple of (M_gen, M_inf) memory matrices
        """
        # Project and flatten patterns
        projected = []
        for patterns in walk_patterns:
            proj = projection_fn(patterns)
            if isinstance(proj, list):
                proj = torch.cat(proj, dim=-1)
            projected.append(proj)

        patterns = torch.stack(projected)  # [T, n_p_total]
        return self.from_patterns(patterns)

    def random(
        self,
        sparsity: float = 0.1,
        scale: float = 0.01,
    ) -> Tuple[HebbianMemory, HebbianMemory]:
        """Generate random memory matrices.

        Args:
            sparsity: Fraction of non-zero elements
            scale: Scale of random weights

        Returns:
            Tuple of (M_gen, M_inf) memory matrices
        """
        M_gen = generate_random_memory(self.n_p_total, sparsity, scale)
        M_inf = generate_random_memory(self.n_p_total, sparsity, scale)

        return M_gen, M_inf

    def block_diagonal(
        self,
        scale: float = 0.01,
        inter_block_scale: float = 0.001,
    ) -> Tuple[HebbianMemory, HebbianMemory]:
        """Generate block-diagonal memory matrices.

        Args:
            scale: Scale of within-block weights
            inter_block_scale: Scale of between-block weights

        Returns:
            Tuple of (M_gen, M_inf) memory matrices
        """
        M = generate_block_diagonal_memory(self.params.n_p, scale, inter_block_scale)
        return M.clone(), M.clone()
