"""Attractor dynamics for the Tolman-Eichenbaum Machine (TEM).

This module implements iterative memory retrieval via attractor dynamics, a core mechanism
in TEM's Hebbian associative memory system. The attractor dynamics refine grounded location
representations (hippocampal place cells) by iteratively querying the learned memory matrix,
with hierarchical early-stopping to stabilize low-frequency components first.

In TEM, grounded locations p represent the conjunction of abstract locations g (grid cells)
and sensory observations x (place cells = grid cells ⊗ sensory input). The memory stores
associations between these grounded locations via Hebbian learning, enabling recall of
spatial relationships and predictive inference.
"""

from typing import List, Protocol

import torch
from torch import Tensor

from ..types import Matrix, Vector


class ModelParams(Protocol):
    """Architecture parameters needed by AttractorDynamics."""

    i_attractor: int
    kappa: float


class AttractorDynamics:
    """Attractor-based memory retrieval with hierarchical early-stopping.

    In the TEM architecture, attractor dynamics implement content-addressable memory
    retrieval by iteratively refining a query pattern through matrix multiplication
    with the learned Hebbian memory matrix. This process converges toward stored
    patterns, enabling the model to recall spatial relationships and predict locations.

    The hierarchical masking mechanism implements early-stopping for low-frequency
    modules first, ensuring coarse spatial representations stabilize before fine-grained
    details are refined. This mirrors the hierarchical organization of grid cells in
    the entorhinal cortex, where low-frequency modules have larger spatial scales.

    Mathematical formulation:
        p[t+1] = κ * p[t] + (M^T @ p[t]) * mask[t]

    Where:
        - p[t]: Current grounded location pattern at iteration t
        - κ (kappa): Decay term controlling stability (0 < κ < 1)
        - M: Hebbian memory matrix [sum(n_p), sum(n_p)]
        - mask[t]: Hierarchical mask enabling progressive refinement

    Attributes:
        kappa: Decay factor for attractor update (controls stability)
        i_attractor: Number of attractor iterations (typically equals n_f_g)
        p_retrieve_mask_inf: Hierarchical masks for inference mode retrieval
        p_retrieve_mask_gen: Hierarchical masks for generative mode retrieval
    """

    def __init__(
        self,
        model_params: ModelParams,
        p_retrieve_mask_inf: List[Tensor],
        p_retrieve_mask_gen: List[Tensor],
    ):
        """Initialize attractor dynamics with hierarchical retrieval masks.

        Args:
            model_params: Architecture configuration (i_attractor, kappa)
            p_retrieve_mask_inf: Hierarchical masks for inference retrieval
            p_retrieve_mask_gen: Hierarchical masks for generative retrieval
        """
        self.kappa = model_params.kappa
        self.i_attractor = model_params.i_attractor
        self.p_retrieve_mask_inf = p_retrieve_mask_inf
        self.p_retrieve_mask_gen = p_retrieve_mask_gen

    def retrieve(self, p_query: Vector, M: Matrix, for_inference: bool = False) -> Vector:
        """Retrieve grounded location from memory via iterative attractor dynamics.

        Implements content-addressable memory recall by iteratively refining the query
        pattern through multiplication with the Hebbian memory matrix. Each iteration
        applies a hierarchical mask that progressively enables refinement of higher
        frequency modules, ensuring coarse-to-fine convergence.

        This process is central to TEM's ability to:
        1. Infer current location from sensory input (inference mode)
        2. Predict next location from abstract transitions (generative mode)
        3. Recall stored spatial relationships via associative memory

        Args:
            p_query: Initial query pattern representing grounded location [B, sum(n_p)]
                    where sum(n_p) is the total number of place cells across all frequencies
            M: Hebbian memory matrix storing location associations [sum(n_p), sum(n_p)]
               learned via outer product updates: M = λ*M + η*outer(p_inf, p_gen)
            for_inference: If True, use inference masks; if False, use generative masks
                          (inference typically requires more iterations for stable convergence)

        Returns:
            p_retrieved: Refined grounded location pattern [B, sum(n_p)] after convergence.
                        **Format**: Concatenated tensor across all frequencies.
                        Use `utils.split_to_frequencies(p_retrieved, n_p)` to convert to
                        per-frequency list format for hierarchical processing components.

        Note:
            The hierarchical masking implements the following schedule:
            - Early iterations: Only low-frequency modules updated (coarse spatial scale)
            - Middle iterations: Mid-frequency modules gradually enabled
            - Late iterations: All frequencies active (fine spatial detail)

            This prevents high-frequency noise from destabilizing the coarse spatial
            representation during early retrieval.

        Example:
            >>> # Memory retrieval returns concatenated format
            >>> p_retrieved = attractor.retrieve(query, M_inf, for_inference=True)
            >>> # Convert to per-frequency format for hierarchical inference
            >>> from torch_tem.utils import split_to_frequencies
            >>> p_list = split_to_frequencies(p_retrieved, model_config.n_p)
            >>> # Now ready for AbstractLocInference
            >>> g_inf = abstract(g_gen, sigma_gen, p_list, ...)
        """
        # Select appropriate hierarchical masks based on retrieval mode
        retrieve_mask = self.p_retrieve_mask_inf if for_inference else self.p_retrieve_mask_gen

        p = p_query
        for it in range(self.i_attractor):
            # Memory readout: Query the Hebbian matrix (associative recall)
            # Matrix multiply retrieves patterns associated with current state
            # Handle both batched [B, n_p, n_p] and unbatched [n_p, n_p] memory
            if M.ndim == 3:
                # Batched memory: [B, n_p, n_p] requires unsqueeze/squeeze
                p_update = torch.matmul(p.unsqueeze(1), M.to(p.device)).squeeze(1)
            else:
                # Unbatched memory: [n_p, n_p] uses standard matmul
                p_update = torch.matmul(p, M.to(p.device))

            # Apply hierarchical mask for coarse-to-fine refinement
            # Early iterations update only low-frequency (coarse) components
            # Later iterations progressively enable higher frequencies (finer detail)
            mask = retrieve_mask[it].unsqueeze(0).to(p.device)
            p_update = p_update * mask

            # Attractor update: Combine decay with memory-driven update
            # κ term provides stability, M^T @ p term pulls toward stored patterns
            p = self.kappa * p + p_update

        return p


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Simple example demonstrating attractor dynamics retrieval.

    This example shows how attractor dynamics refine a noisy query pattern
    by iteratively pulling it toward stored patterns in the memory matrix.
    """
    from torch_tem.config.parameters import Parameters

    # Create configuration with 3 frequency modules
    params = Parameters(
        n_g_subsampled=[10, 8, 6],  # 3 frequency modules
        n_x_c=5,  # 5 compressed sensory dimensions
        kappa=0.8,  # Moderate decay for stable convergence
        i_attractor=3,  # 3 iterations (one per frequency)
    )

    # Initialize attractor dynamics
    attractor = AttractorDynamics(params)

    # Create a simple memory matrix (normally learned via Hebbian updates)
    n_p_total = sum(params.n_p)
    batch_size = 4

    # Random memory matrix (in practice, this is learned during training)
    M = torch.randn(n_p_total, n_p_total) * 0.01
    M = M + M.T  # Make symmetric for stable dynamics

    # Noisy query pattern representing uncertain grounded location
    p_query = torch.randn(batch_size, n_p_total) * 0.5

    # Retrieve refined pattern from memory
    p_retrieved = attractor.retrieve(p_query, M, for_inference=True)

    print(f"Query pattern shape: {p_query.shape}")
    print(f"Retrieved pattern shape: {p_retrieved.shape}")
    print(f"Query mean activation: {p_query.abs().mean():.4f}")
    print(f"Retrieved mean activation: {p_retrieved.abs().mean():.4f}")
    print("\nHierarchical mask schedule:")
    for it, mask in enumerate(attractor.p_retrieve_mask_inf):
        print(f"  Iteration {it}: {mask.sum().item()}/{n_p_total} neurons active")
