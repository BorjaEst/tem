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

import torch
from torch import Tensor

from torch_tem.config.facets import AttractorParams


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

    def __init__(self, params: AttractorParams):
        """Initialize attractor dynamics with hierarchical retrieval masks.

        Args:
            params: Protocol providing attractor configuration (kappa, iterations, masks)
        """
        self.kappa = params.kappa
        self.i_attractor = params.i_attractor_calculated
        self.p_retrieve_mask_inf = params.p_retrieve_mask_inf_calculated
        self.p_retrieve_mask_gen = params.p_retrieve_mask_gen_calculated

    def retrieve(self, p_query: Tensor, M: Tensor, for_inference: bool = False) -> Tensor:
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
            p_retrieved: Refined grounded location pattern [B, sum(n_p)] after convergence

        Note:
            The hierarchical masking implements the following schedule:
            - Early iterations: Only low-frequency modules updated (coarse spatial scale)
            - Middle iterations: Mid-frequency modules gradually enabled
            - Late iterations: All frequencies active (fine spatial detail)

            This prevents high-frequency noise from destabilizing the coarse spatial
            representation during early retrieval.
        """
        # Select appropriate hierarchical masks based on retrieval mode
        retrieve_mask = self.p_retrieve_mask_inf if for_inference else self.p_retrieve_mask_gen

        p = p_query
        for it in range(self.i_attractor):
            # Memory readout: Query the Hebbian matrix (associative recall)
            # Matrix multiply retrieves patterns associated with current state
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
    n_p_total = sum(params.n_p_calculated)
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
