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

from ..types import Matrix, MultiScaleCode


class ModelParams(Protocol):
    """Architecture parameters needed by AttractorDynamics."""

    i_attractor: int
    kappa: float
    n_p: List[int]


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
        mask_inf: Hierarchical masks for inference mode retrieval
        mask_gen: Hierarchical masks for generative mode retrieval
    """

    def __init__(self, model_params: ModelParams, mask_inf: List[Matrix], mask_gen: List[Matrix]):
        """Initialize attractor dynamics with hierarchical retrieval masks.

        Args:
            model_params: Architecture configuration (i_attractor, kappa, n_p)
            mask_inf: Hierarchical masks for inference retrieval
            mask_gen: Hierarchical masks for generative retrieval
        """
        self.kappa = model_params.kappa
        self.i_attractor = model_params.i_attractor
        self.n_p = model_params.n_p
        self.p_retrieve_mask_inf = mask_inf
        self.p_retrieve_mask_gen = mask_gen

    def __call__(self, p_query: MultiScaleCode, M: Matrix, for_inference: bool = False) -> MultiScaleCode:
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
            p_query: Initial query pattern representing grounded location per frequency
                    List of [B, n_p[f]] tensors (one per frequency module)
            M: Hebbian memory matrix storing location associations [sum(n_p), sum(n_p)]
               learned via outer product updates: M = λ*M + η*outer(p_inf, p_gen)
            for_inference: If True, use inference masks; if False, use generative masks
                          (inference typically requires more iterations for stable convergence)

        Returns:
            p_retrieved: Refined grounded location pattern per frequency module.
                        **Format**: List of [B, n_p[f]] tensors (one per frequency).
                        Ready for direct use by downstream TEM components.

        Note:
            The hierarchical masking implements the following schedule:
            - Early iterations: Only low-frequency modules updated (coarse spatial scale)
            - Middle iterations: Mid-frequency modules gradually enabled
            - Late iterations: All frequencies active (fine spatial detail)

            This prevents high-frequency noise from destabilizing the coarse spatial
            representation during early retrieval.

        Example:
            >>> # Memory retrieval with per-frequency list format
            >>> p_retrieved = attractor(p_query_list, M_inf, for_inference=True)
            >>> # Ready for AbstractLocInference
            >>> g_inf = abstract(g_gen, sigma_gen, p_retrieved, ...)
        """
        # Select appropriate hierarchical masks based on retrieval mode
        retrieve_mask = self.p_retrieve_mask_inf if for_inference else self.p_retrieve_mask_gen

        # Concatenate per-frequency query into single vector
        p = torch.cat(p_query, dim=1)

        # Apply activation to initial query (stability)
        p = torch.nn.functional.leaky_relu(torch.clamp(p, min=-1.0, max=1.0))

        for tau in range(self.i_attractor):
            # Memory readout: Query the Hebbian matrix (associative recall)
            # Matrix multiply retrieves patterns associated with current state
            # Handle batched [B, n_p, n_p] memory
            p_readout = torch.matmul(p.unsqueeze(1), M.to(p.device)).squeeze(1)

            # Calculate candidate update with decay and activation
            p_candidate = self.kappa * p + p_readout
            p_candidate = torch.nn.functional.leaky_relu(torch.clamp(p_candidate, min=-1.0, max=1.0))

            # Apply hierarchical mask for coarse-to-fine refinement
            # Early iterations update only low-frequency (coarse) components
            # Later iterations progressively enable higher frequencies (finer detail)
            mask = retrieve_mask[tau].unsqueeze(0).to(p.device)

            # Update only active frequencies, keep others unchanged
            p = (1 - mask) * p + mask * p_candidate

        # Split concatenated result back into per-frequency list (like legacy)
        n_p_cumsum = [0] + torch.cumsum(torch.tensor(self.n_p), dim=0).tolist()
        p_list = [p[:, n_p_cumsum[f] : n_p_cumsum[f + 1]] for f in range(len(self.n_p))]

        return p_list


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Simple example demonstrating attractor dynamics retrieval.

    This example shows how attractor dynamics refine a noisy query pattern
    by iteratively pulling it toward stored patterns in the memory matrix.
    """
    import types

    # Create simple params with required fields
    params = types.SimpleNamespace(
        kappa=0.8,
        i_attractor=3,
        n_p=[15, 12, 10],  # 3 frequency modules with different dimensions
    )

    # Create simple masks (all ones for this demo)
    n_p_total = sum(params.n_p)
    mask_inf = [torch.ones(n_p_total) for _ in range(params.i_attractor)]
    mask_gen = [torch.ones(n_p_total) for _ in range(params.i_attractor)]

    # Initialize attractor dynamics
    attractor = AttractorDynamics(params, mask_inf=mask_inf, mask_gen=mask_gen)

    # Create a simple memory matrix (normally learned via Hebbian updates)
    batch_size = 4

    # Random memory matrix (in practice, this is learned during training)
    M = torch.randn(n_p_total, n_p_total) * 0.01
    M = M + M.T  # Make symmetric for stable dynamics

    # Noisy query pattern representing uncertain grounded location (per-frequency list)
    p_query = [torch.randn(batch_size, n_p) * 0.5 for n_p in params.n_p]

    # Retrieve refined pattern from memory
    p_retrieved = attractor(p_query, M, for_inference=True)

    print(f"Query pattern (list of {len(p_query)} frequencies)")
    for f, p in enumerate(p_query):
        print(f"  Frequency {f}: shape {tuple(p.shape)}")
    print(f"\nRetrieved pattern (list of {len(p_retrieved)} frequencies)")
    for f, p in enumerate(p_retrieved):
        print(f"  Frequency {f}: shape {tuple(p.shape)}")
    print(f"\nQuery mean activation: {torch.cat(p_query, dim=1).abs().mean():.4f}")
    print(f"Retrieved mean activation: {torch.cat(p_retrieved, dim=1).abs().mean():.4f}")
    print("\nHierarchical mask schedule:")
    for it, mask in enumerate(attractor.p_retrieve_mask_inf):
        print(f"  Iteration {it}: {mask.sum().item()}/{n_p_total} neurons active")
