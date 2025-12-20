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

from torch_tem.types import Matrix, MultiScaleCode


class AttractorParams(Protocol):
    """Architecture parameters needed by AttractorDynamics."""

    i_attractor: int  # Number of attractor iterations
    kappa: float  # Decay factor for attractor updates
    n_p: List[int]  # Dimensions of grounded location per frequency


class AttractorDynamics:
    """Iterative memory retrieval via attractor dynamics.

    Refines grounded location representations by iteratively querying the Hebbian
    memory matrix. Implements hierarchical coarse-to-fine refinement where low-frequency
    (coarse spatial) components converge first, followed by high-frequency (fine detail)
    components.

    The attractor update rule is:
        p_new = mask * activation(κ * p_old + M @ p_old) + (1-mask) * p_old

    where mask progressively enables higher frequencies across iterations.
    """

    def __init__(self, params: AttractorParams, mask_inf: List[Matrix], mask_gen: List[Matrix]):
        """Initialize attractor dynamics with hierarchical retrieval masks.

        Args:
            params: Model configuration with kappa, i_attractor, and n_p
            mask_inf: Hierarchical masks for inference retrieval [i_attractor x sum(n_p)]
            mask_gen: Hierarchical masks for generative retrieval [i_attractor x sum(n_p)]
        """
        self.kappa = params.kappa
        self.i_attractor = params.i_attractor
        self.n_p = params.n_p
        self.p_retrieve_mask_inf = mask_inf
        self.p_retrieve_mask_gen = mask_gen

    def __call__(self, p_query: MultiScaleCode, M: Matrix, for_inference: bool = False) -> MultiScaleCode:
        """Retrieve refined grounded location from memory via attractor dynamics.

        Args:
            p_query: Initial query pattern as list of per-frequency tensors [B, n_p_f]
            M: Hebbian memory matrix [B, sum(n_p), sum(n_p)]
            for_inference: If True, use inference masks (more conservative early-stopping)

        Returns:
            Refined grounded location as list of per-frequency tensors [B, n_p_f]

        Note:
            The retrieval is hierarchical: low-frequency (coarse) components stabilize
            first, providing a stable foundation for high-frequency (fine) refinement.
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
    by iteratively pulling it toward stored patterns in the memory matrix,
    using hierarchical coarse-to-fine refinement.
    """
    from torch_tem.config.architecture import ModelConfig
    from torch_tem.utils.masks import create_p_retrieve_mask

    # Create configuration
    params = ModelConfig(
        n_g_subsampled=[5, 4, 3],  # 3 frequency modules (coarse to fine)
        n_x_c=3,  # Compressed sensory dimensions
        kappa=0.8,  # Attractor decay factor
    )

    # Create hierarchical retrieval masks
    # Inference: conservative early-stopping (more stable)
    # Generative: less early-stopping (all frequencies active)
    mask_inf = create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_inf)
    mask_gen = create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_gen)

    # Initialize attractor dynamics
    attractor = AttractorDynamics(params, mask_inf=mask_inf, mask_gen=mask_gen)

    # Create a simple memory matrix (normally learned via Hebbian updates)
    batch_size = 4
    n_p_total = sum(params.n_p)

    # Batched memory matrix (in practice, learned during training)
    M = torch.randn(batch_size, n_p_total, n_p_total) * 0.01
    # Make approximately symmetric for stable dynamics
    M = (M + M.transpose(1, 2)) / 2

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
