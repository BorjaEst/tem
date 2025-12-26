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

from typing import List

import torch
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.types import Matrix, MultiScaleCode


class AttractorConfig(BaseModel):
    """Attractor dynamics configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    kappa: float = Field(default=0.8, ge=0, le=1, description="Decay factor for attractor updates (κ in attractor update)")


class AttractorDynamics:
    """Iterative memory retrieval via attractor dynamics.

    Refines grounded location representations by iteratively querying the Hebbian
    memory matrix. Implements hierarchical coarse-to-fine refinement where low-frequency
    (coarse spatial) components converge first, followed by high-frequency (fine detail)
    components.

    The attractor update rule is:
        p_new = mask * activation(κ * p_old + M @ p_old) + (1-mask) * p_old

    where mask progressively enables higher frequencies across iterations.
    ...
    """

    def __init__(self, mask_inf: List[Matrix], mask_gen: List[Matrix], config: AttractorConfig):
        """Initialize attractor dynamics with hierarchical retrieval masks.

        Args:
            mask_inf: Hierarchical masks for inference retrieval [i_attractor] of (sum(n_p),)
            mask_gen: Hierarchical masks for generative retrieval [i_attractor] of (sum(n_p),)
            config: Hyperparameters for attractor dynamics (kappa decay factor)
        """
        super().__init__()
        self._config = config
        self._retrieve_mask_inf = mask_inf
        self._retrieve_mask_gen = mask_gen

    @property
    def i_attractor(self) -> int:
        """Number of attractor iterations (inferred from mask count)."""
        return len(self._retrieve_mask_inf)

    @property
    def n_p(self) -> List[int]:
        """Place cell dimensions per frequency."""
        return [mask.shape[0] for mask in self._retrieve_mask_inf[0]]

    @property
    def kappa(self) -> float:
        """Decay factor for attractor updates."""
        return self._config.kappa

    @kappa.setter
    def kappa(self, value: float):
        """Set decay factor for attractor updates."""
        self._config.kappa = value

    def __call__(self, p_query: MultiScaleCode, M: Matrix, for_inference: bool = False) -> MultiScaleCode:
        """Retrieve refined grounded location from memory via attractor dynamics.

        Args:
            p_query (MultiScaleCode): Initial query pattern as list of per-frequency tensors [B, n_p_f].
            M (Matrix): Hebbian memory matrix [B, sum(n_p), sum(n_p)].
            for_inference (bool): If True, use inference masks (more conservative early-stopping). Defaults to False.

        Returns:
            MultiScaleCode: Refined grounded location as list of per-frequency tensors [B, n_p_f].

        Note:
            The retrieval is hierarchical: low-frequency (coarse) components stabilize
            first, providing a stable foundation for high-frequency (fine) refinement.
        """
        # Select appropriate hierarchical masks based on retrieval mode
        retrieve_mask = self._retrieve_mask_inf if for_inference else self._retrieve_mask_gen

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
            p_candidate = self._config.kappa * p + p_readout
            p_candidate = torch.nn.functional.leaky_relu(torch.clamp(p_candidate, min=-1.0, max=1.0))

            # Apply hierarchical mask for coarse-to-fine refinement
            # Early iterations update only low-frequency (coarse) components
            # Later iterations progressively enable higher frequencies (finer detail)
            mask = retrieve_mask[tau].unsqueeze(0).to(p.device)

            # Update only active frequencies, keep others unchanged
            p = (1 - mask) * p + mask * p_candidate

        # Split concatenated result back into per-frequency list (like legacy)
        n_p_cumsum = [0] + torch.cumsum(torch.tensor(self._n_p), dim=0).tolist()
        p_list = [p[:, n_p_cumsum[f] : n_p_cumsum[f + 1]] for f in range(len(self._n_p))]

        return p_list


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Attractor dynamics example: hierarchical memory retrieval.

    Demonstrates how attractor dynamics refine noisy query patterns via
    iterative convergence with hierarchical early-stopping.
    """
    from dataclasses import dataclass

    print("=" * 80)
    print("Attractor Dynamics Example - Hierarchical Memory Retrieval")
    print("=" * 80)

    # Configuration
    n_p = [10, 10, 8, 6, 6]  # Place cells per frequency
    n_p_total = sum(n_p)
    i_attractor = 3
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Frequencies: {len(n_p)}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Total place cells: {n_p_total}")
    print(f"  Attractor iterations: {i_attractor}")

    # Create simple context
    @dataclass
    class SimpleAttractorContext:
        """Minimal context for demonstration."""

        n_p: List[int]
        mask_inference: List[torch.Tensor]
        mask_generative: List[torch.Tensor]

    # Simple hierarchical masks (all frequencies active at all iterations)
    masks = [torch.ones(n_p_total) for _ in range(i_attractor)]

    # Create configuration and attractor
    config = AttractorConfig(kappa=0.8)
    attractor = AttractorDynamics(masks, masks, n_p, config)
    print(f"\n✓ Attractor initialized (κ={config.kappa}, iterations={attractor.i_attractor})")

    # Create memory matrix (normally learned via Hebbian updates)
    M = torch.randn(batch_size, n_p_total, n_p_total) * 0.01
    M = (M + M.transpose(1, 2)) / 2  # Symmetric for stable dynamics
    print(f"✓ Memory matrix: {M.shape}")

    # Create noisy query pattern
    p_target = [torch.randn(batch_size, n) * 0.5 for n in n_p]
    noise = [torch.randn(batch_size, n) * 0.2 for n in n_p]
    p_query = [t + n for t, n in zip(p_target, noise)]
    print(f"✓ Query pattern: {[p.shape for p in p_query]}")

    # Retrieve via attractor dynamics
    p_retrieved = attractor(p_query, M, for_inference=True)
    print(f"✓ Retrieved pattern: {[p.shape for p in p_retrieved]}")

    # Evaluate retrieval quality
    query_flat = torch.cat(p_query, dim=1)
    retrieved_flat = torch.cat(p_retrieved, dim=1)
    target_flat = torch.cat(p_target, dim=1)

    cosine_before = torch.nn.functional.cosine_similarity(query_flat, target_flat, dim=1).mean()
    cosine_after = torch.nn.functional.cosine_similarity(retrieved_flat, target_flat, dim=1).mean()

    print(f"\nRetrieval quality:")
    print(f"  Query → Target similarity: {cosine_before:.4f}")
    print(f"  Retrieved → Target similarity: {cosine_after:.4f}")
    print(f"  Improvement: {(cosine_after - cosine_before):.4f}")

    print("\n" + "=" * 80)
