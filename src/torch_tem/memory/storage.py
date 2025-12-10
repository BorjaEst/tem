"""Hebbian memory storage for the Tolman-Eichenbaum Machine (TEM).

This module implements the associative memory system that stores spatial relationships
via Hebbian plasticity. The memory matrices learn associations between grounded locations
(place cells) by updating based on co-activation patterns during exploration.

In TEM, the memory system serves as the bridge between:
1. Inference: Retrieving grounded locations from sensory input (x → p via memory)
2. Generation: Predicting grounded locations from abstract transitions (g → p via memory)

The Hebbian update rule implements a biologically-inspired learning mechanism where
synaptic connections strengthen when pre- and post-synaptic neurons fire together,
with gradual decay (forgetting) over time.
"""

from typing import List, Protocol

import torch
from torch import Tensor

from ..types import Matrix, Vector


class StorageParams(Protocol):
    n_p: List[int]  # Dimensions of grounded location per frequency
    lambda_: float  # Memory retention factor
    common_memory: bool  # Whether to use a common memory for inference and generation
    batch_size: int  # Number of parallel environments / memory instances


class MemoryStorage:
    """Hebbian memory storage with batched parallel memories.

    Each batch element maintains its own independent memory matrix.
    All operations are vectorized using batch matrix operations.
    """

    def __init__(self, params: StorageParams, p_update_mask: Matrix):
        self.n_p = params.n_p
        self.lambda_ = params.lambda_
        self.use_dual_memory = not params.common_memory
        self.p_update_mask = p_update_mask
        self.batch_size = params.batch_size

        # Initialize memory matrices: [batch_size, sum(n_p), sum(n_p)]
        # Each batch element has its own independent memory
        n_p_total = sum(self.n_p)
        self.M_gen = torch.zeros(self.batch_size, n_p_total, n_p_total)
        self.M_inf = torch.zeros(self.batch_size, n_p_total, n_p_total) if self.use_dual_memory else None

    def update(self, p_inferred: Vector, p_generated: Vector, eta: float) -> None:
        """Update memory matrices using Hebbian plasticity.

        Args:
            p_inferred: Inferred grounded locations [B, N]
            p_generated: Generated grounded locations [B, N]
            eta: Learning rate (remembering strength)
        """
        # Move mask to same device
        mask = self.p_update_mask.to(p_inferred.device)

        # Generative memory update: M_gen = λ*M + η*(p_inf + p_gen) ⊗ (p_inf - p_gen)
        term1 = p_inferred + p_generated
        term2 = p_inferred - p_generated
        outer_gen = torch.bmm(term1.unsqueeze(2), term2.unsqueeze(1))  # [B, N, N]

        self.M_gen = self.M_gen.to(outer_gen.device)
        self.M_gen = torch.clamp(self.lambda_ * self.M_gen + eta * (outer_gen * mask), min=-1.0, max=1.0)

        # Inference memory update (if using dual-memory architecture)
        if self.use_dual_memory:
            self.M_inf = self.M_inf.to(outer_gen.device)
            self.M_inf = torch.clamp(self.lambda_ * self.M_inf + eta * outer_gen, min=-1.0, max=1.0)

    def get_memory(self, for_inference: bool = False) -> Matrix:
        """Retrieve memory matrix for attractor dynamics.

        Args:
            for_inference: If True and dual-memory is enabled, return inference memory

        Returns:
            Memory matrix [B, N, N] where B is batch size
        """
        if for_inference and self.M_inf is not None:
            return self.M_inf
        return self.M_gen

    def get_all_memories(self) -> List[Matrix]:
        """Get all memory matrices for checkpointing.

        Returns:
            List containing [M_gen] or [M_gen, M_inf] if dual-memory is enabled
        """
        if self.use_dual_memory:
            return [self.M_gen, self.M_inf]
        return [self.M_gen]

    def set_memories(self, memories: List[Matrix]) -> None:
        """Restore memory matrices from checkpoint.

        Args:
            memories: List containing [M_gen] or [M_gen, M_inf]
        """
        self.M_gen = memories[0]
        if self.use_dual_memory and len(memories) > 1:
            self.M_inf = memories[1]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Simple example demonstrating Hebbian memory storage and updates.

    This example shows how the memory system learns associations between grounded
    locations through repeated co-activation patterns during simulated navigation.
    """
    from torch_tem.config.architecture import ModelConfig

    # Create configuration with 3 frequency modules
    params = ModelConfig(
        n_g_subsampled=[10, 8, 6],  # 3 frequency modules (coarse to fine)
        n_x_c=5,  # 5 compressed sensory dimensions
        lambda_=0.95,  # 95% memory retention (slow forgetting)
        eta=0.3,  # 30% learning rate (moderate remembering)
        common_memory=False,  # Separate matrices for inference/generation
    )

    # Initialize memory storage
    storage = MemoryStorage(params)

    print(f"Memory dimensions: {sum(params.n_p)} place cells total")
    print(f"  Per frequency: {params.n_p}")
    print(f"Dual memory mode: {storage.use_dual_memory}")

    # Simulate a sequence of grounded locations during navigation
    batch_size = 8
    n_p_total = sum(params.n_p)

    # Simulate 5 timesteps of navigation
    for t in range(5):
        # Generate random grounded locations (in practice, computed by model)
        p_inferred = torch.randn(batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(batch_size, n_p_total).softmax(dim=1)

        # Update memory with Hebbian rule
        storage.update(p_inferred, p_generated, eta=params.eta)

        # Check memory strength (Frobenius norm)
        m_gen_strength = torch.norm(storage.M_gen)
        print(f"Timestep {t+1}: M_gen strength = {m_gen_strength:.4f}")

    # Retrieve memory for use in attractor dynamics
    M_gen = storage.get_memory(for_inference=False)
    M_inf = storage.get_memory(for_inference=True)

    print(f"\nFinal memory matrices:")
    print(f"  M_gen shape: {M_gen.shape}, mean: {M_gen.mean():.6f}")
    print(f"  M_inf shape: {M_inf.shape}, mean: {M_inf.mean():.6f}")
    print(f"  Difference: {(M_gen - M_inf).abs().mean():.6f}")

    # Save and restore memory state
    saved_memories = storage.get_all_memories()
    print(f"\nSaved {len(saved_memories)} memory matrices for checkpointing")
