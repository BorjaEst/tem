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
    eta: float  # Learning rate for memory updates
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
        self.eta = params.eta
        self.use_dual_memory = not params.common_memory
        self.p_update_mask = p_update_mask
        self.batch_size = params.batch_size

    def update(self, p_inferred: Vector, p_generated: Vector, M: Matrix) -> Matrix:
        """Update memory matrix using Hebbian plasticity (functional interface).

        Args:
            p_inferred: Inferred grounded locations [B, N]
            p_generated: Generated grounded locations [B, N]
            M: Current memory matrix [B, N, N]

        Returns:
            Updated memory matrix [B, N, N]
        """
        # Move mask to same device
        mask = self.p_update_mask.to(p_inferred.device)
        M = M.to(p_inferred.device)

        # Hebbian update: M_new = λ*M + η*(p_inf + p_gen) ⊗ (p_inf - p_gen)
        term1 = p_inferred + p_generated
        term2 = p_inferred - p_generated
        outer = torch.bmm(term1.unsqueeze(2), term2.unsqueeze(1))  # [B, N, N]

        # Apply mask only to generative memory (inference memory uses full outer product)
        update_term = outer * mask if mask is not None else outer
        return torch.clamp(self.lambda_ * M + self.eta * update_term, min=-1.0, max=1.0)


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
