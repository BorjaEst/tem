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
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.types import Matrix, Vector


class StorageConfig(BaseModel):
    """Memory storage configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    lambda_: float = Field(default=0.9, ge=0, le=1, description="Memory retention factor (λ in memory decay)")
    eta: float = Field(default=0.5, ge=0, le=1, description="Learning rate for memory updates (η in memory update)")


class StorageContext(Protocol):
    """Protocol for memory storage context providing architecture parameters."""

    update_mask: Matrix  # Mask for memory updates


class MemoryStorage:
    """Hebbian memory storage with batched parallel memories.

    Each batch element maintains its own independent memory matrix.
    All operations are vectorized using batch matrix operations.
    """

    def __init__(self, context: StorageContext, config: StorageConfig):
        """Initialize memory storage with context and configuration.

        Args:
            context (StorageContext): Architectural context providing update masks.
            config (StorageConfig): Hyperparameters for Hebbian learning.
        """
        super().__init__()
        self._config = config
        self.p_update_mask = context.update_mask

    @property
    def config(self) -> StorageConfig:
        """Return the storage configuration.

        Returns:
            StorageConfig: Hebbian learning hyperparameters.
        """
        return self._config

    def update(self, p_inferred: Vector, p_generated: Vector, M: Matrix) -> Matrix:
        """Update memory matrix using Hebbian plasticity (functional interface).

        Args:
            p_inferred (Vector): Inferred grounded locations [B, N].
            p_generated (Vector): Generated grounded locations [B, N].
            M (Matrix): Current memory matrix [B, N, N].

        Returns:
            Matrix: Updated memory matrix [B, N, N].
        """
        lambda_, eta = self.config.lambda_, self.config.eta

        # Move mask to same device
        mask = self.p_update_mask.to(p_inferred.device)
        M = M.to(p_inferred.device)

        # Hebbian update: M_new = λ*M + η*(p_inf + p_gen) ⊗ (p_inf - p_gen)
        term1 = p_inferred + p_generated
        term2 = p_inferred - p_generated
        outer = torch.bmm(term1.unsqueeze(2), term2.unsqueeze(1))  # [B, N, N]

        # Apply mask only to generative memory (inference memory uses full outer product)
        update_term = outer * mask if mask is not None else outer
        return torch.clamp(lambda_ * M + eta * update_term, min=-1.0, max=1.0)


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Memory storage example: Hebbian learning.

    Demonstrates how memory matrices learn associations between grounded
    locations through repeated Hebbian updates.
    """
    from dataclasses import dataclass

    print("=" * 80)
    print("Memory Storage Example - Hebbian Plasticity")
    print("=" * 80)

    # Configuration
    n_p_total = 40  # Total place cells
    batch_size = 4
    n_steps = 5

    print(f"\nConfiguration:")
    print(f"  Place cells: {n_p_total}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training steps: {n_steps}")

    # Create simple context
    @dataclass
    class SimpleStorageContext:
        """Minimal context for demonstration."""

        update_mask: torch.Tensor

    # Create update mask (allow all connections)
    ctx = SimpleStorageContext(update_mask=torch.ones(n_p_total, n_p_total))

    # Create configuration and storage
    config = StorageConfig(lambda_=0.9, eta=0.5)
    storage = MemoryStorage(ctx, config)
    print(f"\n✓ Memory storage initialized (λ={config.lambda_}, η={config.eta})")

    # Initialize memory matrix
    M = torch.zeros(batch_size, n_p_total, n_p_total)
    print(f"✓ Memory matrix: {M.shape}")

    # Simulate learning sequence
    print(f"\nTraining:")
    for t in range(n_steps):
        # Generate random grounded locations
        p_inferred = torch.randn(batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(batch_size, n_p_total).softmax(dim=1)

        # Hebbian update
        M = storage.update(p_inferred, p_generated, M)

        # Check memory strength
        strength = M.abs().mean().item()
        print(f"  Step {t+1}/{n_steps}: memory strength = {strength:.4f}")

    print(f"\n✓ Training complete")

    # Memory statistics
    print(f"\nFinal memory matrix:")
    print(f"  Shape: {M.shape}")
    print(f"  Mean: {M.mean():.6f}")
    print(f"  Std: {M.std():.6f}")
    print(f"  Range: [{M.min():.3f}, {M.max():.3f}]")

    print("\n" + "=" * 80)
