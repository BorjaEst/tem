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


class ModelParams(Protocol):
    """Architecture parameters needed by MemoryStorage."""

    n_p: List[int]
    common_memory: bool
    use_p_inf: bool


class MemoryStorage:
    """Manages Hebbian memory matrices with hierarchical updates.

    In TEM, memory matrices store learned associations between grounded locations
    (hippocampal place cells) via Hebbian plasticity. These associations enable:
    1. Spatial inference: Recall locations from sensory observations
    2. Predictive coding: Generate expected locations from abstract representations
    3. Path integration: Maintain spatial relationships during navigation

    The memory system can maintain separate matrices for inference (M_inf) and
    generation (M_gen) to allow different learning dynamics, or use a common
    memory for both (controlled by use_p_inf and common_memory parameters).

    Hebbian Update Rule:
        M[t+1] = λ * M[t] + η * outer(p_inf, p_gen) * mask

    Where:
        - λ (lambda): Forgetting rate (0 < λ < 1), controls memory decay
        - η (eta): Remembering rate (0 < η < 1), controls learning strength
        - outer(p_inf, p_gen): Outer product of inferred and generated locations
        - mask: Hierarchical mask limiting connections (low→high frequency only)

    The hierarchical mask ensures information flows from coarse (low frequency)
    to fine (high frequency) spatial scales, mirroring the organization of grid
    cells in the entorhinal cortex.

    Attributes:
        n_p: List of place cell counts per frequency module
        p_update_mask: Hierarchical mask for Hebbian updates [sum(n_p), sum(n_p)]
        use_dual_memory: Whether to maintain separate inference/generation matrices
        M_gen: Generative memory matrix [sum(n_p), sum(n_p)]
        M_inf: Inference memory matrix (optional) [sum(n_p), sum(n_p)]
    """

    def __init__(self, model_params: ModelParams, p_update_mask: Tensor, batch_size: int = 1):
        """Initialize memory storage with zero-initialized matrices.

        Args:
            model_params: Architecture configuration (n_p, common_memory, use_p_inf)
            p_update_mask: Hierarchical mask for Hebbian updates
            batch_size: Number of parallel environments (default=1 for single env)
        """
        self.n_p = model_params.n_p
        self.use_dual_memory = model_params.use_p_inf and not model_params.common_memory
        self.p_update_mask = p_update_mask
        self.batch_size = batch_size

        # Initialize memory matrices as zero matrices
        # These will be populated during training via Hebbian updates
        # Shape: [batch_size, sum(n_p), sum(n_p)] for batched training
        # or [sum(n_p), sum(n_p)] for single environment (batch_size=1, backward compat)
        n_p_total = sum(self.n_p)
        if batch_size > 1:
            self.M_gen = torch.zeros(batch_size, n_p_total, n_p_total)
            self.M_inf = torch.zeros(batch_size, n_p_total, n_p_total) if self.use_dual_memory else None
        else:
            # Backward compatibility: single environment without batch dimension
            self.M_gen = torch.zeros(n_p_total, n_p_total)
            self.M_inf = torch.zeros(n_p_total, n_p_total) if self.use_dual_memory else None

    def update(self, p_inferred: Vector, p_generated: Vector, eta: float, lamb: float) -> None:
        """Update memory matrices using Hebbian plasticity rule.

        Implements the core learning mechanism of TEM's associative memory system.
        The outer product of inferred and generated grounded locations strengthens
        connections between co-active place cells, while the decay term (λ) gradually
        forgets older associations.

        This biologically-inspired learning rule enables the model to:
        1. Learn spatial relationships through experience (remembering)
        2. Generalize across similar contexts (Hebbian association)
        3. Adapt to changing environments (controlled forgetting)

        The hierarchical mask restricts learning to valid connections, ensuring
        information flows from low-frequency (coarse) to high-frequency (fine)
        spatial representations, preventing unstable feedback loops.

        Args:
            p_inferred: Inferred grounded location from sensory input [B, sum(n_p)]
                       Represents "where the agent thinks it is" based on observations
            p_generated: Generated grounded location from predictions [B, sum(n_p)]
                        Represents "where the agent expects to be" from transitions
            eta: Remembering rate controlling learning strength (0 < η ≤ 1)
                Higher values = faster learning but potentially unstable
            lamb: Forgetting rate controlling memory decay (0 < λ < 1)
                  Higher values = slower forgetting, longer memory retention

        Note:
            Supports both batched [B, sum(n_p)] and unbatched [sum(n_p)] inputs.
            For batched inputs with batch_size=1, uses per-batch updates (legacy compat).
            For unbatched inputs, averages across batch before updating (new behavior).

            Mathematical formulation:
                M_gen_new = λ * M_gen_old + η * outer(p_inf, p_gen) * mask
                M_inf_new = λ * M_inf_old + η * outer(p_inf, p_inf) * mask

            Following original TEM implementation:
            - M_gen learns associations between inferred and generated locations
            - M_inf learns associations within inferred locations (sensory-driven)
        """
        # Handle batch dimension
        if p_inferred.dim() == 1:
            p_inferred = p_inferred.unsqueeze(0)
        if p_generated.dim() == 1:
            p_generated = p_generated.unsqueeze(0)

        # Move hierarchical mask to same device as data (handles CPU/GPU transfers)
        mask = self.p_update_mask.to(p_inferred.device)

        if self.batch_size > 1:
            # Batched memory update (legacy compatibility mode)
            # Each environment has its own memory matrix: [B, sum(n_p), sum(n_p)]
            # Compute outer product: [B, sum(n_p), 1] @ [B, 1, sum(n_p)] → [B, sum(n_p), sum(n_p)]
            batch_outer_gen = torch.bmm(p_inferred.unsqueeze(2), p_generated.unsqueeze(1))

            # Hebbian update with decay and learning, applied per batch element
            self.M_gen = lamb * self.M_gen.to(batch_outer_gen.device) + eta * (batch_outer_gen * mask)

            # Update inference memory (if using dual-memory architecture)
            if self.use_dual_memory:
                batch_outer_inf = torch.bmm(p_inferred.unsqueeze(2), p_inferred.unsqueeze(1))
                self.M_inf = lamb * self.M_inf.to(batch_outer_inf.device) + eta * (batch_outer_inf * mask)
        else:
            # Single global memory update (new modular approach)
            # Average across batch to get typical association pattern
            # Compute outer product then average: mean([B, sum(n_p), sum(n_p)]) → [sum(n_p), sum(n_p)]
            batch_outer_gen = torch.mean(torch.bmm(p_inferred.unsqueeze(2), p_generated.unsqueeze(1)), dim=0)

            # Hebbian update for generative memory with decay and learning
            self.M_gen = lamb * self.M_gen.to(batch_outer_gen.device) + eta * (batch_outer_gen * mask)

            # Update inference memory (if using dual-memory architecture)
            if self.use_dual_memory:
                batch_outer_inf = torch.mean(torch.bmm(p_inferred.unsqueeze(2), p_inferred.unsqueeze(1)), dim=0)
                self.M_inf = lamb * self.M_inf.to(batch_outer_inf.device) + eta * (batch_outer_inf * mask)

    def get_memory(self, for_inference: bool = False) -> Matrix:
        """Get appropriate memory matrix for retrieval.

        Returns the inference memory when available and requested (dual-memory mode),
        otherwise returns the generative memory. This enables different retrieval
        strategies for inference vs. generation tasks.

        Args:
            for_inference: If True and dual memory enabled, return M_inf; else M_gen

        Returns:
            Memory matrix [sum(n_p), sum(n_p)] for attractor dynamics retrieval
        """
        if for_inference and self.M_inf is not None:
            return self.M_inf
        return self.M_gen

    def get_all_memories(self) -> List[Matrix]:
        """Get both memory matrices for state storage/checkpointing.

        Used to save the complete memory state during training for later restoration
        or analysis. Essential for model checkpointing and reproducibility.

        Returns:
            List of memory matrices: [M_gen, M_inf] or [M_gen] if single memory
        """
        if self.use_dual_memory:
            return [self.M_gen, self.M_inf]
        return [self.M_gen]

    def set_memories(self, memories: List[Matrix]) -> None:
        """Set memory matrices from saved state.

        Restores memory state from checkpoints or pre-trained models. Critical for
        continuing training, transfer learning, or analysis of trained models.

        Args:
            memories: List of memory matrices [M_gen] or [M_gen, M_inf]
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
    from torch_tem.config.parameters import Parameters

    # Create configuration with 3 frequency modules
    params = Parameters(
        n_g_subsampled=[10, 8, 6],  # 3 frequency modules (coarse to fine)
        n_x_c=5,  # 5 compressed sensory dimensions
        lambda_=0.95,  # 95% memory retention (slow forgetting)
        eta=0.3,  # 30% learning rate (moderate remembering)
        use_p_inf=True,  # Enable dual memory (inference + generation)
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
        storage.update(p_inferred, p_generated, eta=params.eta, lamb=params.lambda_)

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
