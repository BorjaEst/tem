#!/usr/bin/env python3
"""Hebbian memory storage example with learning dynamics visualization.

This example demonstrates the torch_tem.memory.storage module capabilities:
- MemoryStorage initialization and Hebbian plasticity
- Memory matrix evolution through associative learning
- Dual memory architecture (inference vs. generative)
- Learning dynamics and convergence analysis
- Memory structure visualization

The Hebbian learning rule implements biologically-inspired associative memory:
    M[t+1] = λ * M[t] + η * outer(p_inf, p_gen) * mask

Where connections strengthen when pre- and post-synaptic neurons fire together,
with gradual decay (forgetting) over time. This enables spatial relationship
learning and predictive coding in the TEM architecture.

Usage:
    python examples/memory_storage.py --n-training-steps 100 --eta 0.4
    python examples/memory_storage.py --lambda 0.98 --use-dual-memory
    python examples/memory_storage.py --help
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

from torch_tem import data, figures, utils
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for memory storage example.

    This config implements the MemoryStorageParams protocol, allowing direct
    instantiation of MemoryStorage component.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="memory_storage")

    # Environment configuration (used only for parameter sizing)
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for spatial environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Memory architecture
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of hierarchical frequency modules")
    n_g_per_module: int = Field(default=10, ge=5, le=20, description="Grid cells per frequency module")
    n_x_c: int = Field(default=5, ge=2, le=20, description="Compressed sensory dimensions")

    # Hebbian learning parameters
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Remembering rate (Hebbian learning strength)")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Forgetting rate (memory decay)")

    # Training configuration
    n_training_steps: int = Field(default=50, ge=10, le=500, description="Number of Hebbian updates")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory updates")

    # Memory configuration
    use_dual_memory: bool = Field(default=True, description="Use separate inference/generative memories")
    common_memory: bool = Field(default=False, description="Share memory between inference and generation")

    # Output
    output_dir: Path = Field(default=Path("outputs/memory_storage"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @computed_field(description="Number of unique observations")
    @property
    def n_x(self) -> int:
        n_locations = self.grid_size * self.grid_size
        return n_locations if self.observation_mode == "unique" else max(4, n_locations // 4)

    # ==============================================================================
    # MemoryStorageParams Protocol Implementation
    # ==============================================================================

    @computed_field(description="Neurons for hippocampal grounded location p per frequency")
    @property
    def n_p_calculated(self) -> List[int]:
        return [self.n_g_per_module * self.n_x_c for _ in range(self.n_frequencies)]

    @computed_field(description="Hierarchical mask for memory updates")
    @property
    def p_update_mask_calculated(self) -> Tensor:
        return utils.create_p_update_mask(
            n_p=self.n_p_calculated,
            n_f=self.n_frequencies,
            n_f_g=self.n_frequencies,  # All modules are grid-based
            n_f_ovc=0,  # No object vector cell modules
            f_initial=[float(i) for i in range(self.n_frequencies)],  # 0, 1, 2, ... (low to high)
        )

    @computed_field(description="Whether to use inference-based grounded locations")
    @property
    def use_p_inf(self) -> bool:
        return self.use_dual_memory


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the memory storage experiment with visualizations."""
    # Parse CLI arguments and create configuration
    # This implements MemoryStorageParams protocol for direct instantiation
    config = ExampleConfig()

    print(f"Memory Storage Experiment Configuration:")
    print(f"  Architecture: {config.n_frequencies} frequencies × {config.n_g_per_module} grid cells × {config.n_x_c} sensory dims")
    print(f"  Total place cells: {sum(config.n_p_calculated)}")
    print(f"  Hebbian learning: η={config.eta}, λ={config.lambda_}")
    print(f"  Dual memory: {config.use_dual_memory}")
    print(f"  Training steps: {config.n_training_steps}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory Storage
    # =========================================================================
    # MemoryStorage manages Hebbian memory matrices (M_gen, M_inf)
    # It implements: M = λ*M + η*outer(p_inf, p_gen) * mask
    storage = MemoryStorage(config)

    n_p_total = sum(config.n_p_calculated)
    print(f"Initialized memory matrices: {n_p_total}×{n_p_total}")
    print(f"  M_gen: {storage.M_gen.shape}")
    if storage.use_dual_memory:
        print(f"  M_inf: {storage.M_inf.shape}")
    print()

    # =========================================================================
    # PHASE 2: Train Memory Through Hebbian Learning
    # =========================================================================
    print("Training memory through Hebbian learning...")
    print("  Generating random spatial patterns and updating associations")
    print()

    # Simulate spatial navigation by generating random place cell patterns
    # In full TEM: p = g ⊗ x (grid cells ⊗ compressed sensory input)

    # Track learning progress over training
    memory_strengths = []  # Frobenius norm of M_gen (overall connection strength)
    cosine_sims = []  # Similarity between M_gen and M_inf (dual memory divergence)
    update_magnitudes = []  # Track size of each Hebbian update

    for step in range(config.n_training_steps):
        # Generate random grounded location patterns (batch_size samples)
        # softmax ensures valid probability distributions (sum to 1, non-negative)
        # In real TEM: p_inferred comes from sensory→location inference
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # In real TEM: p_generated comes from abstract→location prediction
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # Store pre-update state to measure change magnitude
        M_gen_before = storage.M_gen.clone()

        # Apply Hebbian learning rule: strengthen connections between co-active patterns
        # M_new = λ*M_old + η*outer(p_inf, p_gen) * mask
        # - λ (lambda): forgetting rate (decay old memories)
        # - η (eta): learning rate (strength of new associations)
        storage.update(p_inferred, p_generated, eta=config.eta, lamb=config.lambda_)

        # Monitor memory strength (how much information is stored)
        # Frobenius norm = sqrt(sum of squared weights)
        m_gen_strength = torch.norm(storage.M_gen).item()
        memory_strengths.append(m_gen_strength)

        # Track update magnitude (how much did the memory change?)
        update_magnitude = torch.norm(storage.M_gen - M_gen_before).item()
        update_magnitudes.append(update_magnitude)

        # Monitor divergence between dual memories (should stay similar if properly tuned)
        # Cosine similarity = 1.0 means identical, 0.0 means orthogonal
        if storage.use_dual_memory:
            m_gen_flat = storage.M_gen.flatten()
            m_inf_flat = storage.M_inf.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(m_gen_flat, m_inf_flat, dim=0).item()
            cosine_sims.append(cosine_sim)

        # Progress reporting
        if (step + 1) % 10 == 0 or step == 0:
            msg = f"  Step {step+1}/{config.n_training_steps}: M_gen strength={m_gen_strength:.4f}, update={update_magnitude:.6f}"
            if storage.use_dual_memory:
                msg += f", similarity={cosine_sim:.4f}"
            print(msg)

    print()
    print("Training complete!")
    print(f"  Final M_gen strength: {memory_strengths[-1]:.4f}")
    print(f"  Final update magnitude: {update_magnitudes[-1]:.6f}")
    if storage.use_dual_memory:
        print(f"  Final M_gen/M_inf similarity: {cosine_sims[-1]:.4f}")
    print()

    # =========================================================================
    # PHASE 3: Analyze Memory Structure
    # =========================================================================
    print("Analyzing learned memory structure...")

    # Compute memory statistics
    M_gen = storage.M_gen
    M_inf = storage.get_memory(for_inference=True)

    # Sparsity (fraction of near-zero weights)
    sparsity_threshold = 0.01
    sparsity_gen = (M_gen.abs() < sparsity_threshold).float().mean().item()
    print(f"  M_gen sparsity (<{sparsity_threshold}): {sparsity_gen*100:.1f}%")

    # Symmetry (should be high due to outer product)
    symmetry_gen = torch.norm(M_gen - M_gen.T) / torch.norm(M_gen)
    print(f"  M_gen symmetry error: {symmetry_gen.item():.6f}")

    # Eigenvalue spectrum (indicates memory capacity)
    eigenvalues_gen = torch.linalg.eigvalsh(M_gen.cpu())
    top_eigenvalues = eigenvalues_gen[-5:].flip(0)
    print(f"  M_gen top 5 eigenvalues: {top_eigenvalues.tolist()}")

    # Hierarchical block structure (within vs. between frequency connections)
    block_stats = []
    start_idx = 0
    for freq_idx, n_p in enumerate(config.n_p_calculated):
        end_idx = start_idx + n_p
        # Within-block strength (same frequency)
        within_block = M_gen[start_idx:end_idx, start_idx:end_idx]
        within_strength = torch.norm(within_block).item()
        # Between-block strength (cross-frequency)
        between_block = M_gen[start_idx:end_idx, :]
        between_block[:, start_idx:end_idx] = 0  # Exclude diagonal
        between_strength = torch.norm(between_block).item()
        block_stats.append((freq_idx, within_strength, between_strength))
        print(f"  Frequency {freq_idx}: within={within_strength:.4f}, between={between_strength:.4f}")
        start_idx = end_idx

    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Generating visualizations...")

    # Plot 1: Memory matrix structure
    # Shows the learned association weights M_gen and M_inf
    # Block structure reveals hierarchical frequency organization
    fig1 = figures.plot_memory_matrices(
        storage.M_gen,  # Generative memory (abstract→location associations)
        storage.get_memory(for_inference=True),  # Inference memory (sensory→location)
        n_p_per_freq=config.n_p_calculated,  # Dimensions for block visualization
        n_training_steps=config.n_training_steps,  # For title annotation
    )
    if config.save_plots:
        save_path = config.output_dir / "01_memory_matrices.png"
        fig1.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    # Plot 2: Learning dynamics over training
    # Track how memory strength grows and dual memories evolve
    fig2 = figures.plot_learning_curve(
        memory_strengths,  # Frobenius norm trajectory
        cosine_sims if storage.use_dual_memory else None,  # Dual memory similarity
    )
    if config.save_plots:
        save_path = config.output_dir / "02_learning_curve.png"
        fig2.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Display plots interactively or just save them
    if config.show_plots:
        plt.show()  # Blocks until user closes windows
    else:
        plt.close("all")  # Clean up memory
