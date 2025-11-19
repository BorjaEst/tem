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
from torch_tem.config import ArchitectureConfig, InferenceConfig
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for memory storage example.

    This config wraps ArchitectureConfig and InferenceConfig for the memory storage demonstration.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="memory_storage")

    # Environment configuration (used only for parameter sizing)
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for spatial environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    @computed_field(description="Sensory observation dimension")
    @property
    def n_x(self) -> int:
        return self.grid_size * self.grid_size

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Training configuration
    n_training_steps: int = Field(default=50, ge=10, le=500, description="Number of Hebbian updates")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory updates")

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


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the memory storage experiment with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    inference_config = InferenceConfig(eta=config.eta, kappa=config.kappa)
    model_config = ArchitectureConfig(n_x=config.n_x, n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial)

    # Compute connectivity matrices from model config
    p_update_mask = utils.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f, 0, model_config.f_initial_extended)

    print("=" * 80)
    print("Memory Storage Experiment")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print(f"  Hebbian learning: η={config.eta}, λ={config.lambda_}")
    print(f"  Dual memory: {not model_config.common_memory}")
    print(f"  Training steps: {config.n_training_steps}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory Storage
    # =========================================================================
    print("Phase 1: Initializing memory storage...")
    # MemoryStorage manages Hebbian memory matrices (M_gen, M_inf)
    # It implements: M = λ*M + η*outer(p_inf, p_gen) * mask
    storage = MemoryStorage(model_config, inference_config, p_update_mask)

    n_p_total = sum(model_config.n_p)
    print(f"  ✓ MemoryStorage: {n_p_total}×{n_p_total} Hebbian matrix")
    print(f"  ✓ M_gen: {storage.M_gen.shape}")
    if not model_config.common_memory:
        print(f"  ✓ M_inf: {storage.M_inf.shape}")
    print()

    # =========================================================================
    # PHASE 2: Train Memory Through Hebbian Learning
    # =========================================================================
    print("Phase 2: Training memory through Hebbian learning...")

    # Simulate spatial navigation by generating random place cell patterns
    # In full TEM: p = g ⊗ x (grid cells ⊗ compressed sensory input)

    # Track learning progress over training
    memory_strengths = []  # Frobenius norm of M_gen (overall connection strength)
    cosine_sims = []  # Similarity between M_gen and M_inf (dual memory divergence)
    update_magnitudes = []  # Track size of each Hebbian update

    for step in range(config.n_training_steps):
        # Generate random grounded location patterns (batch_size samples)
        # In real TEM: p_inferred comes from sensory→location inference
        # In real TEM: p_generated comes from abstract→location prediction
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
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
        if not model_config.common_memory:
            m_gen_flat = storage.M_gen.flatten()
            m_inf_flat = storage.M_inf.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(m_gen_flat, m_inf_flat, dim=0).item()
            cosine_sims.append(cosine_sim)

        # Progress reporting
        if (step + 1) % 10 == 0 or step == 0:
            msg = f"  Step {step+1}/{config.n_training_steps}: M_gen strength={m_gen_strength:.4f}, update={update_magnitude:.6f}"
            if not model_config.common_memory:
                msg += f", similarity={cosine_sim:.4f}"
            print(msg)

    print(f"  ✓ Processed {config.n_training_steps} training steps")
    print()
    print(f"Training Summary:")
    print(f"  Final M_gen strength: {memory_strengths[-1]:.4f}")
    print(f"  Final update magnitude: {update_magnitudes[-1]:.6f}")
    if not model_config.common_memory:
        print(f"  Final M_gen/M_inf similarity: {cosine_sims[-1]:.4f}")
    print()

    # =========================================================================
    # PHASE 3: Analyze Memory Structure
    # =========================================================================
    print("Phase 3: Analyzing learned memory structure...")

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
    for freq_idx, n_p in enumerate(model_config.n_p):
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
    print("Phase 4: Generating visualizations...")

    # Plot 1: Memory matrix structure
    fig1 = figures.plot_memory_matrices(
        storage.M_gen,
        storage.get_memory(for_inference=True),
        n_p_per_freq=model_config.n_p,
        n_training_steps=config.n_training_steps,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_memory_matrices.png")

    # Plot 2: Learning dynamics over training
    fig2 = figures.plot_learning_curve(
        memory_strengths,
        cosine_sims if not model_config.common_memory else None,
    )
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_learning_curve.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_learning_curve.png")

    print()
    print("=" * 80)
    print("Memory Storage Summary:")
    print("=" * 80)
    print(f"Architecture: {model_config.n_f} frequencies, {sum(model_config.n_p)} total place cells")
    print(f"Training: {config.n_training_steps} steps with η={config.eta}, λ={config.lambda_}")
    print(f"Final memory strength: {memory_strengths[-1]:.4f}")
    if not model_config.common_memory:
        print(f"Dual memory similarity: {cosine_sims[-1]:.4f}")
    print(f"Sparsity: {sparsity_gen*100:.1f}%")
    print(f"Symmetry error: {symmetry_gen.item():.6f}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
