#!/usr/bin/env python3
"""Attractor dynamics retrieval example with hierarchical convergence visualization.

This example demonstrates the torch_tem.memory.attractor module capabilities:
- AttractorDynamics retrieval with hierarchical early-stopping
- Content-addressable memory recall from noisy queries
- Convergence analysis across different query patterns
- Hierarchical mask effects on retrieval quality
- Robustness testing across noise levels
- Dual memory (inference vs. generative) comparison

The attractor dynamics implement iterative memory retrieval:
    p[t+1] = κ * p[t] + (M^T @ p[t]) * mask[t]

Where noisy or partial query patterns are iteratively refined toward stored
spatial patterns through hierarchical coarse-to-fine convergence.

Usage:
    python examples/memory_attractor.py --n-test-queries 10 --noise-level 0.4
    python examples/memory_attractor.py --kappa 0.9 --n-frequencies 4
    python examples/memory_attractor.py --help
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
from torch_tem.config import ModelConfig
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for attractor dynamics example.

    This config handles example-specific parameters, while ModelConfig
    and InferenceConfig handle the model architecture and inference settings.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="memory_attractor")

    # Architecture configuration
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")
    n_memory_init_steps: int = Field(default=20, ge=5, le=100, description="Steps for memory initialization")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory initialization")

    # Retrieval testing configuration
    n_test_queries: int = Field(default=5, ge=1, le=20, description="Number of test retrieval queries")
    noise_level: float = Field(default=1.0, ge=0.0, le=5.0, description="Noise level in logit space (std of Gaussian noise added before softmax)")

    # Output
    output_dir: Path = Field(default=Path("outputs/memory_attractor"), description="Directory for saving plots")
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
    """Run the attractor dynamics experiment with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    model_config = ModelConfig(n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial, eta=config.eta, kappa=config.kappa)

    # Compute connectivity matrices from model config
    p_update_mask = utils.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f_g, model_config.n_f_ovc, model_config.f_extended)
    mask_inf = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf)
    mask_gen = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_gen)

    print("=" * 80)
    print("Attractor Dynamics Memory Retrieval")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print(f"  Attractor: κ={config.kappa}, iterations={model_config.i_attractor}")
    print(f"  Test queries: {config.n_test_queries} with noise_level={config.noise_level}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory and Attractor
    # =========================================================================
    print("Phase 1: Initializing memory and attractor components...")
    print("  Note: Using random patterns for demonstration. In real TEM:")
    print("    - p_inferred comes from sensory → location inference")
    print("    - p_generated comes from abstract → location prediction")
    print("    - M_inf learns outer(p_inf, p_inf) for sensory-driven completion")
    print("    - M_gen learns outer(p_inf, p_gen) for predictive associations")

    # Initialize memory storage with proper config objects
    storage = MemoryStorage(model_config, p_update_mask)
    n_p_total = sum(model_config.n_p)
    print(f"  ✓ MemoryStorage: {n_p_total}×{n_p_total} Hebbian matrix")

    for step in range(config.n_memory_init_steps):
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        storage.update(p_inferred, p_generated, eta=config.eta, lamb=config.lambda_)

    m_gen_strength = torch.norm(storage.M_gen).item()
    m_inf_strength = torch.norm(storage.M_inf).item() if storage.use_dual_memory else 0
    m_diff = torch.norm(storage.M_gen - storage.M_inf).item() if storage.use_dual_memory else 0
    print(f"  Memory initialized after {config.n_memory_init_steps} steps:")
    print(f"    M_gen strength={m_gen_strength:.4f}")
    if storage.use_dual_memory:
        print(f"    M_inf strength={m_inf_strength:.4f}")
        print(f"    Difference={m_diff:.4f} ({m_diff/m_gen_strength:.1%} relative)")

    # Initialize attractor dynamics with proper config objects
    attractor = AttractorDynamics(model_config, mask_inf, mask_gen)
    print(f"  ✓ AttractorDynamics: {model_config.i_attractor} iterations with hierarchical masking")
    print()

    print(f"Hierarchical masking schedule (inference mode):")
    for it, mask in enumerate(attractor.p_retrieve_mask_inf):
        n_active = mask.sum().item()
        print(f"  Iteration {it+1}: {n_active}/{n_p_total} neurons active ({n_active/n_p_total*100:.1f}%)")
    print()

    # =========================================================================
    # PHASE 2: Test Attractor Retrieval Quality
    # =========================================================================
    print("Phase 2: Testing attractor retrieval quality...")
    print(f"  Test queries: {config.n_test_queries}")
    print(f"  Noise level: {config.noise_level}")
    print()

    # Create clean target patterns from the learned distribution
    # These represent "ground truth" locations we want to retrieve
    # Generate in logit space first to allow meaningful noise addition
    target_logits = torch.randn(config.n_test_queries, n_p_total)
    test_targets = target_logits.softmax(dim=1)

    # Add noise in logit space (before softmax) to preserve probability structure
    # This creates realistic corruption while maintaining valid distributions
    noise_logits = torch.randn_like(target_logits) * config.noise_level
    query_logits = target_logits + noise_logits
    test_queries = query_logits.softmax(dim=1)

    # Compute signal-to-noise ratio for reporting
    snr_db = utils.compute_snr_db(test_targets, test_queries)
    print(f"  Signal-to-noise ratio: {snr_db:.2f} dB")
    print()

    # Run attractor dynamics: iteratively refine queries toward stored patterns
    # p[t+1] = κ*p[t] + M^T@p[t] * mask[t]
    # - κ (kappa): decay term for stability
    # - M^T@p[t]: memory-driven update (pulls toward associated patterns)
    # - mask[t]: hierarchical early-stopping (coarse→fine refinement)
    M_inf = storage.get_memory(for_inference=True)

    # Convert batched tensors to per-frequency list format
    test_queries_list = utils.split_to_frequencies(test_queries, model_config.n_p)
    # Attractor expects and returns per-frequency lists (MultiScaleCode)
    test_retrievals_list = attractor(test_queries_list, M_inf, for_inference=True)

    # Compute retrieval quality metrics
    # MSE measures how close retrieved patterns are to ground truth
    print("Retrieval quality metrics:")
    improvements = []

    # Concatenate for metrics computation
    test_queries_cat = torch.cat(test_queries_list, dim=1)
    test_retrievals_cat = torch.cat(test_retrievals_list, dim=1)

    for i in range(config.n_test_queries):
        query_error = torch.nn.functional.mse_loss(test_queries_cat[i], test_targets[i]).item()
        retrieval_error = torch.nn.functional.mse_loss(test_retrievals_cat[i], test_targets[i]).item()
        improvement = ((query_error - retrieval_error) / query_error) * 100  # Percentage improvement
        improvements.append(improvement)
        print(f"  Query {i+1}: query_error={query_error:.6f}, retrieval_error={retrieval_error:.6f}, improvement={improvement:.1f}%")

    avg_improvement = np.mean(improvements)
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print()

    # =========================================================================
    # PHASE 3: Robustness Analysis Across Noise Levels
    # =========================================================================
    print("Phase 3: Testing robustness across noise levels...")
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]  # Logit-space noise levels
    errors_by_mode = {"Inference": [], "Generative": []}  # Compare dual memories

    for noise_level in noise_levels:
        # Add noise in logit space for realistic corruption
        # This preserves the probability distribution structure
        noise_logits = torch.randn_like(target_logits) * noise_level
        noisy_query_logits = target_logits + noise_logits
        noisy_queries = noisy_query_logits.softmax(dim=1)

        # Convert to per-frequency list format for attractor
        noisy_queries_list = utils.split_to_frequencies(noisy_queries, model_config.n_p)

        # Test inference memory (used for sensory→location inference)
        retrieved_inf_list = attractor(noisy_queries_list, storage.get_memory(for_inference=True), for_inference=True)
        error_inf = torch.nn.functional.mse_loss(torch.cat(retrieved_inf_list, dim=1), test_targets).item()
        errors_by_mode["Inference"].append(error_inf)

        # Test generative memory (used for abstract→location prediction)
        retrieved_gen_list = attractor(noisy_queries_list, storage.get_memory(for_inference=False), for_inference=False)
        error_gen = torch.nn.functional.mse_loss(torch.cat(retrieved_gen_list, dim=1), test_targets).item()
        errors_by_mode["Generative"].append(error_gen)

        # Compute SNR for this noise level
        snr_db = utils.compute_snr_db(test_targets, noisy_queries)

        print(f"  Noise {noise_level:.1f} (SNR={snr_db:+.1f}dB): Inference MSE={error_inf:.6f}, Generative MSE={error_gen:.6f}")

    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Plot 1: Hierarchical retrieval masks
    # Visualize progressive unmasking schedule (coarse→fine)
    fig1 = figures.plot_hierarchical_masks(
        attractor.p_retrieve_mask_inf,  # Binary masks for each iteration
        n_p_per_freq=model_config.n_p,  # Frequency boundaries
    )
    if config.save_plots:
        save_path = config.output_dir / "01_hierarchical_masks.png"
        fig1.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    # Plot 2: Attractor convergence trajectories
    # Show how noisy queries are iteratively refined toward targets
    # Split targets to per-frequency format for visualization
    test_targets_list = utils.split_to_frequencies(test_targets, model_config.n_p)

    fig2 = figures.plot_attractor_convergence(
        test_queries_list,  # Batched MultiScaleCode: List of [n_queries, n_p[f]]
        test_retrievals_list,  # Batched MultiScaleCode
        test_targets_list,  # Batched MultiScaleCode
        n_p_per_freq=model_config.n_p,  # Enable frequency module visualization
    )
    if config.save_plots:
        save_path = config.output_dir / "02_attractor_convergence.png"
        fig2.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    # Plot 3: Robustness to noise
    # Compare inference vs generative memory across noise levels
    fig3 = figures.plot_retrieval_quality(
        errors_by_mode,  # MSE for each mode at each noise level
        noise_levels,  # X-axis values
    )
    if config.save_plots:
        save_path = config.output_dir / "03_retrieval_quality.png"
        fig3.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    print()
    print("=" * 80)
    print("Experiment Summary:")
    print("=" * 80)
    print(f"Memory architecture: {n_p_total} place cells across {model_config.n_f} frequencies")
    print(f"Attractor dynamics: {model_config.i_attractor} iterations with κ={config.kappa}")
    print(f"Memory strength: M_gen={m_gen_strength:.4f}, M_inf={m_inf_strength:.4f}")
    print(f"Test results: {config.n_test_queries} queries, avg improvement={avg_improvement:.1f}%")
    print(f"Robustness: Tested across {len(noise_levels)} noise levels")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
