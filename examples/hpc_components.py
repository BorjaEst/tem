#!/usr/bin/env python3
"""HPC components example demonstrating Hebbian memory and attractor dynamics.

This example demonstrates the Hippocampus (HPC) memory components:
- MemoryStorage: Hebbian plasticity for associative learning
- AttractorDynamics: Iterative pattern completion with hierarchical refinement

The HPC pipeline:
  Storage: (p_inf, p_gen) → Hebbian update → M (memory matrix)
  Retrieval: p_query → Attractor dynamics → p_retrieved

Note: This example uses synthetic place cell patterns. For spatial navigation,
see tem_inference.py and tem_generative.py.

Architecture:
-------------
    HPC MemoryStorage (hpc.storage.MemoryStorage):
        - Input: Grounded locations (p_inf, p_gen) [B, sum(n_p)]
        - Output: Updated memory matrix M [B, sum(n_p), sum(n_p)]
        - Method: Hebbian plasticity with decay
        - Formula: M = λ*M + η*outer(p_inf + p_gen, p_inf - p_gen) * mask

    HPC AttractorDynamics (hpc.attractor.AttractorDynamics):
        - Input: Noisy query p_query [List[n_f] of [B, n_p[f]]]
        - Output: Refined retrieval p_retrieved [List[n_f] of [B, n_p[f]]]
        - Method: Iterative refinement with hierarchical masking
        - Formula: p[t+1] = mask[t] * activation(κ*p[t] + M@p[t]) + (1-mask[t])*p[t]

Usage Examples:
---------------
    # Default: 50 training steps, 5 test queries
    python examples/hpc_components.py

    # More training with higher learning rate
    python examples/hpc_components.py --n_training_steps 100 --eta 0.4

    # Test retrieval robustness
    python examples/hpc_components.py --n_test_queries 10 --noise_level 2.0

    # Custom architecture
    python examples/hpc_components.py --f_initial "[0.95, 0.7, 0.4]"

    # Show plots interactively
    python examples/hpc_components.py --show_plots true --save_plots false

    # Full help
    python examples/hpc_components.py --help

Outputs:
--------
When save_plots=true, generates 6 visualizations in outputs/hpc_components/:
    1. 01_memory_matrices.png - Learned memory structure (M_gen and M_inf)
    2. 02_learning_dynamics.png - Training convergence over time
    3. 03_hierarchical_masks.png - Attractor retrieval mask schedule
    4. 04_attractor_convergence.png - Query refinement trajectories
    5. 05_retrieval_quality.png - Performance across noise levels
    6. 06_memory_structure.png - Eigenvalue spectrum and block analysis
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import figures, utils
from torch_tem.config import ModelConfig
from torch_tem.hpc import attractor, storage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for HPC memory components example.

    Defines architecture parameters for demonstrating HPC components
    in isolation using entirely synthetic data (random place cell patterns).
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="hpc_components")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Training configuration
    n_training_steps: int = Field(default=50, ge=10, le=500, description="Number of Hebbian updates")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory updates")

    # Retrieval testing configuration
    n_test_queries: int = Field(default=5, ge=1, le=20, description="Number of test retrieval queries")
    noise_level: float = Field(default=1.0, ge=0.0, le=5.0, description="Noise level in logit space")

    # Output
    output_dir: Path = Field(default=Path("outputs/hpc_components"), description="Directory for saving plots")
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
    """Run the HPC memory components experiment with visualizations.

    This script demonstrates the memory storage and retrieval pathway:
      1. Initialize HPC components (MemoryStorage, AttractorDynamics)
      2. Train memory through Hebbian learning
      3. Test attractor retrieval with noisy queries
      4. Generate visualizations of the complete HPC system
    """
    config = ExampleConfig()

    # Create model config
    model_config = ModelConfig(
        n_x_c=config.n_x_c,
        n_g_subsampled=config.n_g_subsampled,
        f_initial=config.f_initial,
        eta=config.eta,
        kappa=config.kappa,
        batch_size=config.batch_size,
    )

    print("=" * 80)
    print("Hippocampal Memory Components")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print(f"  Memory: η={config.eta}, λ={config.lambda_}, κ={config.kappa}")
    print(f"  Training: {config.n_training_steps} steps, batch_size={config.batch_size}")
    print(f"  Testing: {config.n_test_queries} queries, noise_level={config.noise_level}")
    print()

    # =========================================================================
    # PHASE 1: Initialize HPC Components
    # =========================================================================
    print("Phase 1: Initializing HPC components...")

    # Compute connectivity masks
    # p_update_mask: Hierarchical mask for Hebbian learning
    # mask_inf: Conservative retrieval masks for stable inference
    # mask_gen: Flexible retrieval masks for generation
    p_update_mask = utils.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f_g, model_config.n_f_ovc, model_config.f_extended)
    mask_inf = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf)
    mask_gen = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_gen)

    # HPC MemoryStorage: Hebbian plasticity
    # Manages M_gen (generative) and M_inf (inference) memory matrices
    # Update rule: M = λ*M + η*outer(p_inf + p_gen, p_inf - p_gen) * mask
    mem_storage = storage.MemoryStorage(model_config, p_update_mask)

    # HPC AttractorDynamics: Iterative pattern completion
    # Refines noisy queries using hierarchical coarse-to-fine retrieval
    # Update rule: p[t+1] = mask[t] * activation(κ*p[t] + M@p[t]) + (1-mask[t])*p[t]
    mem_attractor = attractor.AttractorDynamics(model_config, mask_inf, mask_gen)

    n_p_total = sum(model_config.n_p)
    print(f"  ✓ MemoryStorage: {n_p_total}×{n_p_total} Hebbian matrix")
    print(f"    Dual memory: {not model_config.common_memory}")
    print(f"  ✓ AttractorDynamics: {model_config.i_attractor} iterations with hierarchical masking")
    print()

    # =========================================================================
    # PHASE 2: Train Memory Through Hebbian Learning
    # =========================================================================
    print("Phase 2: Training memory through Hebbian learning...")
    print("  Note: Using random patterns for demonstration. In real TEM:")
    print("    - p_inferred comes from sensory → location inference")
    print("    - p_generated comes from abstract → location prediction")
    print()

    # Track learning dynamics
    memory_strengths = []  # Frobenius norm (connection strength)
    cosine_sims = []  # M_gen vs M_inf similarity
    update_magnitudes = []  # Size of each update

    for step in range(config.n_training_steps):
        # Generate random grounded location patterns
        # In real TEM: p = g ⊗ x (grid cells ⊗ sensory)
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # Store pre-update state
        M_gen_before = mem_storage.M_gen.clone()

        # Apply Hebbian learning
        mem_storage.update(p_inferred, p_generated, eta=config.eta)

        # Monitor learning dynamics
        m_gen_strength = torch.norm(mem_storage.M_gen).item()
        memory_strengths.append(m_gen_strength)

        update_magnitude = torch.norm(mem_storage.M_gen - M_gen_before).item()
        update_magnitudes.append(update_magnitude)

        if not model_config.common_memory:
            # Measure dual memory divergence
            M_gen_flat = mem_storage.M_gen.flatten()
            M_inf_flat = mem_storage.M_inf.flatten()
            cos_sim = torch.nn.functional.cosine_similarity(M_gen_flat.unsqueeze(0), M_inf_flat.unsqueeze(0)).item()
            cosine_sims.append(cos_sim)

        # Progress reporting
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1}/{config.n_training_steps}: M_gen strength={m_gen_strength:.4f}")

    print(f"  ✓ Processed {config.n_training_steps} training steps")
    print()

    # =========================================================================
    # PHASE 3: Test Attractor Retrieval Quality
    # =========================================================================
    print("Phase 3: Testing attractor retrieval with noisy queries...")

    # Create clean target patterns (ground truth)
    # Use same batch size as training for compatibility with memory matrices
    test_targets = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

    # Add noise in logit space before softmax
    target_logits = torch.log(test_targets + 1e-8)  # Convert back to logits
    noise_logits = torch.randn_like(target_logits) * config.noise_level
    query_logits = target_logits + noise_logits
    test_queries = query_logits.softmax(dim=1)

    # Compute signal-to-noise ratio
    snr_db = utils.compute_snr_db(test_targets, test_queries)
    print(f"  Signal-to-noise ratio: {snr_db:.2f} dB")

    # Convert to per-frequency format for attractor
    test_queries_list = utils.split_to_frequencies(test_queries, model_config.n_p)
    test_targets_list = utils.split_to_frequencies(test_targets, model_config.n_p)

    # Run attractor dynamics
    M_inf = mem_storage.get_memory(for_inference=True)
    test_retrievals_list = mem_attractor(test_queries_list, M_inf, for_inference=True)

    # Compute retrieval quality
    test_queries_cat = torch.cat(test_queries_list, dim=1)
    test_retrievals_cat = torch.cat(test_retrievals_list, dim=1)

    improvements = []
    for i in range(config.batch_size):
        mse_query = torch.nn.functional.mse_loss(test_queries_cat[i], test_targets[i]).item()
        mse_retrieved = torch.nn.functional.mse_loss(test_retrievals_cat[i], test_targets[i]).item()
        improvement = ((mse_query - mse_retrieved) / mse_query) * 100
        improvements.append(improvement)

    avg_improvement = np.mean(improvements)
    print(f"  Average retrieval improvement: {avg_improvement:.1f}%")
    print()

    # =========================================================================
    # PHASE 4: Robustness Analysis Across Noise Levels
    # =========================================================================
    print("Phase 4: Testing robustness across noise levels...")

    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]
    errors_by_mode = {"Inference": [], "Generative": []}

    # Use same targets for consistency
    for noise_level in noise_levels:
        # Generate noisy queries from same targets
        noise = torch.randn_like(target_logits) * noise_level
        noisy_queries = (target_logits + noise).softmax(dim=1)
        noisy_queries_list = utils.split_to_frequencies(noisy_queries, model_config.n_p)

        # Test inference mode
        M_inf = mem_storage.get_memory(for_inference=True)
        retrieved_inf = mem_attractor(noisy_queries_list, M_inf, for_inference=True)
        retrieved_inf_cat = torch.cat(retrieved_inf, dim=1)
        mse_inf = torch.nn.functional.mse_loss(retrieved_inf_cat, test_targets).item()
        errors_by_mode["Inference"].append(mse_inf)

        # Test generative mode
        M_gen = mem_storage.get_memory(for_inference=False)
        retrieved_gen = mem_attractor(noisy_queries_list, M_gen, for_inference=False)
        retrieved_gen_cat = torch.cat(retrieved_gen, dim=1)
        mse_gen = torch.nn.functional.mse_loss(retrieved_gen_cat, test_targets).item()
        errors_by_mode["Generative"].append(mse_gen)

    print(f"  ✓ Tested across {len(noise_levels)} noise levels")
    print()

    # =========================================================================
    # PHASE 5: Analyze Memory Structure
    # =========================================================================
    print("Phase 5: Analyzing learned memory structure...")

    M_gen = mem_storage.M_gen
    M_inf = mem_storage.get_memory(for_inference=True)

    # Compute statistics
    sparsity_threshold = 0.01
    sparsity_gen = (M_gen.abs() < sparsity_threshold).float().mean().item()
    symmetry_gen = torch.norm(M_gen - M_gen.transpose(1, 2)) / torch.norm(M_gen)

    print(f"  M_gen sparsity (<{sparsity_threshold}): {sparsity_gen*100:.1f}%")
    print(f"  M_gen symmetry error: {symmetry_gen.item():.6f}")

    # Eigenvalue spectrum (memory capacity indicator)
    # Use first batch element for analysis
    eigenvalues_gen = torch.linalg.eigvalsh(M_gen[0].cpu())
    top_eigenvalues = eigenvalues_gen[-5:].flip(0)
    print(f"  M_gen top 5 eigenvalues: {[f'{x:.4f}' for x in top_eigenvalues.tolist()]}")
    print()

    # =========================================================================
    # PHASE 6: Generate Visualizations
    # =========================================================================
    print("Phase 6: Generating visualizations...")

    # Plot 1: Memory matrices structure
    fig1 = figures.plot_memory_matrices(
        mem_storage.M_gen[0],  # Use first batch element for visualization
        mem_storage.get_memory(for_inference=True)[0],
        n_p_per_freq=model_config.n_p,
        n_training_steps=config.n_training_steps,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_memory_matrices.png")

    # Plot 2: Learning dynamics
    fig2 = figures.plot_learning_curve(
        memory_strengths,
        cosine_sims if not model_config.common_memory else None,
    )
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_learning_dynamics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_learning_dynamics.png")

    # Plot 3: Hierarchical retrieval masks
    # Compare inference vs generative modes to show hierarchical early-stopping
    fig3 = figures.plot_hierarchical_masks(
        mem_attractor.p_retrieve_mask_inf,
        mem_attractor.p_retrieve_mask_gen,
        n_p_per_freq=model_config.n_p,
        f_initial=model_config.f_initial,
        title="Hierarchical Mask Schedule: Inference vs Generative Modes",
    )
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_hierarchical_masks.png")

    # Plot 4: Attractor convergence trajectories
    fig4 = figures.plot_attractor_convergence(
        test_queries_list,
        test_retrievals_list,
        test_targets_list,
        n_p_per_freq=model_config.n_p,
    )
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_attractor_convergence.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_attractor_convergence.png")

    # Plot 5: Retrieval quality across noise levels
    fig5 = figures.plot_retrieval_quality(
        errors_by_mode,
        noise_levels,
        xlabel="Noise Level (logit-space σ)",
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_retrieval_quality.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_retrieval_quality.png")

    # Plot 6: Memory structure analysis (eigenvalues and block structure)
    fig6, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Eigenvalue spectrum
    axes[0].plot(eigenvalues_gen.numpy(), linewidth=2, color="steelblue")
    axes[0].set_title("Eigenvalue Spectrum", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True, alpha=0.3)

    # Block structure (within vs between frequency)
    block_stats = []
    start_idx = 0
    for freq_idx, n_p in enumerate(model_config.n_p):
        end_idx = start_idx + n_p
        within_block = M_gen[0, start_idx:end_idx, start_idx:end_idx]
        within_strength = torch.norm(within_block).item()
        between_block = M_gen[0, start_idx:end_idx, :].clone()
        between_block[:, start_idx:end_idx] = 0
        between_strength = torch.norm(between_block).item()
        block_stats.append({"freq": freq_idx, "within": within_strength, "between": between_strength})
        start_idx = end_idx

    freqs = [s["freq"] for s in block_stats]
    within = [s["within"] for s in block_stats]
    between = [s["between"] for s in block_stats]

    x = np.arange(len(freqs))
    width = 0.35
    axes[1].bar(x - width / 2, within, width, label="Within-frequency", color="forestgreen")
    axes[1].bar(x + width / 2, between, width, label="Between-frequency", color="coral")
    axes[1].set_title("Hierarchical Block Structure", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Frequency Module")
    axes[1].set_ylabel("Connection Strength")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"f{i}" for i in freqs])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    fig6.tight_layout()

    if config.save_plots:
        fig6.savefig(config.output_dir / "06_memory_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_memory_structure.png")

    print()
    print("=" * 80)
    print("HPC Components Summary")
    print("=" * 80)
    print("\nSTORAGE (Hebbian Learning):")
    print(f"  Input: p_inferred, p_generated - [batch, {n_p_total}]")
    print(f"    ↓ Hebbian update (η={config.eta}, λ={config.lambda_})")
    print(f"  Output: Memory matrices M_gen, M_inf - [batch, {n_p_total}, {n_p_total}]")
    print(f"  Training: {config.n_training_steps} steps")
    print(f"  Final strength: {memory_strengths[-1]:.4f}")
    if not model_config.common_memory:
        print(f"  M_gen/M_inf similarity: {cosine_sims[-1]:.4f}")
    print("\nRETRIEVAL (Attractor Dynamics):")
    print(f"  Input: p_query (noisy) - List[{model_config.n_f}] of [batch, n_p[f]]")
    print(f"    ↓ Attractor iterations (κ={config.kappa}, {model_config.i_attractor} steps)")
    print(f"  Output: p_retrieved (refined) - List[{model_config.n_f}] of [batch, n_p[f]]")
    print(f"  Retrieval improvement: {avg_improvement:.1f}%")
    print(f"  SNR: {snr_db:.2f} dB")
    print("=" * 80)
    print()
    print("TEM Theory:")
    print("  • HPC stores spatial associations via Hebbian plasticity")
    print("  • Memory update strengthens co-active patterns (p_inf, p_gen)")
    print("  • Attractor dynamics refine noisy queries through iterative retrieval")
    print("  • Hierarchical masking enables coarse-to-fine convergence")
    print("  • Dual memory supports both inference (x→p) and generation (g→p)")
    print("=" * 80)
    print()
    if config.save_plots:
        print(f"All 6 visualizations saved to: {config.output_dir}")
    else:
        print("Plots not saved (use --save_plots true to save)")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
