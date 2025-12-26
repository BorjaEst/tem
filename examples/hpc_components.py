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
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import figures, hpc, utils
from torch_tem.core.hpc import AttractorConfig, GroundedLocConfig, StorageConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for HPC memory components example.

    Defines architecture parameters for demonstrating HPC components
    in isolation using entirely synthetic data (random place cell patterns).
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="hpc_components")

    # Memory structure parameters
    common_memory: bool = Field(default=False, description="Use shared memory for inference and generation")
    attractor: AttractorConfig = Field(default_factory=AttractorConfig, description="Attractor dynamics configuration")
    grounded: GroundedLocConfig = Field(default_factory=GroundedLocConfig, description="Grounded location inference configuration")
    storage: StorageConfig = Field(default_factory=StorageConfig, description="Memory storage configuration")

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


# Model architecture; not configurable via CLI
N_P = [10, 10, 8, 6, 6]  # Place cells per frequency (5 modules)
N_F = len(N_P)  # Number of frequency modules
N_F_G = 3  # Grid cell modules (for mask generation)
I_ATTRACTOR = 3  # Attractor iterations
F_EXTENDED = [0.95, 0.7, 0.4, 0.2, 0.1]  # Extended frequencies
MAX_FREQ_INF = [2, 3, 3, 3, 3]  # Max attractor iteration per frequency (conservative for inference)
MAX_FREQ_GEN = [3, 3, 3, 3, 3]  # Max attractor iteration per frequency (all active for generation)
DEVICE = torch.device("cpu")  # Change to "cuda" if GPU is available

# Create context for HPC components
update_mask = utils.create_p_update_mask(N_P, N_F_G, F_EXTENDED)
mask_inf = utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_INF)
mask_gen = utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_GEN)


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

    print("=" * 80)
    print("Hippocampal Memory Components")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Place cells per frequency: {N_P}")
    print(f"  Total place cells: {sum(N_P)}")
    print(f"  Memory: η={config.storage.eta}, λ={config.storage.lambda_}, κ={config.attractor.kappa}")
    print(f"  Training: {config.n_training_steps} steps, batch_size={config.batch_size}")
    print(f"  Testing: {config.n_test_queries} queries, noise_level={config.noise_level}")
    print()

    # =========================================================================
    # PHASE 1: Initialize HPC Components
    # =========================================================================
    print("Phase 1: Initializing HPC components...")

    # HPC AttractorDynamics: Iterative pattern completion
    # Refines noisy queries using hierarchical coarse-to-fine retrieval
    # Update rule: p[t+1] = mask[t] * activation(κ*p[t] + M@p[t]) + (1-mask[t])*p[t]
    attractor = hpc.AttractorDynamics(mask_inf, mask_gen, config.attractor)

    # TODO: We need to add demonstrations about grounded location inference
    ground = hpc.GroundedLocInference(config.grounded)

    # HPC MemoryStorage: Hebbian plasticity
    # Manages M_gen (generative) and M_inf (inference) memory matrices
    # Update rule: M = λ*M + η*outer(p_inf + p_gen, p_inf - p_gen) * mask
    storage = hpc.MemoryStorage(update_mask, config.storage)

    # Initialize memory matrices (functional interface)
    n_p_total = sum(N_P)
    memory = utils.create_initial_memory(n_p_total, config.batch_size, not config.common_memory, DEVICE)
    M_gen = memory[0]
    M_inf = memory[1]

    print(f"  ✓ MemoryStorage: η={config.storage.eta}, λ={config.storage.lambda_}")
    print(f"    Dual memory: {not config.common_memory}")
    print(f"  ✓ AttractorDynamics: {I_ATTRACTOR} iterations with hierarchical masking")
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
        p_inferred = torch.randn(config.batch_size, sum(N_P)).softmax(dim=1)
        p_generated = torch.randn(config.batch_size, sum(N_P)).softmax(dim=1)

        # Store pre-update state
        M_gen_before = M_gen.clone()

        # Apply Hebbian learning (functional interface)
        M_gen = storage.update(p_inferred, p_generated, M_gen)
        if M_inf is not None:
            M_inf = storage.update(p_inferred, p_generated, M_inf)

        # Monitor learning dynamics
        m_gen_strength = torch.norm(M_gen).item()
        memory_strengths.append(m_gen_strength)

        update_magnitude = torch.norm(M_gen - M_gen_before).item()
        update_magnitudes.append(update_magnitude)

        if M_inf is not None:
            # Measure dual memory divergence
            M_gen_flat = M_gen.flatten()
            M_inf_flat = M_inf.flatten()
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
    test_targets = torch.randn(config.batch_size, sum(N_P)).softmax(dim=1)

    # Add noise in logit space before softmax
    target_logits = torch.log(test_targets + 1e-8)  # Convert back to logits
    noise_logits = torch.randn_like(target_logits) * config.noise_level
    query_logits = target_logits + noise_logits
    test_queries = query_logits.softmax(dim=1)

    # Compute signal-to-noise ratio
    signal_power = (test_targets**2).mean()
    noise_power = ((test_queries - test_targets) ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    print(f"  Signal-to-noise ratio: {snr_db.item():.2f} dB")

    # Convert to per-frequency format for attractor
    test_queries_list = []
    test_targets_list = []
    cumsum = 0
    for n_p in N_P:
        test_queries_list.append(test_queries[:, cumsum : cumsum + n_p])
        test_targets_list.append(test_targets[:, cumsum : cumsum + n_p])
        cumsum += n_p

    # Run attractor dynamics
    M_for_retrieval = M_inf if M_inf is not None else M_gen
    test_retrievals_list = attractor(test_queries_list, M_for_retrieval, for_inference=True)

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

        # Convert to per-frequency format
        noisy_queries_list = []
        cumsum = 0
        for n_p in N_P:
            noisy_queries_list.append(noisy_queries[:, cumsum : cumsum + n_p])
            cumsum += n_p

        # Test inference mode
        M_for_inf = M_inf if M_inf is not None else M_gen
        retrieved_inf = attractor(noisy_queries_list, M_for_inf, for_inference=True)
        retrieved_inf_cat = torch.cat(retrieved_inf, dim=1)
        mse_inf = torch.nn.functional.mse_loss(retrieved_inf_cat, test_targets).item()
        errors_by_mode["Inference"].append(mse_inf)

        # Test generative mode
        retrieved_gen = attractor(noisy_queries_list, M_gen, for_inference=False)
        retrieved_gen_cat = torch.cat(retrieved_gen, dim=1)
        mse_gen = torch.nn.functional.mse_loss(retrieved_gen_cat, test_targets).item()
        errors_by_mode["Generative"].append(mse_gen)

    print(f"  ✓ Tested across {len(noise_levels)} noise levels")
    print()

    # =========================================================================
    # PHASE 5: Analyze Memory Structure
    # =========================================================================
    print("Phase 5: Analyzing learned memory structure...")

    # M_gen and M_inf already in scope from training loop
    M_for_analysis = M_inf if M_inf is not None else M_gen

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
        M_gen[0],  # Use first batch element for visualization
        M_for_analysis[0],
        n_p_per_freq=N_P,
        n_training_steps=config.n_training_steps,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_memory_matrices.png")

    # Plot 2: Learning dynamics
    fig2 = figures.plot_learning_curve(
        memory_strengths,
        cosine_sims if not config.common_memory else None,
    )
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_learning_dynamics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_learning_dynamics.png")

    # Plot 3: Hierarchical retrieval masks
    # Compare inference vs generative modes to show hierarchical early-stopping
    fig3 = figures.plot_hierarchical_masks(mask_inf, mask_gen, N_P, F_EXTENDED, title="Hierarchical Mask Schedule: Inference vs Generative Modes")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_hierarchical_masks.png")

    # Plot 4: Attractor convergence trajectories
    fig4 = figures.plot_attractor_convergence(test_queries_list, test_retrievals_list, test_targets_list, n_p_per_freq=N_P)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_attractor_convergence.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_attractor_convergence.png")

    # Plot 5: Retrieval quality across noise levels
    fig5 = figures.plot_retrieval_quality(errors_by_mode, noise_levels, xlabel="Noise Level (logit-space σ)")
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_retrieval_quality.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_retrieval_quality.png")

    # Plot 6: Memory structure analysis (eigenvalues and block structure)
    fig6 = figures.plot_memory_structure_analysis(M_gen[0], n_p_per_freq=N_P)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_memory_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_memory_structure.png")

    print()
    print("=" * 80)
    print("HPC Components Summary")
    print("=" * 80)
    print("\nSTORAGE (Hebbian Learning):")
    print(f"  Input: p_inferred, p_generated - [batch, {sum(N_P)}]")
    print(f"    ↓ Hebbian update (η={config.storage.eta}, λ={config.storage.lambda_})")
    print(f"  Output: Memory matrices M_gen, M_inf - [batch, {sum(N_P)}, {sum(N_P)}]")
    print(f"  Training: {config.n_training_steps} steps")
    print(f"  Final strength: {memory_strengths[-1]:.4f}")
    if not config.common_memory:
        print(f"  M_gen/M_inf similarity: {cosine_sims[-1]:.4f}")
    print("\nRETRIEVAL (Attractor Dynamics):")
    print(f"  Input: p_query (noisy) - List[{N_F}] of [batch, n_p[f]]")
    print(f"    ↓ Attractor iterations (κ={config.attractor.kappa}, {I_ATTRACTOR} steps)")
    print(f"  Output: p_retrieved (refined) - List[{N_F}] of [batch, n_p[f]]")
    print(f"  Retrieval improvement: {avg_improvement:.1f}%")
    print(f"  SNR: {snr_db.item():.2f} dB")
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
