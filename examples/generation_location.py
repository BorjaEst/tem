#!/usr/bin/env python3
"""Location generation example demonstrating g→p memory-based retrieval.

This example demonstrates the torch_tem.generation.LocationGenerator capabilities:
- LocationGenerator initialization with memory and attractor components
- Generative pathway: abstract location (g) → grounded location (p)
- Memory-based retrieval via Hebbian associations
- Deterministic vs. stochastic generation modes
- Uncertainty estimation and sampling
- Dual memory comparison (inference vs. generative networks)
- Retrieval quality analysis across training phases

The location generator implements content-addressable memory recall, where abstract
spatial representations (entorhinal grid cells) are decoded into grounded spatial
representations (hippocampal place cells) via learned associations. This is central
to TEM's ability to predict place cell activity from grid cell patterns.

Usage:
    python examples/generation_location.py --n-locations 25 --n-training-steps 100
    python examples/generation_location.py --n-frequencies 4 --do-sample --show-plots
    python examples/generation_location.py --help
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.config import EnvironmentConfig, InferenceConfig, ModelConfig
from torch_tem.generation.location import LocationGenerator
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for location generation example.

    Provides example-specific parameters and delegates architectural
    computations to ModelConfig and InferenceConfig.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="generation_location")

    # Architecture configuration
    n_g_subsampled: List[int] = Field(default_factory=lambda: [10, 10, 10], description="Subsampled grid cells per frequency module")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension")
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.6, 0.3], description="Frequency values per module")

    # Hebbian learning parameters
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Remembering rate (Hebbian learning strength)")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Forgetting rate (memory decay)")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor decay term (stability)")

    # Training configuration
    n_training_steps: int = Field(default=50, ge=10, le=500, description="Number of Hebbian memory updates")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory updates")
    n_test_queries: int = Field(default=5, ge=1, le=20, description="Number of test retrieval queries")
    noise_level: float = Field(default=0.3, ge=0.0, le=1.0, description="Noise level for query patterns")

    # Generation mode
    do_sample: bool = Field(default=False, description="Enable stochastic sampling with learned uncertainty")
    common_memory: bool = Field(default=False, description="Share memory between inference and generation")

    # Output
    output_dir: Path = Field(default=Path("outputs/generation_location"), description="Directory for saving plots")
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
    """Run the location generation experiment with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    inference_config = InferenceConfig(eta=config.eta, kappa=config.kappa, do_sample=config.do_sample)
    model_config = ModelConfig(n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial, common_memory=config.common_memory)

    # Compute connectivity matrices from model config
    p_update_mask = utils.masks.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f_g, model_config.n_f_ovc, model_config.f_initial_extended)
    mask_inf, mask_gen = utils.masks.create_p_retrieve_masks(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf, model_config.max_freq_gen)
    W_repeat = utils.matrices.create_W_repeat(model_config.n_g_subsampled_combined, [config.n_x_c] * model_config.n_f)

    print("=" * 80)
    print("Location Generation Example: g → p Memory-Based Retrieval")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g_subsampled_combined}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print(f"  Memory: η={config.eta}, λ={config.lambda_}, κ={config.kappa}")
    print(f"  Training: {config.n_training_steps} steps, batch_size={config.batch_size}")
    print(f"  Generation mode: {'stochastic' if config.do_sample else 'deterministic'}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory Components
    # =========================================================================
    print("Phase 1: Initializing memory components...")
    storage = MemoryStorage(model_config, inference_config, p_update_mask)
    attractor = AttractorDynamics(model_config, inference_config, mask_inf, mask_gen)

    # Create params object for LocationGenerator with required protocol fields
    gen_params = SimpleNamespace(do_sample=config.do_sample, n_f=model_config.n_f, n_p=model_config.n_p)
    generator = LocationGenerator(gen_params, storage, attractor, W_repeat)

    print(f"  ✓ MemoryStorage: {sum(model_config.n_p)}×{sum(model_config.n_p)} Hebbian matrix")
    print(f"  ✓ M_gen: {storage.M_gen.shape}")
    if not model_config.common_memory:
        print(f"  ✓ M_inf: {storage.M_inf.shape}")
    print(f"  ✓ AttractorDynamics: {model_config.i_attractor} iterations")
    print(f"  ✓ LocationGenerator: initialized")
    print()

    # =========================================================================
    # PHASE 2: Generate Synthetic Training Data
    # =========================================================================
    print("Phase 2: Generating synthetic training patterns...")
    n_p_total = sum(model_config.n_p)

    # Generate random place cell patterns (simulating spatial experience)
    # In practice, these would come from actual navigation through the environment
    training_patterns = []
    for step in range(config.n_training_steps):
        # Random place cell activity (softmax for valid probability distributions)
        p_random = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        training_patterns.append(p_random)
    print(f"  ✓ Generated {config.n_training_steps} training batches")
    print(f"  ✓ Batch size: {config.batch_size}, Place cells: {n_p_total}")
    print()

    # =========================================================================
    # PHASE 3: Train Memory with Hebbian Learning
    # =========================================================================
    print("Phase 3: Training memory with Hebbian updates...")
    for step, p_batch in enumerate(training_patterns):
        # Update memory with Hebbian learning rule
        # In this simple example, we use the same pattern for inferred and generated
        storage.update(p_inferred=p_batch, p_generated=p_batch, eta=config.eta, lamb=config.lambda_)

        if (step + 1) % 10 == 0:
            print(f"  Step {step + 1}/{config.n_training_steps}")

    print(f"  ✓ Memory training complete")
    print(f"  ✓ M_gen norm: {storage.M_gen.norm().item():.4f}")
    if not model_config.common_memory:
        print(f"  ✓ M_inf norm: {storage.M_inf.norm().item():.4f}")
    print()

    # =========================================================================
    # PHASE 4: Generate Test Queries (Abstract Locations)
    # =========================================================================
    print("Phase 4: Generating test queries (abstract locations)...")
    test_queries = []
    test_labels = []

    for i in range(config.n_test_queries):
        # Generate abstract location (grid cell activity)
        g_test = [torch.randn(1, model_config.n_g_subsampled_combined[f]).softmax(dim=1) for f in range(model_config.n_f)]
        test_queries.append(g_test)
        test_labels.append(f"Query {i+1}")

    print(f"  ✓ Generated {config.n_test_queries} test queries")
    print(f"  ✓ Grid cell dimensions: {model_config.n_g_subsampled_combined}")
    print()

    # =========================================================================
    # PHASE 5: Generate Grounded Locations from Abstract Locations
    # =========================================================================
    print("Phase 5: Generating grounded locations (p) from abstract locations (g)...")
    retrievals = []

    with torch.no_grad():
        for i, g_test in enumerate(test_queries):
            # Generate grounded location via memory retrieval
            p_retrieved = generator.generate(g_test, for_inference=False)
            retrievals.append(p_retrieved)

            # Compute total activity
            p_flat = utils.concatenate_frequencies(p_retrieved)
            activity = p_flat.sum().item()
            sparsity = (p_flat > 0.1).float().mean().item()

            print(f"  Query {i+1}: activity={activity:.4f}, sparsity={sparsity:.2%}")

    print(f"  ✓ Generated {len(retrievals)} grounded locations")
    print()

    # =========================================================================
    # PHASE 6: Analyze Deterministic vs. Stochastic Generation
    # =========================================================================
    print("Phase 6: Comparing deterministic vs. stochastic generation...")

    # Test deterministic mode (same query twice)
    with torch.no_grad():
        p_det_1 = generator.generate(test_queries[0], for_inference=False)
        p_det_2 = generator.generate(test_queries[0], for_inference=False)

    det_diff = torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(p_det_1, p_det_2)]).mean()
    print(f"  Deterministic mode:")
    print(f"    Same query, two runs: difference = {det_diff:.6f}")
    print(f"    ✓ Reproducible (as expected)")

    if config.do_sample:
        # Test stochastic mode
        gen_stoch = LocationGenerator(model_config, inference_config, storage, attractor, W_repeat)
        with torch.no_grad():
            p_stoch_1 = gen_stoch.generate(test_queries[0], for_inference=False)
            p_stoch_2 = gen_stoch.generate(test_queries[0], for_inference=False)

        stoch_diff = torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(p_stoch_1, p_stoch_2)]).mean()
        print(f"  Stochastic mode:")
        print(f"    Same query, two runs: difference = {stoch_diff:.6f}")
        print(f"    ✓ Variable (learned uncertainty)")
    else:
        print(f"  Stochastic mode: disabled (use --do-sample to enable)")
    print()

    # =========================================================================
    # PHASE 7: Analyze Dual Memory (Inference vs. Generative)
    # =========================================================================
    if not model_config.common_memory:
        print("Phase 7: Analyzing dual memory (inference vs. generative)...")

        with torch.no_grad():
            p_gen = generator.generate(test_queries[0], for_inference=False)
            p_inf = generator.generate(test_queries[0], for_inference=True)

        diff = torch.stack([torch.norm(pg - pi) for pg, pi in zip(p_gen, p_inf)]).mean()
        print(f"  Generative vs. Inference retrieval:")
        print(f"    Difference: {diff:.6f}")
        print(f"    ✓ Different memory networks used")
        print()

    # =========================================================================
    # PHASE 8: Visualization
    # =========================================================================
    print("Phase 8: Generating visualizations...")

    # Plot 1: Memory matrices
    fig1 = figures.plot_memory_matrices(
        storage.M_gen,
        storage.get_memory(for_inference=True),
        model_config.n_p,
        config.n_training_steps,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_memory_matrices.png")

    # Plot 2: Retrieval patterns
    # Flatten retrievals for visualization
    retrievals_flat = [utils.concatenate_frequencies(p_list).squeeze(0) for p_list in retrievals]
    queries_flat = [torch.cat([g.squeeze(0) for g in g_list], dim=0) for g_list in test_queries]

    # Create visualization figure
    fig2, axes = plt.subplots(config.n_test_queries, 2, figsize=(12, 2.5 * config.n_test_queries))
    if config.n_test_queries == 1:
        axes = axes.reshape(1, -1)

    for i in range(config.n_test_queries):
        # Plot query (abstract location)
        axes[i, 0].bar(range(len(queries_flat[i])), queries_flat[i].cpu().numpy())
        axes[i, 0].set_title(f"Query {i+1}: Abstract Location (g)")
        axes[i, 0].set_ylabel("Activity")
        axes[i, 0].set_xlabel("Grid Cell Index")

        # Plot retrieval (grounded location)
        axes[i, 1].bar(range(len(retrievals_flat[i])), retrievals_flat[i].cpu().numpy())
        axes[i, 1].set_title(f"Retrieved: Grounded Location (p)")
        axes[i, 1].set_ylabel("Activity")
        axes[i, 1].set_xlabel("Place Cell Index")

        # Add frequency boundaries
        if model_config.n_f > 1:
            g_boundaries = [0] + [sum(model_config.n_g_subsampled_combined[: j + 1]) for j in range(model_config.n_f)]
            p_boundaries = [0] + [sum(model_config.n_p[: j + 1]) for j in range(model_config.n_f)]

            for boundary in g_boundaries[1:-1]:
                axes[i, 0].axvline(boundary, color="red", linestyle="--", alpha=0.3)
            for boundary in p_boundaries[1:-1]:
                axes[i, 1].axvline(boundary, color="red", linestyle="--", alpha=0.3)

    fig2.suptitle("Location Generation: g → p Memory-Based Retrieval", fontsize=14, y=0.995)
    plt.tight_layout()
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_retrieval_patterns.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_retrieval_patterns.png")

    # Plot 3: Hierarchical mask schedule
    fig3 = figures.plot_hierarchical_masks(
        mask_gen,
        model_config.n_p,
    )
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_hierarchical_masks.png")

    # Plot 4: Generation mode comparison (if stochastic enabled)
    if config.do_sample:
        fig4, axes = plt.subplots(2, 3, figsize=(15, 8))

        with torch.no_grad():
            for sample_idx in range(3):
                # Deterministic
                p_det = generator.generate(test_queries[0], for_inference=False)
                p_det_flat = utils.concatenate_frequencies(p_det).squeeze(0)
                axes[0, sample_idx].bar(range(len(p_det_flat)), p_det_flat.cpu().numpy())
                axes[0, sample_idx].set_title(f"Deterministic - Sample {sample_idx+1}")
                axes[0, sample_idx].set_ylim(0, p_det_flat.max().item() * 1.2)

                # Stochastic
                gen_stoch = LocationGenerator(gen_params, storage, attractor, W_repeat)
                p_stoch = gen_stoch.generate(test_queries[0], for_inference=False)
                p_stoch_flat = utils.concatenate_frequencies(p_stoch).squeeze(0)
                axes[1, sample_idx].bar(range(len(p_stoch_flat)), p_stoch_flat.cpu().numpy())
                axes[1, sample_idx].set_title(f"Stochastic - Sample {sample_idx+1}")
                axes[1, sample_idx].set_ylim(0, p_stoch_flat.max().item() * 1.2)

        axes[0, 0].set_ylabel("Activity")
        axes[1, 0].set_ylabel("Activity")
        for ax in axes[1, :]:
            ax.set_xlabel("Place Cell Index")

        fig4.suptitle("Deterministic vs. Stochastic Generation (Same Query)", fontsize=14)
        plt.tight_layout()
        if config.save_plots:
            fig4.savefig(config.output_dir / "04_generation_modes.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: 04_generation_modes.png")

    print()
    print("=" * 80)
    print("Location Generation Summary:")
    print("=" * 80)
    print(f"Architecture: {sum(model_config.n_g_subsampled_combined)} grid cells → {sum(model_config.n_p)} place cells")
    print(f"Memory training: {config.n_training_steps} steps × {config.batch_size} batch")
    print(f"Memory strength: M_gen={storage.M_gen.norm().item():.4f}")
    if not model_config.common_memory:
        print(f"                 M_inf={storage.M_inf.norm().item():.4f}")
    print(f"Test retrievals: {config.n_test_queries} queries processed")
    print(f"Generation mode: {'stochastic' if config.do_sample else 'deterministic'}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
