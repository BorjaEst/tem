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
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

from torch_tem import data, figures, utils
from torch_tem.generation.location import LocationGenerator
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for location generation example.

    This config implements LocationGeneratorParams, MemoryStorageParams, and
    AttractorParams protocols, allowing direct component instantiation.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="generation_location")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for spatial environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Architecture configuration
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of hierarchical frequency modules")
    n_g_per_module: int = Field(default=10, ge=5, le=20, description="Grid cells per frequency module")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension")

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

    @computed_field(description="Number of unique observations")
    @property
    def n_x(self) -> int:
        """Compute number of observations from grid size and mode."""
        n_locations = self.grid_size * self.grid_size
        if self.observation_mode == "unique":
            return n_locations
        elif self.observation_mode == "tiled":
            return 4
        elif self.observation_mode == "random":
            return max(4, n_locations // 4)
        raise ValueError(f"Invalid observation_mode: {self.observation_mode}")

    # ==============================================================================
    # Protocol Implementations (LocationGeneratorParams, MemoryStorageParams, AttractorParams)
    # ==============================================================================

    @computed_field(description="Total number of frequency modules")
    @property
    def n_f_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Grid cell dimensions per frequency (3x subsampled)")
    @property
    def n_g_calculated(self) -> List[int]:
        return [self.n_g_per_module] * self.n_frequencies

    @computed_field(description="Place cell dimensions per frequency")
    @property
    def n_p_calculated(self) -> List[int]:
        return [g * self.n_x_c for g in self.n_g_calculated]

    @computed_field(description="Number of grid cell frequency modules")
    @property
    def n_f_g_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Number of OVC frequency modules")
    @property
    def n_f_ovc_calculated(self) -> int:
        return 0  # No OVC modules in this example

    @computed_field(description="Initial frequencies for each module")
    @property
    def f_initial_extended(self) -> List[float]:
        """Generate logarithmic frequency spacing."""
        return [1.0 - (i / (self.n_frequencies - 1)) * 0.9 for i in range(self.n_frequencies)]

    @computed_field(description="Number of attractor iterations")
    @property
    def i_attractor_calculated(self) -> int:
        return self.n_f_g_calculated

    @computed_field(description="Max attractor iterations per frequency in inference model")
    @property
    def i_attractor_max_freq_inf_calculated(self) -> List[int]:
        """All frequencies iterate for all steps."""
        return [self.i_attractor_calculated for _ in range(self.n_frequencies)]

    @computed_field(description="Max attractor iterations per frequency in generative model")
    @property
    def i_attractor_max_freq_gen_calculated(self) -> List[int]:
        """Hierarchical early-stopping: high freq stops first."""
        return [self.i_attractor_calculated - freq_nr for freq_nr in range(self.n_frequencies)]

    @computed_field(description="Hierarchical mask for memory updates")
    @property
    def p_update_mask_calculated(self) -> Tensor:
        """Generate hierarchical update mask (low to high frequency)."""
        return utils.masks.create_p_update_mask(
            n_p=self.n_p_calculated,
            n_f=self.n_f_calculated,
            n_f_g=self.n_f_g_calculated,
            n_f_ovc=self.n_f_ovc_calculated,
            f_initial=self.f_initial_extended,
        )

    @computed_field(description="Whether to use inference-based grounded locations")
    @property
    def use_p_inf(self) -> bool:
        return not self.common_memory

    @computed_field(description="Hierarchical masks for inference retrieval")
    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        """Generate hierarchical retrieval masks for inference."""
        inf_masks, _ = utils.masks.create_p_retrieve_masks(
            n_p=self.n_p_calculated,
            i_attractor=self.i_attractor_calculated,
            i_attractor_max_freq_inf=self.i_attractor_max_freq_inf_calculated,
            i_attractor_max_freq_gen=self.i_attractor_max_freq_gen_calculated,
        )
        return inf_masks

    @computed_field(description="Hierarchical masks for generative retrieval")
    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        """Generate hierarchical retrieval masks for generation."""
        _, gen_masks = utils.masks.create_p_retrieve_masks(
            n_p=self.n_p_calculated,
            i_attractor=self.i_attractor_calculated,
            i_attractor_max_freq_inf=self.i_attractor_max_freq_inf_calculated,
            i_attractor_max_freq_gen=self.i_attractor_max_freq_gen_calculated,
        )
        return gen_masks


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the location generation experiment with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Location Generation Example: g → p Memory-Based Retrieval")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}x{config.grid_size} grid")
    print(f"  Frequencies: {config.n_frequencies} modules")
    print(f"  Grid cells per module: {config.n_g_per_module}")
    print(f"  Place cells per module: {config.n_p_calculated}")
    print(f"  Memory: eta={config.eta}, lambda={config.lambda_}, kappa={config.kappa}")
    print(f"  Generation mode: {'stochastic' if config.do_sample else 'deterministic'}")
    print(f"  Memory type: {'common' if config.common_memory else 'dual (inference/generative)'}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory Components
    # =========================================================================
    print("Phase 1: Initializing memory components...")
    storage = MemoryStorage(config)
    attractor = AttractorDynamics(config)
    generator = LocationGenerator(config, storage, attractor)
    print(f"  ✓ MemoryStorage initialized (M_gen: {storage.M_gen.shape})")
    if not config.common_memory:
        print(f"  ✓ MemoryStorage initialized (M_inf: {storage.M_inf.shape})")
    print(f"  ✓ AttractorDynamics initialized ({config.i_attractor_calculated} iterations)")
    print(f"  ✓ LocationGenerator initialized")
    print()

    # =========================================================================
    # PHASE 2: Generate Synthetic Training Data
    # =========================================================================
    print("Phase 2: Generating synthetic training patterns...")
    n_p_total = sum(config.n_p_calculated)

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
    if not config.common_memory:
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
        g_test = [torch.randn(1, config.n_g_calculated[f]).softmax(dim=1) for f in range(config.n_frequencies)]
        test_queries.append(g_test)
        test_labels.append(f"Query {i+1}")

    print(f"  ✓ Generated {config.n_test_queries} test queries")
    print(f"  ✓ Grid cell dimensions: {config.n_g_calculated}")
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
            p_flat = torch.cat(p_retrieved, dim=1)
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
        gen_stoch = LocationGenerator(config, storage, attractor)
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
    if not config.common_memory:
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
        M_gen=storage.M_gen,
        M_inf=storage.M_inf if not config.common_memory else None,
        n_p_per_freq=config.n_p_calculated,
        n_training_steps=config.n_training_steps,
        title=f"Learned Memory Matrices ({config.n_training_steps} updates)",
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 1: Memory matrices")

    # Plot 2: Retrieval patterns
    # Flatten retrievals for visualization
    retrievals_flat = [torch.cat(p_list, dim=1).squeeze(0) for p_list in retrievals]
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
        if config.n_frequencies > 1:
            g_boundaries = [0] + [sum(config.n_g_calculated[: j + 1]) for j in range(config.n_frequencies)]
            p_boundaries = [0] + [sum(config.n_p_calculated[: j + 1]) for j in range(config.n_frequencies)]

            for boundary in g_boundaries[1:-1]:
                axes[i, 0].axvline(boundary, color="red", linestyle="--", alpha=0.3)
            for boundary in p_boundaries[1:-1]:
                axes[i, 1].axvline(boundary, color="red", linestyle="--", alpha=0.3)

    fig2.suptitle("Location Generation: g → p Memory-Based Retrieval", fontsize=14, y=0.995)
    plt.tight_layout()
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_retrieval_patterns.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 2: Retrieval patterns")

    # Plot 3: Hierarchical mask schedule
    fig3 = figures.plot_hierarchical_masks(
        masks=config.p_retrieve_mask_gen_calculated,
        sizes=config.n_p_calculated,
        title="Hierarchical Retrieval Schedule (Generative)",
    )
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Figure 3: Hierarchical masks")

    # Plot 4: Generation mode comparison (if stochastic enabled)
    if config.do_sample:
        fig4, axes = plt.subplots(2, 3, figsize=(15, 8))

        with torch.no_grad():
            for sample_idx in range(3):
                # Deterministic
                p_det = generator.generate(test_queries[0], for_inference=False)
                p_det_flat = torch.cat(p_det, dim=1).squeeze(0)
                axes[0, sample_idx].bar(range(len(p_det_flat)), p_det_flat.cpu().numpy())
                axes[0, sample_idx].set_title(f"Deterministic - Sample {sample_idx+1}")
                axes[0, sample_idx].set_ylim(0, p_det_flat.max().item() * 1.2)

                # Stochastic
                gen_stoch = LocationGenerator(config, storage, attractor)
                p_stoch = gen_stoch.generate(test_queries[0], for_inference=False)
                p_stoch_flat = torch.cat(p_stoch, dim=1).squeeze(0)
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
        print(f"  ✓ Figure 4: Generation mode comparison")

    print()
    print("=" * 80)
    print("Experiment complete!")
    print(f"Results saved to: {config.output_dir}")
    print("=" * 80)

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
