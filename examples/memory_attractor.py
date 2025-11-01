#!/usr/bin/env python3
"""Memory attractor dynamics example with Hebbian learning visualization.

This example demonstrates the torch_tem.memory module capabilities:
- MemoryStorage initialization and Hebbian updates
- AttractorDynamics retrieval with hierarchical early-stopping
- Memory learning through simulated spatial navigation
- Convergence analysis across different query patterns
- Hierarchical mask effects on retrieval quality
- Dual memory (inference vs. generative) comparison

The attractor dynamics implement content-addressable memory recall, where noisy
or partial query patterns are iteratively refined toward stored spatial patterns.
This is central to TEM's ability to infer locations from sensory observations
and predict future locations from abstract transitions.

Usage:
    python examples/memory_attractor.py --n-locations 25 --n-training-steps 100
    python examples/memory_attractor.py --kappa 0.9 --eta 0.4 --lambda 0.95
    python examples/memory_attractor.py --help
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

# Import visualization functions from torch_tem.figures
from torch_tem import data, figures
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for memory attractor dynamics example.

    This config implements both AttractorParams and MemoryStorageParams protocols,
    allowing direct instantiation of memory components.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="memory_attractor")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for spatial environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Memory architecture
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of hierarchical frequency modules")
    n_g_per_module: int = Field(default=10, ge=5, le=20, description="Grid cells per frequency module")
    n_x_c: int = Field(default=5, ge=2, le=20, description="Compressed sensory dimensions")

    # Hebbian learning parameters
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Remembering rate (Hebbian learning strength)")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Forgetting rate (memory decay)")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor decay term (stability)")

    # Training configuration
    n_training_steps: int = Field(default=50, ge=10, le=500, description="Number of Hebbian updates")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory updates")
    n_test_queries: int = Field(default=5, ge=1, le=20, description="Number of test retrieval queries")
    noise_level: float = Field(default=0.3, ge=0.0, le=1.0, description="Noise level for query patterns")

    # Memory configuration
    use_dual_memory: bool = Field(default=True, description="Use separate inference/generative memories")
    common_memory: bool = Field(default=False, description="Share memory between inference and generation")

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
        """Place cells = grid cells * sensory dimensions."""
        return [self.n_g_per_module * self.n_x_c for _ in range(self.n_frequencies)]

    @computed_field(description="Hierarchical mask for memory updates")
    @property
    def p_update_mask_calculated(self) -> Tensor:
        """Allow connections only from low to high frequency (coarse to fine)."""
        n_p_total = sum(self.n_p_calculated)
        mask = torch.zeros(n_p_total, n_p_total)

        # Build hierarchical mask: low freq can connect to all, high freq only to higher
        start_idx = 0
        for freq_idx, n_p in enumerate(self.n_p_calculated):
            # This frequency can connect to itself and all higher frequencies
            end_idx = start_idx + n_p
            mask[start_idx:end_idx, start_idx:] = 1.0
            start_idx = end_idx

        return mask

    @property
    def use_p_inf(self) -> bool:
        """Whether to use inference-based grounded locations."""
        return self.use_dual_memory

    # ==============================================================================
    # AttractorParams Protocol Implementation
    # ==============================================================================

    @computed_field(description="Number of attractor iterations")
    @property
    def i_attractor_calculated(self) -> int:
        """One iteration per frequency for hierarchical convergence."""
        return self.n_frequencies

    @computed_field(description="Hierarchical masks for inference retrieval")
    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        """Progressive unmasking: start with low freq, gradually enable higher."""
        masks = []
        n_p_total = sum(self.n_p_calculated)

        for iteration in range(self.n_frequencies):
            mask = torch.zeros(n_p_total)
            # Enable frequencies up to current iteration (0 = lowest freq)
            start_idx = 0
            for freq_idx in range(iteration + 1):
                n_p = self.n_p_calculated[freq_idx]
                mask[start_idx : start_idx + n_p] = 1.0
                start_idx += n_p
            masks.append(mask)

        return masks

    @computed_field(description="Hierarchical masks for generative retrieval")
    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        """Same schedule as inference (can be customized for different dynamics)."""
        return self.p_retrieve_mask_inf_calculated


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the memory attractor dynamics experiment with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Memory Attractor Dynamics Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size} ({config.n_x} observations)")
    print(f"  Frequency modules: {config.n_frequencies}")
    print(f"  Place cells per module: {config.n_p_calculated}")
    print(f"  Total place cells: {sum(config.n_p_calculated)}")
    print(f"  Hebbian rates: η={config.eta}, λ={config.lambda_}, κ={config.kappa}")
    print(f"  Training steps: {config.n_training_steps}")

    # Initialize memory components
    storage = MemoryStorage(config)
    attractor = AttractorDynamics(config)

    print(f"\nMemory Architecture:")
    print(f"  Dual memory: {storage.use_dual_memory}")
    print(f"  Attractor iterations: {attractor.i_attractor}")
    print(f"  Hierarchical masks: {len(attractor.p_retrieve_mask_inf)}")

    # Generate synthetic spatial patterns for training
    n_p_total = sum(config.n_p_calculated)
    memory_strengths = []
    cosine_sims = []

    print(f"\n{'Training Progress':-^80}")
    for step in range(config.n_training_steps):
        # Generate random grounded locations (simulating spatial navigation)
        # In TEM, these come from: p = g ⊗ x (grid cells ⊗ sensory input)
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # Hebbian update: M = λ*M + η*outer(p_inf, p_gen)
        storage.update(p_inferred, p_generated, eta=config.eta, lamb=config.lambda_)

        # Track learning progress
        m_gen_strength = torch.norm(storage.M_gen).item()
        memory_strengths.append(m_gen_strength)

        if storage.use_dual_memory:
            # Measure divergence between inference and generative memories
            m_gen_flat = storage.M_gen.flatten()
            m_inf_flat = storage.M_inf.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(m_gen_flat, m_inf_flat, dim=0).item()
            cosine_sims.append(cosine_sim)

        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Step {step+1:3d}: M_gen strength = {m_gen_strength:8.4f}", end="")
            if storage.use_dual_memory:
                print(f", M_gen ↔ M_inf similarity = {cosine_sim:.4f}")
            else:
                print()

    # Test attractor dynamics with noisy queries
    print(f"\n{'Testing Attractor Retrieval':-^80}")

    # Create test patterns (stored patterns from training distribution)
    test_targets = torch.randn(config.n_test_queries, n_p_total).softmax(dim=1)

    # Add noise to create queries
    noise = torch.randn_like(test_targets) * config.noise_level
    test_queries = test_targets + noise
    test_queries = test_queries.clamp(min=0)  # Ensure non-negative activations

    # Retrieve using attractor dynamics
    M_inf = storage.get_memory(for_inference=True)
    test_retrievals = attractor.retrieve(test_queries, M_inf, for_inference=True)

    # Compute retrieval quality
    queries_list = [test_queries[i] for i in range(config.n_test_queries)]
    retrievals_list = [test_retrievals[i] for i in range(config.n_test_queries)]
    targets_list = [test_targets[i] for i in range(config.n_test_queries)]

    for i in range(config.n_test_queries):
        query_error = torch.nn.functional.mse_loss(test_queries[i], test_targets[i]).item()
        retrieval_error = torch.nn.functional.mse_loss(test_retrievals[i], test_targets[i]).item()
        improvement = ((query_error - retrieval_error) / query_error) * 100
        print(f"  Query {i+1}: Error {query_error:.6f} → {retrieval_error:.6f} (↓{improvement:.1f}%)")

    # Test robustness to different noise levels
    print(f"\n{'Robustness Analysis':-^80}")
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors_by_mode = {"Inference": [], "Generative": []}

    for noise_level in noise_levels:
        noise = torch.randn_like(test_targets) * noise_level
        noisy_queries = (test_targets + noise).clamp(min=0)

        # Test inference memory
        retrieved_inf = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=True), for_inference=True)
        error_inf = torch.nn.functional.mse_loss(retrieved_inf, test_targets).item()
        errors_by_mode["Inference"].append(error_inf)

        # Test generative memory
        retrieved_gen = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=False), for_inference=False)
        error_gen = torch.nn.functional.mse_loss(retrieved_gen, test_targets).item()
        errors_by_mode["Generative"].append(error_gen)

    print(f"  Noise levels tested: {noise_levels}")
    print(f"  Inference errors: {[f'{e:.6f}' for e in errors_by_mode['Inference']]}")
    print(f"  Generative errors: {[f'{e:.6f}' for e in errors_by_mode['Generative']]}")

    # Generate visualizations
    print(f"\n{'Generating Visualizations':-^80}")

    # Plot 1: Memory matrices
    fig1 = figures.plot_memory_matrices(
        storage.M_gen,
        storage.get_memory(for_inference=True),
        n_p_per_freq=config.n_p_calculated,
        n_training_steps=config.n_training_steps,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_memory_matrices.png")

    # Plot 2: Learning curve
    fig2 = figures.plot_learning_curve(memory_strengths, cosine_sims if storage.use_dual_memory else None)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_learning_curve.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_learning_curve.png")

    # Plot 3: Hierarchical masks
    fig3 = figures.plot_hierarchical_masks(attractor.p_retrieve_mask_inf, n_p_per_freq=config.n_p_calculated)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_hierarchical_masks.png")

    # Plot 4: Attractor convergence
    fig4 = figures.plot_attractor_convergence(queries_list, retrievals_list, targets_list)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_attractor_convergence.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_attractor_convergence.png")

    # Plot 5: Retrieval quality
    fig5 = figures.plot_retrieval_quality(errors_by_mode, noise_levels)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_retrieval_quality.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_retrieval_quality.png")

    print(f"\n{'Summary':-^80}")
    print(f"  Final memory strength: {memory_strengths[-1]:.4f}")
    if storage.use_dual_memory:
        print(f"  Final M_gen ↔ M_inf similarity: {cosine_sims[-1]:.4f}")
    print(f"  Mean retrieval improvement: {improvement:.1f}%")
    print(f"  Output directory: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        print("\nDisplaying plots...")
        plt.show()
    else:
        plt.close("all")

    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)
