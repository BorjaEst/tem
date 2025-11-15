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
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for attractor dynamics example.

    This config implements the AttractorParams protocol, allowing direct
    instantiation of AttractorDynamics component.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="memory_attractor")

    # Environment configuration (used only for parameter sizing)
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for spatial environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Memory architecture
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of hierarchical frequency modules")
    n_g_per_module: int = Field(default=10, ge=5, le=20, description="Grid cells per frequency module")
    n_x_c: int = Field(default=5, ge=2, le=20, description="Compressed sensory dimensions")

    # Attractor dynamics parameters
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor decay term (stability)")

    # Memory initialization (minimal training for realistic memory)
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Remembering rate for memory initialization")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Forgetting rate for memory initialization")
    n_memory_init_steps: int = Field(default=20, ge=5, le=100, description="Steps for memory initialization")
    batch_size: int = Field(default=8, ge=1, le=32, description="Batch size for memory initialization")

    # Retrieval testing configuration
    n_test_queries: int = Field(default=5, ge=1, le=20, description="Number of test retrieval queries")
    noise_level: float = Field(default=1.0, ge=0.0, le=5.0, description="Noise level in logit space (std of Gaussian noise added before softmax)")

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
    # MemoryStorageParams Protocol Implementation (for initialization)
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
    # AttractorParams Protocol Implementation
    # ==============================================================================

    @computed_field(description="Number of attractor iterations")
    @property
    def i_attractor_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Hierarchical masks for inference retrieval")
    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        i_attractor_max_freq = list(range(1, self.n_frequencies + 1))
        inf_masks, _ = utils.create_p_retrieve_masks(
            n_p=self.n_p_calculated,  # Pass number of place cells per frequency
            i_attractor=self.i_attractor_calculated,  # Total iterations
            i_attractor_max_freq_inf=i_attractor_max_freq,  # Progressive unmasking
            i_attractor_max_freq_gen=i_attractor_max_freq,  # Not used, but required
        )
        return inf_masks

    @computed_field(description="Hierarchical masks for generative retrieval")
    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        i_attractor_max_freq = list(range(1, self.n_frequencies + 1))
        _, gen_masks = utils.create_p_retrieve_masks(
            n_p=self.n_p_calculated,  # Pass number of place cells per frequency
            i_attractor=self.i_attractor_calculated,  # Total iterations
            i_attractor_max_freq_inf=i_attractor_max_freq,  # Not used, but required
            i_attractor_max_freq_gen=i_attractor_max_freq,  # Progressive unmasking
        )
        return gen_masks


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the attractor dynamics experiment with visualizations."""
    # Parse CLI arguments and create configuration
    # This implements AttractorParams protocol for direct instantiation
    config = ExampleConfig()

    print(f"Attractor Dynamics Experiment Configuration:")
    print(f"  Architecture: {config.n_frequencies} frequencies × {config.n_g_per_module} grid cells × {config.n_x_c} sensory dims")
    print(f"  Total place cells: {sum(config.n_p_calculated)}")
    print(f"  Attractor parameters: κ={config.kappa}, iterations={config.i_attractor_calculated}")
    print(f"  Test queries: {config.n_test_queries} with {config.noise_level} noise")
    print()

    # =========================================================================
    # PHASE 1: Initialize Memory and Attractor
    # =========================================================================
    # We need a memory matrix for retrieval. Create one through minimal Hebbian training.
    print("Initializing memory matrix through minimal Hebbian learning...")
    storage = MemoryStorage(config)
    n_p_total = sum(config.n_p_calculated)

    for step in range(config.n_memory_init_steps):
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)
        storage.update(p_inferred, p_generated, eta=config.eta, lamb=config.lambda_)

    m_gen_strength = torch.norm(storage.M_gen).item()
    print(f"  Memory initialized: M_gen strength={m_gen_strength:.4f}")
    print()

    # AttractorDynamics implements iterative retrieval with hierarchical masking
    # It implements: p[t+1] = κ*p[t] + M^T@p[t] * mask[t]
    attractor = AttractorDynamics(config)

    print(f"Attractor dynamics initialized with {config.i_attractor_calculated} iterations")
    print(f"  Hierarchical masking schedule (inference mode):")
    for it, mask in enumerate(attractor.p_retrieve_mask_inf):
        n_active = mask.sum().item()
        print(f"    Iteration {it+1}: {n_active}/{n_p_total} neurons active ({n_active/n_p_total*100:.1f}%)")
    print()

    # =========================================================================
    # PHASE 2: Test Attractor Retrieval Quality
    # =========================================================================
    print(f"Testing attractor retrieval with {config.n_test_queries} queries...")
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
    signal_power = (test_targets**2).mean()
    noise_power = ((test_targets - test_queries) ** 2).mean()
    snr_db = 10 * torch.log10(signal_power / noise_power) if noise_power > 0 else float("inf")
    print(f"  Signal-to-noise ratio: {snr_db:.2f} dB")
    print()

    # Run attractor dynamics: iteratively refine queries toward stored patterns
    # p[t+1] = κ*p[t] + M^T@p[t] * mask[t]
    # - κ (kappa): decay term for stability
    # - M^T@p[t]: memory-driven update (pulls toward associated patterns)
    # - mask[t]: hierarchical early-stopping (coarse→fine refinement)
    M_inf = storage.get_memory(for_inference=True)
    test_retrievals = attractor.retrieve(test_queries, M_inf, for_inference=True)

    # Package results for visualization
    # Convert from batched tensors to lists of individual patterns
    queries_list = [test_queries[i] for i in range(config.n_test_queries)]
    retrievals_list = [test_retrievals[i] for i in range(config.n_test_queries)]
    targets_list = [test_targets[i] for i in range(config.n_test_queries)]

    # Compute retrieval quality metrics
    # MSE measures how close retrieved patterns are to ground truth
    print("Retrieval quality metrics:")
    improvements = []
    for i in range(config.n_test_queries):
        query_error = torch.nn.functional.mse_loss(test_queries[i], test_targets[i]).item()
        retrieval_error = torch.nn.functional.mse_loss(test_retrievals[i], test_targets[i]).item()
        improvement = ((query_error - retrieval_error) / query_error) * 100  # Percentage improvement
        improvements.append(improvement)
        print(f"  Query {i+1}: query_error={query_error:.6f}, retrieval_error={retrieval_error:.6f}, improvement={improvement:.1f}%")

    avg_improvement = np.mean(improvements)
    print(f"  Average improvement: {avg_improvement:.1f}%")
    print()

    # =========================================================================
    # PHASE 3: Robustness Analysis Across Noise Levels
    # =========================================================================
    print("Testing robustness across noise levels...")
    noise_levels = [0.5, 1.0, 1.5, 2.0, 3.0]  # Logit-space noise levels
    errors_by_mode = {"Inference": [], "Generative": []}  # Compare dual memories

    for noise_level in noise_levels:
        # Add noise in logit space for realistic corruption
        # This preserves the probability distribution structure
        noise_logits = torch.randn_like(target_logits) * noise_level
        noisy_query_logits = target_logits + noise_logits
        noisy_queries = noisy_query_logits.softmax(dim=1)

        # Test inference memory (used for sensory→location inference)
        retrieved_inf = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=True), for_inference=True)
        error_inf = torch.nn.functional.mse_loss(retrieved_inf, test_targets).item()
        errors_by_mode["Inference"].append(error_inf)

        # Test generative memory (used for abstract→location prediction)
        retrieved_gen = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=False), for_inference=False)
        error_gen = torch.nn.functional.mse_loss(retrieved_gen, test_targets).item()
        errors_by_mode["Generative"].append(error_gen)

        # Compute SNR for this noise level
        signal_power = (test_targets**2).mean()
        noise_power = ((test_targets - noisy_queries) ** 2).mean()
        snr_db = 10 * torch.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

        print(f"  Noise {noise_level:.1f} (SNR={snr_db:+.1f}dB): Inference MSE={error_inf:.6f}, Generative MSE={error_gen:.6f}")

    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Generating visualizations...")

    # Plot 1: Hierarchical retrieval masks
    # Visualize progressive unmasking schedule (coarse→fine)
    fig1 = figures.plot_hierarchical_masks(
        attractor.p_retrieve_mask_inf,  # Binary masks for each iteration
        n_p_per_freq=config.n_p_calculated,  # Frequency boundaries
    )
    if config.save_plots:
        save_path = config.output_dir / "01_hierarchical_masks.png"
        fig1.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")

    # Plot 2: Attractor convergence trajectories
    # Show how noisy queries are iteratively refined toward targets
    fig2 = figures.plot_attractor_convergence(
        queries_list,  # Initial noisy patterns
        retrievals_list,  # Final retrieved patterns
        targets_list,  # Ground truth patterns
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
    print(f"All outputs saved to: {config.output_dir}")

    # Display plots interactively or just save them
    if config.show_plots:
        plt.show()  # Blocks until user closes windows
    else:
        plt.close("all")  # Clean up memory
