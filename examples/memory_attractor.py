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
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

# Import visualization functions from torch_tem.figures
from torch_tem import data, figures, utils
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
    """Run the memory attractor dynamics experiment with visualizations."""
    # Parse CLI arguments and create configuration
    # This implements Protocol interfaces for MemoryStorage and AttractorDynamics
    config = ExampleConfig()

    # =========================================================================
    # PHASE 1: Initialize Memory Components
    # =========================================================================
    # MemoryStorage manages Hebbian memory matrices (M_gen, M_inf)
    # It implements: M = λ*M + η*outer(p_inf, p_gen)
    storage = MemoryStorage(config)

    # AttractorDynamics implements iterative retrieval with hierarchical masking
    # It implements: p[t+1] = κ*p[t] + M^T@p[t] * mask[t]
    attractor = AttractorDynamics(config)

    # =========================================================================
    # PHASE 2: Train Memory Through Hebbian Learning
    # =========================================================================
    # Simulate spatial navigation by generating random place cell patterns
    # In full TEM: p = g ⊗ x (grid cells ⊗ compressed sensory input)
    n_p_total = sum(config.n_p_calculated)

    # Track learning progress over training
    memory_strengths = []  # Frobenius norm of M_gen (overall connection strength)
    cosine_sims = []  # Similarity between M_gen and M_inf (dual memory divergence)

    for step in range(config.n_training_steps):
        # Generate random grounded location patterns (batch_size samples)
        # softmax ensures valid probability distributions (sum to 1, non-negative)
        # In real TEM: p_inferred comes from sensory→location inference
        p_inferred = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # In real TEM: p_generated comes from abstract→location prediction
        p_generated = torch.randn(config.batch_size, n_p_total).softmax(dim=1)

        # Apply Hebbian learning rule: strengthen connections between co-active patterns
        # M_new = λ*M_old + η*outer(p_inf, p_gen)
        # - λ (lambda): forgetting rate (decay old memories)
        # - η (eta): learning rate (strength of new associations)
        storage.update(p_inferred, p_generated, eta=config.eta, lamb=config.lambda_)

        # Monitor memory strength (how much information is stored)
        # Frobenius norm = sqrt(sum of squared weights)
        m_gen_strength = torch.norm(storage.M_gen).item()
        memory_strengths.append(m_gen_strength)

        # Monitor divergence between dual memories (should stay similar if properly tuned)
        # Cosine similarity = 1.0 means identical, 0.0 means orthogonal
        m_gen_flat = storage.M_gen.flatten()
        m_inf_flat = storage.M_inf.flatten()
        cosine_sim = torch.nn.functional.cosine_similarity(m_gen_flat, m_inf_flat, dim=0).item()
        cosine_sims.append(cosine_sim)

    # =========================================================================
    # PHASE 3: Test Attractor Retrieval Quality
    # =========================================================================
    # Evaluate memory recall: can noisy/partial queries be corrected via attractor dynamics?

    # Create clean target patterns from the learned distribution
    # These represent "ground truth" locations we want to retrieve
    test_targets = torch.randn(config.n_test_queries, n_p_total).softmax(dim=1)

    # Corrupt targets with Gaussian noise to simulate partial/uncertain observations
    # This tests the memory's ability to "clean up" noisy inputs
    noise = torch.randn_like(test_targets) * config.noise_level
    test_queries = test_targets + noise
    test_queries = test_queries.clamp(min=0)  # Ensure valid activations (no negatives)

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
    for i in range(config.n_test_queries):
        query_error = torch.nn.functional.mse_loss(test_queries[i], test_targets[i]).item()
        retrieval_error = torch.nn.functional.mse_loss(test_retrievals[i], test_targets[i]).item()
        improvement = ((query_error - retrieval_error) / query_error) * 100  # Percentage improvement

    # =========================================================================
    # PHASE 4: Robustness Analysis Across Noise Levels
    # =========================================================================
    # Test how well memory performs under varying degrees of corruption
    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]  # From slight to severe noise
    errors_by_mode = {"Inference": [], "Generative": []}  # Compare dual memories

    for noise_level in noise_levels:
        # Create noisy queries at this noise level
        noise = torch.randn_like(test_targets) * noise_level
        noisy_queries = (test_targets + noise).clamp(min=0)

        # Test inference memory (used for sensory→location inference)
        retrieved_inf = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=True), for_inference=True)
        error_inf = torch.nn.functional.mse_loss(retrieved_inf, test_targets).item()
        errors_by_mode["Inference"].append(error_inf)

        # Test generative memory (used for abstract→location prediction)
        retrieved_gen = attractor.retrieve(noisy_queries, storage.get_memory(for_inference=False), for_inference=False)
        error_gen = torch.nn.functional.mse_loss(retrieved_gen, test_targets).item()
        errors_by_mode["Generative"].append(error_gen)

    # =========================================================================
    # PHASE 5: Generate Visualizations
    # =========================================================================
    # Create comprehensive figures to understand memory structure and dynamics

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
        fig1.savefig(config.output_dir / "01_memory_matrices.png", dpi=150, bbox_inches="tight")

    # Plot 2: Learning dynamics over training
    # Track how memory strength grows and dual memories evolve
    fig2 = figures.plot_learning_curve(
        memory_strengths,  # Frobenius norm trajectory
        cosine_sims if storage.use_dual_memory else None,  # Dual memory similarity
    )
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_learning_curve.png", dpi=150, bbox_inches="tight")

    # Plot 3: Hierarchical retrieval masks
    # Visualize progressive unmasking schedule (coarse→fine)
    fig3 = figures.plot_hierarchical_masks(
        attractor.p_retrieve_mask_inf,  # Binary masks for each iteration
        n_p_per_freq=config.n_p_calculated,  # Frequency boundaries
    )
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_hierarchical_masks.png", dpi=150, bbox_inches="tight")

    # Plot 4: Attractor convergence trajectories
    # Show how noisy queries are iteratively refined toward targets
    fig4 = figures.plot_attractor_convergence(
        queries_list,  # Initial noisy patterns
        retrievals_list,  # Final retrieved patterns
        targets_list,  # Ground truth patterns
    )
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_attractor_convergence.png", dpi=150, bbox_inches="tight")

    # Plot 5: Robustness to noise
    # Compare inference vs generative memory across noise levels
    fig5 = figures.plot_retrieval_quality(
        errors_by_mode,  # MSE for each mode at each noise level
        noise_levels,  # X-axis values
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_retrieval_quality.png", dpi=150, bbox_inches="tight")

    # Display plots interactively or just save them
    if config.show_plots:
        plt.show()  # Blocks until user closes windows
    else:
        plt.close("all")  # Clean up memory
