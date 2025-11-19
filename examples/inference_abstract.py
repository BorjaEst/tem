#!/usr/bin/env python3
"""Abstract location inference example demonstrating precision-weighted fusion.

This example demonstrates the torch_tem.inference.AbstractLocationInference capabilities:
- Precision-weighted fusion of multiple information sources
- Transition-based prediction (g_gen) with uncertainty
- Memory-based inference (p_x → g_mem) via learned projections
- Salient object ("shiny") signals integration (optional)
- Scheduled memory influence via p2g_scale_offset
- Uncertainty estimation from memory quality indicators
- Source contribution analysis and visualization

Pipeline Stages:
----------------
1. Generate synthetic walk trajectory in environment
2. Process observations through sensory encoder and temporal filtering
3. Project filtered sensory to hippocampal space
4. Retrieve patterns from memory via attractor dynamics
5. Infer abstract location by fusing memory and generative paths
6. Update memory with Hebbian learning
7. Visualize precision-weighted fusion and source contributions

Data Flow:
----------
    x (observation)
    → x_c (compressed/two-hot encoding)
    → x_f (temporal filtering per frequency)
    → ~x_t (sensory projected to p-space via W_tile)
    → p_x (hippocampal retrieval from sensory: M^T @ ~x_t)
    → g_inf (abstract location from p_x and g_gen fusion)

Usage Examples:
---------------
    # Default: 100 timesteps, memory enabled, save plots
    python examples/inference_abstract.py

    # Disable memory path, use only transition prediction
    python examples/inference_abstract.py --use_p_inf false

    # Different grid size and walk length
    python examples/inference_abstract.py --grid_size 7 --walk_length 150

    # Show plots interactively
    python examples/inference_abstract.py --show_plots true --save_plots false

    # Full help
    python examples/inference_abstract.py --help

Outputs:
--------
When save_plots=true, generates visualizations in outputs/inference_abstract/:
    1. 01_environment.png - Grid layout
    2. 02_walk_trajectory.png - Agent trajectory
    3. 03_source_contributions.png - Precision weights over time
    4. 04_uncertainty_evolution.png - Uncertainty per source
    5. 05_g_inf_evolution.png - Abstract location inference
    6. 06_schedule_effect.png - p2g scheduling influence
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.config import ArchitectureConfig, EnvironmentConfig, InferenceConfig
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.core.projection import ProjectionHead
from torch_tem.core.tiling import SensoryProjection
from torch_tem.inference.abstract import AbstractLocationInference
from torch_tem.inference.sensory import SensoryProcessor
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for abstract location inference example.

    This config implements all inference-related protocols for component
    instantiation.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_abstract")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    @computed_field(description="Number of sensory dimensions")
    @property
    def n_x(self) -> int:
        return self.grid_size * self.grid_size

    # Source configuration
    use_p_inf: bool = Field(default=True, description="Enable memory-based inference path")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Scheduling
    p2g_schedule_start: float = Field(default=2.0, ge=0.0, le=5.0, description="Initial p2g scale offset (high = low memory influence)")
    p2g_schedule_end: float = Field(default=0.5, ge=0.0, le=2.0, description="Final p2g scale offset (low = high memory influence)")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference_abstract"), description="Directory for saving plots")
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
    """Run the abstract location inference experiment with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    environment_config = EnvironmentConfig(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
    inference_config = InferenceConfig(eta=config.eta, kappa=config.kappa, use_p_inf=config.use_p_inf)
    model_config = ArchitectureConfig(n_x=config.n_x, n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial)

    # Compute connectivity matrices from model config
    two_hot_table = utils.create_two_hot_table(model_config.n_x, model_config.n_x_c)
    g_downsampled = utils.create_g_downsample(model_config.n_g, model_config.n_g_subsampled_combined)
    p_update_mask = utils.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f, 0, model_config.f_initial_extended)
    mask_inf, mask_gen = utils.create_p_retrieve_masks(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf, model_config.max_freq_gen)
    W_repeat = utils.create_W_repeat(model_config.n_g_subsampled_combined, model_config.n_x_f)
    W_tile = utils.create_W_tile(model_config.n_g_subsampled_combined, model_config.n_x_f)

    print("=" * 80)
    print("Abstract Location Inference: Precision-Weighted Fusion")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print(f"  Memory path (use_p_inf): {config.use_p_inf}")
    print(f"  p2g schedule: {config.p2g_schedule_start:.2f} → {config.p2g_schedule_end:.2f}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk Generation
    # =========================================================================
    print("Phase 1: Generating walk trajectory...")
    env = data.Environment(environment_config)
    env.validate()

    policy_gen = data.PolicyGenerator(env)
    policy = policy_gen.random_policy()

    walk_gen = data.WalkGenerator(env)
    walks = walk_gen.generate_walks(n_walks=1, walk_length=config.walk_length, policy=policy)
    walk = walks[0]

    observations = [obs.clone().detach() for obs in walk.observations]  # List[T] of [n_x]
    locations = torch.as_tensor(walk.locations, dtype=torch.long)  # [T]
    print(f"  ✓ Generated walk: {len(walk)} timesteps")
    print()

    # =========================================================================
    # PHASE 2: Initialize All Components
    # =========================================================================
    print("Phase 2: Initializing inference components...")

    # Sensory processing
    encoder = SensoryEncoder(model_config, two_hot_table)
    processor = SensoryProcessor(model_config)
    print(f"  ✓ SensoryEncoder: {model_config.n_x} → {model_config.n_x_c} (two-hot)")
    print(f"  ✓ SensoryProcessor: {model_config.n_f} frequency channels")

    # Sensory projection to p-space
    sensory_projection = SensoryProjection(model_config, W_tile)
    print(f"  ✓ SensoryProjection: x_f → ~x_t (W_tile transformation to p-space)")

    # Memory system
    storage = MemoryStorage(model_config, inference_config, p_update_mask)
    attractor = AttractorDynamics(model_config, inference_config, mask_inf, mask_gen)
    print(f"  ✓ MemoryStorage: {sum(model_config.n_p)}×{sum(model_config.n_p)} Hebbian matrix")
    print(f"  ✓ AttractorDynamics: {model_config.i_attractor} iterations with hierarchical masking")

    # Abstract location inference
    abstract = AbstractLocationInference(model_config, inference_config)
    print(f"  ✓ AbstractLocationInference: precision-weighted fusion")

    # Projection head for downsampling
    projection = ProjectionHead(model_config, g_downsampled)
    print(f"  ✓ ProjectionHead: Laplacian transform + downsampling")
    print()

    # =========================================================================
    # PHASE 3: Generate Synthetic Grid Cell Patterns
    # =========================================================================
    print("Phase 3: Generating synthetic grid cell patterns...")
    grid_generator = data.SyntheticGridGenerator(model_config, config.walk_length, batch_size=1)
    g_history = grid_generator.generate()  # List[T] of List[n_f] of [1, n_g[f]]
    print(f"  ✓ Generated {config.walk_length} timesteps of grid cell activity")
    print()

    # =========================================================================
    # PHASE 4: Generate p2g Schedule
    # =========================================================================
    print("Phase 4: Setting up p2g schedule...")
    p2g_schedule = np.linspace(config.p2g_schedule_start, config.p2g_schedule_end, config.walk_length)
    print(f"  ✓ p2g schedule: {p2g_schedule[0]:.2f} → {p2g_schedule[-1]:.2f}")
    print()

    # =========================================================================
    # PHASE 5: Run Abstract Inference Pipeline
    # =========================================================================
    print("Phase 5: Running abstract location inference...")

    x_c_history = []
    x_f_history = []
    x_projected_history = []  # ~x_t: sensory input projected to p-space
    p_x_history = []  # p_x: hippocampal patterns retrieved from sensory
    g_inf_history = []
    precisions_history = []
    sigma_history_dict = {"transition": [], "memory": []}

    x_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # [n_x] → [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Step 2: Temporal filtering → multi-frequency representation
        x_f = processor(x_c, x_prev)  # List[n_f] of [1, n_x_c]

        # Step 3: Project sensory to p-space (theory: ~x_t = W_tile * w_p * f_n(x_f))
        x_projected = sensory_projection(x_f)  # List[n_f] of [1, n_p[f]]
        x_proj_concat = torch.cat(x_projected, dim=1)  # [1, sum(n_p)]

        # Step 4: Retrieve hippocampal patterns from sensory input via attractor
        M_inf = storage.get_memory(for_inference=True)
        p_x_concat = attractor.retrieve(x_proj_concat, M_inf, for_inference=True)  # [1, sum(n_p)]

        # Split retrieved patterns back to per-frequency lists
        p_x = []
        start_idx = 0
        for f in range(model_config.n_f):
            end_idx = start_idx + model_config.n_p[f]
            p_x.append(p_x_concat[:, start_idx:end_idx])  # [1, n_p[f]]
            start_idx = end_idx

        # Step 5: Get synthetic grid cells at time t (for generative path)
        g_t = g_history[t]  # List[n_f] of [1, n_g[f]]

        # Step 6: Prepare inputs for abstract location inference
        # For g_mem path: project p_x to abstract location space via W_repeat
        g_mem_downsampled = [torch.matmul(p_x[f], W_repeat[f].t()) for f in range(model_config.n_f)]  # List[n_f] of [1, n_g_sub[f]]

        # Use g_t as generative prediction with moderate uncertainty
        sigma_gen = [torch.ones(1, n_g) * 0.5 for n_g in model_config.n_g]

        # Step 7: Infer abstract location via precision-weighted fusion
        offset = p2g_schedule[t]
        g_inf = abstract(g_t, sigma_gen, g_mem_downsampled if config.use_p_inf else None, shiny_signals=None, p2g_scale_offset=offset)

        # Step 8: Update memory with Hebbian learning (using p_x from retrieval)
        storage.update(p_x_concat, p_x_concat, eta=config.eta, lamb=config.lambda_)

        # Store history (extract batch dimension for single-trajectory storage)
        x_c_history.append(x_c[0])
        x_f_history.append([x[0] for x in x_f])
        x_projected_history.append([x[0] for x in x_projected])
        p_x_history.append([p[0] for p in p_x])
        g_inf_history.append(g_inf)

        # Track precisions for visualization
        precisions = {"transition": [1.0 / (sigma_gen[f] ** 2 + 1e-8) for f in range(model_config.n_f)]}
        if config.use_p_inf:
            # Approximate memory uncertainty (simplified)
            sigma_mem_approx = [torch.ones_like(g_t[f]) * (0.3 + offset) for f in range(model_config.n_f)]
            precisions["memory"] = [1.0 / (sigma_mem_approx[f] ** 2 + 1e-8) for f in range(model_config.n_f)]
            sigma_history_dict["memory"].append(sigma_mem_approx)
        else:
            sigma_history_dict["memory"].append(None)

        sigma_history_dict["transition"].append(sigma_gen)
        precisions_history.append(precisions)

        x_prev = x_f

    print(f"  ✓ Processed {config.walk_length} timesteps through abstract inference pipeline")
    print()

    # =========================================================================
    # PHASE 6: Generate Visualizations
    # =========================================================================
    print("Phase 6: Generating visualizations...")

    # =========================================================================
    # PHASE 6: Generate Visualizations
    # =========================================================================
    print("Phase 6: Generating visualizations...")

    # Plot 1 & 2: Environment and walk trajectory
    fig1 = figures.plot_environment_layout(env, title=f"Environment: {config.grid_size}×{config.grid_size} Grid")
    fig2 = figures.plot_walks(env, [walk], title=f"Walk Trajectory ({config.walk_length} steps)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_environment.png, 02_walk_trajectory.png")

    # Plot 3: Source precision contributions
    timesteps_to_plot = [0, config.walk_length // 4, config.walk_length // 2, 3 * config.walk_length // 4, config.walk_length - 1]
    fig3 = figures.plot_source_contributions(precisions_history, timesteps_to_plot, model_config.n_f)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_source_contributions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_source_contributions.png")

    # Plot 4: Uncertainty evolution
    fig4 = figures.plot_uncertainty_evolution(sigma_history_dict, model_config.n_f)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_uncertainty_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_uncertainty_evolution.png")

    # Plot 5: g_inf evolution
    fig5 = figures.plot_g_inf_evolution(g_inf_history, model_config.n_f, config.walk_length)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_g_inf_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_g_inf_evolution.png")

    # Plot 6: p2g schedule effect
    fig6 = figures.plot_schedule_effect(p2g_schedule)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_schedule_effect.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_schedule_effect.png")

    print()
    print("=" * 80)
    print("Pipeline Summary:")
    print("=" * 80)
    print(f"Input:  {model_config.n_x}-dim observations ({config.observation_mode} mode)")
    print(f"  ↓ SensoryEncoder (two-hot)")
    print(f"Stage 1: {model_config.n_x_c}-dim compressed sensory (x_c)")
    print(f"  ↓ SensoryProcessor ({model_config.n_f} frequencies)")
    print(f"Stage 2: Multi-frequency filtered sensory (x_f)")
    print(f"  ↓ SensoryProjection (~x_t = W_tile @ x_f)")
    print(f"Stage 3: Sensory input to hippocampus (~x_t) - {model_config.n_p}")
    print(f"  ↓ AttractorDynamics (M^T @ ~x_t)")
    print(f"Stage 4: Hippocampal patterns from sensory (p_x) - {model_config.n_p}")
    print(f"  ↓ Projection to abstract (p_x @ W_repeat^T)")
    print(f"Stage 5: Memory-based abstract (g_mem) - {model_config.n_g_subsampled_combined}")
    print(f"  ↓ AbstractLocationInference (precision-weighted fusion)")
    print(f"Output: Abstract location (g_inf) - {model_config.n_g}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
