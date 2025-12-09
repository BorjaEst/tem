#!/usr/bin/env python3
"""Grounded location inference example demonstrating outer product computation.

This example demonstrates the torch_tem.inference.GroundedLocInference capabilities:
- Outer product computation: p = g ⊗ x (hippocampal place cells from grid cells + sensory)
- Integration with sensory encoder and processor (full inference pipeline)
- Abstract location generation via synthetic grid cell patterns
- Visualization of conjunctive coding and place field emergence
- Multi-frequency hierarchical processing

The grounded location inference creates hippocampal-like place cell representations
by binding abstract spatial location (grid cells) with sensory context via outer product.

Pipeline Stages:
----------------
1. Sensory Processing: Observation encoding and temporal filtering
   - SensoryEncoder: Two-hot encoding (n_x → n_x_c)
   - SensoryProcessor: Multi-frequency temporal filtering

2. Abstract Location: Synthetic grid cell patterns (simulating entorhinal cortex)
   - OscillatoryGridGenerator: Generate grid cell activity patterns

3. Grounded Location Inference: Final place cells from abstract location
   - ProjectionHead: Transform and downsample g
   - GroundedLocInference: p = g ⊗ x_f (outer product)

Usage Examples:
---------------
    # Default: 100 timesteps, save plots
    python examples/inference_grounded.py

    # Longer walk with different architecture
    python examples/inference_grounded.py --walk_length 200

    # Different grid size and observation mode
    python examples/inference_grounded.py --grid_size 7 --observation_mode tiled

    # Show plots interactively
    python examples/inference_grounded.py --show_plots true --save_plots false

    # Full help
    python examples/inference_grounded.py --help

Outputs:
--------
When save_plots=true, generates 6 visualizations in outputs/inference_grounded/:
    1. 01_environment.png - Grid layout
    2. 02_walk_trajectory.png - Agent trajectory
    3. 03_sensory_processing.png - Temporal filtering heatmaps
    4. 04_place_cell_activity.png - Place field evolution
    5. 05_outer_product_structure.png - Decomposition at mid-point
    6. 06_place_cell_dynamics.png - Multi-frequency dynamics
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.config import EnvironmentConfig, ModelConfig
from torch_tem.core.projection import ProjectionHead
from torch_tem.inference.grounded import GroundedLocInference
from torch_tem.inference.sensory import SensoryEncoder, SensoryProcessor, SensoryProjection


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for grounded location inference example.

    Simplified configuration that creates proper config objects for component instantiation.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_grounded")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    @computed_field(description="Number of observation neurons (grid_size^2)")
    @property
    def n_x(self) -> int:
        return self.grid_size * self.grid_size

    # Memory configuration (used for InferenceConfig)
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference_grounded"), description="Directory for saving plots")
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
    """Run the grounded location inference experiment with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    environment_config = EnvironmentConfig(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
    model_config = ModelConfig(n_x=config.n_x, n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial, eta=config.eta, kappa=config.kappa)

    # Compute connectivity matrices from model config
    two_hot_table = utils.create_two_hot_table(model_config.n_x, model_config.n_x_c)
    g_downsampled = utils.create_g_downsample(model_config.n_g, model_config.n_g_subsampled_combined)
    W_repeat = utils.create_W_repeat(model_config.n_g_subsampled_combined, model_config.n_x_f)
    W_tile = utils.create_W_tile(model_config.n_g_subsampled_combined, model_config.n_x_f)

    print("=" * 80)
    print("Grounded Location Inference Example")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
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
    tiling = SensoryProjection(model_config, W_tile)
    print(f"  ✓ SensoryEncoder: {model_config.n_x} → {model_config.n_x_c} (two-hot)")
    print(f"  ✓ SensoryProcessor: {model_config.n_f} frequency channels")
    print(f"  ✓ SensoryProjection: x_f → x_ (W_tile expansion + w_p gating)")

    # Grounded location inference
    projection = ProjectionHead(model_config, g_downsampled, W_repeat)
    grounded = GroundedLocInference(model_config)
    print(f"  ✓ ProjectionHead: Laplacian transform + downsampling")
    print(f"  ✓ GroundedLocInference: g_ ⊙ x_ → p (element-wise product)")
    print()

    # =========================================================================
    # PHASE 3: Generate Synthetic Grid Cell Patterns
    # =========================================================================
    print("Phase 3: Generating synthetic grid cell patterns...")
    grid_generator = data.OscillatoryGridGenerator(model_config, config.walk_length, batch_size=1)
    transition_history = grid_generator.generate()  # List[T] of Transition (g, sigma)
    print(f"  ✓ Generated {config.walk_length} timesteps of grid cell activity")
    print()

    # =========================================================================
    # PHASE 4: Run Grounded Inference Pipeline
    # =========================================================================
    print("Phase 4: Running grounded inference pipeline...")

    x_c_history = []
    x_f_history = []
    p_history = []
    x_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # [n_x] → [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Step 2: Temporal filtering → multi-frequency representation
        x_f = processor(x_c, x_prev)  # List[n_f] of [1, n_x_c]

        # Step 3: Expand sensory to place dimensions (W_tile + w_p gating)
        x_expanded = tiling(x_f)  # List[n_f] of [1, n_p[f]]

        # Step 4: Get synthetic grid cells at time t (simulating abstract location)
        transition = transition_history[t]  # Transition with mean and uncertainty
        g_t = transition.mean  # Extract mean abstract location

        # Step 5: Expand grid cells to place dimensions (transform + downsample + W_repeat)
        g_expanded = projection(g_t)  # List[n_f] of [1, n_p[f]]

        # Step 6: Compute grounded location via element-wise product
        p_t = grounded(g_expanded, x_expanded)  # List[n_f] of [1, n_p[f]]

        # Store history (extract batch dimension for single-trajectory storage)
        x_c_history.append(x_c[0])
        x_f_history.append([x[0] for x in x_f])
        p_history.append([p[0] for p in p_t])
        x_prev = x_f

    print(f"  ✓ Processed {config.walk_length} timesteps through grounded inference")
    print()

    # =========================================================================
    # PHASE 5: Generate Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")

    # Plot 1 & 2: Environment and walk trajectory
    fig1 = figures.plot_environment_layout(env, title=f"Environment: {config.grid_size}×{config.grid_size} Grid")
    fig2 = figures.plot_walks(env, [walk], title=f"Walk Trajectory ({config.walk_length} steps)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_environment.png, 02_walk_trajectory.png")

    # Plot 3: Sensory processing (temporal filtering)
    fig3 = figures.plot_temporal_filtering(x_c_history, x_f_history, model_config.f_extended)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_sensory_processing.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_sensory_processing.png")

    # Plot 4: Grounded location activity (place cells)
    fig4 = figures.plot_grounded_location_activity(p_history, observations, locations, model_config.f_extended, model_config.n_p)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_place_cell_activity.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_place_cell_activity.png")

    # Plot 5: Outer product structure (mid-point)
    mid_point = config.walk_length // 2
    transition_mid = transition_history[mid_point]  # Get Transition at mid-point
    g_mid = transition_mid.mean  # Extract mean abstract location
    g_mid_downsampled = projection.downsample(g_mid)  # Just downsample for plotting
    g_sample = [g_mid_downsampled[f][0] for f in range(model_config.n_f)]
    fig5 = figures.plot_outer_product_structure(g_sample, x_f_history[mid_point], p_history[mid_point], model_config.f_extended)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_outer_product_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_outer_product_structure.png")

    # Plot 6: Place cell dynamics across frequencies
    fig6 = figures.plot_place_cell_dynamics(p_history, observations, model_config.f_extended, model_config.n_p)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_place_cell_dynamics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_place_cell_dynamics.png")

    print()
    print("=" * 80)
    print("Pipeline Summary:")
    print("=" * 80)
    print(f"Input:  {model_config.n_x}-dim observations ({config.observation_mode} mode)")
    print(f"  ↓ SensoryEncoder (two-hot)")
    print(f"Stage 1: {model_config.n_x_c}-dim compressed sensory (x_c)")
    print(f"  ↓ SensoryProcessor ({model_config.n_f} frequencies)")
    print(f"Stage 2: Multi-frequency filtered sensory (x_f) - {model_config.n_x_c} per freq")
    print(f"  ↓ SensoryProjection (W_tile expansion + w_p gating)")
    print(f"Stage 3: Expanded sensory (x_) - {model_config.n_p}")
    print(f"  ↓ Synthetic grid cell generation")
    print(f"Stage 4: Abstract location (g) - {model_config.n_g}")
    print(f"  ↓ ProjectionHead (transform + downsample + W_repeat expansion)")
    print(f"Stage 5: Expanded abstract (g_) - {model_config.n_p}")
    print(f"  ↓ GroundedLocInference (g_ ⊙ x_ element-wise product)")
    print(f"Output: Final grounded location (p) - {model_config.n_p}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
