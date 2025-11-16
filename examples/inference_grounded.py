#!/usr/bin/env python3
"""Grounded location inference example demonstrating outer product computation.

This example demonstrates the torch_tem.inference.GroundedLocationInference capabilities:
- Outer product computation: p = g ⊗ x (hippocampal place cells from grid cells + sensory)
- Integration with sensory encoder and processor (full inference pipeline)
- Abstract location generation via synthetic grid cell patterns
- Visualization of conjunctive coding and place field emergence
- Multi-frequency hierarchical processing

The grounded location inference creates hippocampal-like place cell representations
by binding abstract spatial location (grid cells) with sensory context via outer product.

Usage:
    python examples/inference_grounded.py --walk-length 50 --n-frequencies 3
    python examples/inference_grounded.py --grid-size 6 --show-plots
    python examples/inference_grounded.py --help
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
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.core.projection import ProjectionHead
from torch_tem.inference.grounded import GroundedLocationInference
from torch_tem.inference.sensory import SensoryProcessor


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for grounded location inference example.

    This config implements EncoderParams, SensoryProcessorParams, GroundedInferenceParams,
    and ProjectionParams protocols for direct component instantiation.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_grounded")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of frequency modules")
    n_g_subsampled_per_freq: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    # Frequency configuration
    f_min: float = Field(default=0.1, ge=0.01, le=0.5, description="Minimum frequency (longest memory)")
    f_max: float = Field(default=0.9, ge=0.5, le=1.0, description="Maximum frequency (shortest memory)")

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

    @field_validator("n_g_subsampled_per_freq")
    @classmethod
    def validate_frequencies_match(cls, v: List[int], info) -> List[int]:
        """Ensure n_g_subsampled matches n_frequencies."""
        n_freq = info.data.get("n_frequencies", 3)
        if len(v) != n_freq:
            # Auto-generate if mismatch
            return [12 - 2 * i for i in range(n_freq)]
        return v

    @computed_field(description="Number of unique observations in the environment")
    @property
    def n_x(self) -> int:
        n_locations = self.grid_size * self.grid_size
        if self.observation_mode == "unique":
            return n_locations
        elif self.observation_mode == "tiled":
            return 4
        elif self.observation_mode == "random":
            return max(4, n_locations // 4)
        raise ValueError(f"Invalid observation_mode: {self.observation_mode}")

    # ==============================================================================
    # Protocol Implementations
    # ==============================================================================

    @computed_field(description="Total number of frequency modules")
    @property
    def n_f_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Full grid cell dimensions (3x subsampled)")
    @property
    def n_g_calculated(self) -> List[int]:
        return [3 * g for g in self.n_g_subsampled_per_freq]

    @computed_field(description="Place cell dimensions per frequency")
    @property
    def n_p_calculated(self) -> List[int]:
        return [g * self.n_x_c for g in self.n_g_subsampled_per_freq]

    @computed_field(description="Sensory dimensions per frequency")
    @property
    def n_x_f_calculated(self) -> List[int]:
        return [self.n_x_c for _ in range(self.n_frequencies)]

    @computed_field(description="Frequency values (logarithmic spacing)")
    @property
    def f_initial_extended(self) -> List[float]:
        if self.n_frequencies == 1:
            return [self.f_max]
        return np.logspace(np.log10(self.f_min), np.log10(self.f_max), self.n_frequencies).tolist()

    @computed_field(description="Two-hot encoding table")
    @property
    def two_hot_table_calculated(self) -> List[Tensor]:
        return utils.create_two_hot_table(self.n_x, self.n_x_c)

    @computed_field(description="Kronecker repeat matrices for outer product")
    @property
    def W_repeat_calculated(self) -> List[Tensor]:
        return utils.create_W_repeat(self.n_g_subsampled_per_freq, [self.n_x_c] * self.n_frequencies)

    @computed_field(description="Kronecker tile matrices for outer product")
    @property
    def W_tile_calculated(self) -> List[Tensor]:
        return utils.create_W_tile(self.n_g_subsampled_per_freq, [self.n_x_c] * self.n_frequencies)

    @computed_field(description="Downsampling matrices for grid cells")
    @property
    def g_downsample_calculated(self) -> List[Tensor]:
        return utils.create_g_downsample(self.n_g_calculated, self.n_g_subsampled_per_freq)


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the grounded location inference experiment with visualizations."""
    config = ExampleConfig()

    # Create environment and generate walk
    env = data.Environment.from_grid(config.grid_size, config.grid_size, config.observation_mode)
    env.validate()

    policy_gen = data.PolicyGenerator(env)
    policy = policy_gen.random_policy()

    walk_gen = data.WalkGenerator(env)
    walks = walk_gen.generate_walks(n_walks=1, walk_length=config.walk_length, policy=policy)
    walk = walks[0]

    # Extract observations and locations
    observations = torch.stack([obs.clone().detach() for obs in walk.observations])  # [T, n_x]
    locations = torch.as_tensor(walk.locations, dtype=torch.long)  # [T]

    # Initialize components using config (implements all required protocols)
    encoder = SensoryEncoder(config)
    processor = SensoryProcessor(config)
    projection = ProjectionHead(config)
    grounded = GroundedLocationInference(config)

    # Generate synthetic grid cell patterns (simulating abstract location)
    grid_generator = data.SyntheticGridGenerator(config, walk_length=config.walk_length, batch_size=1)
    g_history = grid_generator.generate()  # List[T] of List[n_f] of [B, n_g[f]]

    # Process observations through full inference pipeline
    x_c_history = []
    x_f_history = []
    p_history = []
    x_prev = [torch.zeros(1, config.n_x_c) for _ in range(config.n_frequencies)]

    for t in range(config.walk_length):
        # 1. Encode observation
        x_t = observations[t : t + 1]  # [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # 2. Temporal filtering and normalization
        x_f = processor(x_c, x_prev)  # [n_f] of [1, n_x_c]

        # 3. Get grid cells at time t (now directly available!)
        g_t = g_history[t]  # List[n_f] of [B, n_g[f]]

        # 4. Transform and downsample grid cells
        g_transformed = projection.transform(g_t)
        g_downsampled = projection.downsample(g_transformed)

        # 5. Compute grounded location via outer product
        p_t = grounded(g_downsampled, x_f)  # [n_f] of [1, n_p[f]]

        # Store history
        x_c_history.append(x_c)
        x_f_history.append(x_f)
        p_history.append(p_t)
        x_prev = x_f

    mid_point = config.walk_length // 2
    g_mid = g_history[mid_point]  # List[n_f] of [B, n_g[f]]
    g_mid_transformed = projection.transform(g_mid)
    g_mid_downsampled = projection.downsample(g_mid_transformed)

    # Plot 1: Grounded location activity over time
    fig1 = figures.plot_grounded_location_activity(p_history, observations, locations, config.f_initial_extended, config.n_p_calculated)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_place_cell_activity.png", dpi=150, bbox_inches="tight")

    # Plot 2: Outer product structure at specific timepoint
    fig2 = figures.plot_outer_product_structure(g_mid_downsampled, x_f_history[mid_point], p_history[mid_point], config.f_initial_extended)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_outer_product_structure.png", dpi=150, bbox_inches="tight")

    # Plot 3: Place cell dynamics across frequencies
    fig3 = figures.plot_place_cell_dynamics(p_history, observations, config.f_initial_extended, config.n_p_calculated)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_place_cell_dynamics.png", dpi=150, bbox_inches="tight")

    # Summary statistics
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
