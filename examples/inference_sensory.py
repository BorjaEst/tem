#!/usr/bin/env python3
"""Sensory processing example demonstrating temporal filtering and normalization.

This example demonstrates the torch_tem.inference.SensoryProcessor capabilities:
- Exponential temporal smoothing across multiple frequency channels
- Learnable normalization with affine transforms
- Memory effects and frequency-dependent filtering
- Integration with synthetic observation sequences
- Visualization of filtered outputs across time

The SensoryProcessor applies frequency-specific exponential moving averages to
sensory inputs, creating multiple temporally-filtered views that help with
credit assignment and temporal stability in downstream processing.

Usage Examples:
---------------
    # Default: 100 timesteps, 3 frequencies, save plots
    python examples/inference_sensory.py

    # Longer walk with different architecture
    python examples/inference_sensory.py --walk_length 200

    # Different grid size and observation mode
    python examples/inference_sensory.py --grid_size 7 --observation_mode tiled

    # Show plots interactively
    python examples/inference_sensory.py --show_plots true --save_plots false

    # Full help
    python examples/inference_sensory.py --help

Outputs:
--------
When save_plots=true, generates 4 visualizations in outputs/inference_sensory/:
    1. 01_frequency_bank.png - Frequency configuration
    2. 02_temporal_filtering.png - Temporal filtering heatmaps
    3. 03_frequency_comparison.png - Single feature comparison
    4. 04_normalization_effects.png - Normalization effects
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.config import ArchitectureConfig
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.inference.sensory import SensoryProcessor


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for sensory processing example.

    Uses ArchitectureConfig to initialize components, matching the pattern from
    the main inference.py example.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_sensory")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference_sensory"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @computed_field(description="Number of observation neurons (grid_size^2)")
    @property
    def n_x(self) -> int:
        return self.grid_size * self.grid_size


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the sensory processing experiment with visualizations."""
    config = ExampleConfig()

    # Create model config using ArchitectureConfig
    model_config = ArchitectureConfig(n_x=config.n_x, n_x_c=config.n_x_c, f_initial=config.f_initial)

    # Compute connectivity matrices from model config
    two_hot_table = utils.create_two_hot_table(model_config.n_x, model_config.n_x_c)

    print("=" * 80)
    print("Sensory Processing Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_x={model_config.n_x}, n_x_c={model_config.n_x_c}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk Generation
    # =========================================================================
    print("Phase 1: Generating walk trajectory...")
    env = data.Environment.from_grid(config.grid_size, config.grid_size, config.observation_mode)
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
    # PHASE 2: Initialize Sensory Processing Components
    # =========================================================================
    print("Phase 2: Initializing sensory processing components...")

    # Sensory processing
    encoder = SensoryEncoder(model_config, two_hot_table)
    processor = SensoryProcessor(model_config)
    print(f"  ✓ SensoryEncoder: {model_config.n_x} → {model_config.n_x_c} (two-hot)")
    print(f"  ✓ SensoryProcessor: {model_config.n_f} frequency channels")
    print()

    # =========================================================================
    # PHASE 3: Run Sensory Processing Pipeline
    # =========================================================================
    print("Phase 3: Running sensory processing pipeline...")
    x_c_history = []
    x_f_history = []
    x_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # [n_x] → [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Step 2: Temporal filtering → multi-frequency representation
        x_f = processor(x_c, x_prev)  # List[n_f] of [1, n_x_c]

        # Store history (extract batch dimension for single-trajectory storage)
        x_c_history.append(x_c[0])
        x_f_history.append([x[0] for x in x_f])

        x_prev = x_f

    print(f"  ✓ Processed {config.walk_length} timesteps through sensory pipeline")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Generate demo data for normalization comparison
    observations_stacked = torch.stack(observations)  # [T, n_x]
    x_c_demo = encoder(observations_stacked[config.walk_length // 2 : config.walk_length // 2 + 5])  # [5, n_x_c]
    x_prev_demo = [torch.zeros(5, model_config.n_x_c) for _ in range(model_config.n_f)]
    x_f_raw = processor.filter_temporal(x_c_demo, x_prev_demo)  # Raw filtering (before normalization)
    x_f_normalized = processor(x_c_demo, x_prev_demo)  # Full processing (with normalization)

    # Plot 1: Frequency bank configuration
    fig1 = figures.plot_frequency_bank(model_config.f_initial_extended)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_frequency_bank.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_frequency_bank.png")

    # Plot 2: Temporal filtering across all frequencies
    fig2 = figures.plot_temporal_filtering(x_c_history, x_f_history, model_config.f_initial_extended)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_temporal_filtering.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_temporal_filtering.png")

    # Plot 3: Single feature comparison
    fig3 = figures.plot_frequency_comparison(x_c_history, x_f_history, model_config.f_initial_extended, feature_idx=0)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_frequency_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_frequency_comparison.png")

    # Plot 4: Normalization effects (single timestep)
    fig4 = figures.plot_normalization_effects(x_f_raw, x_f_normalized, model_config.f_initial_extended)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_normalization_effects.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_normalization_effects.png")

    print()
    print("=" * 80)
    print("Pipeline Summary:")
    print("=" * 80)
    print(f"Input:  {model_config.n_x}-dim observations ({config.observation_mode} mode)")
    print(f"  ↓ SensoryEncoder (two-hot)")
    print(f"Stage 1: {model_config.n_x_c}-dim compressed sensory (x_c)")
    print(f"  ↓ SensoryProcessor ({model_config.n_f} frequencies)")
    print(f"Stage 2: Multi-frequency filtered sensory (x_f)")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
