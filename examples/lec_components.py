#!/usr/bin/env python3
"""LEC sensory processing example demonstrating encoding, temporal filtering, and normalization.

This example demonstrates the Lateral Entorhinal Cortex (LEC) components from torch_tem.lec:
- Encoder: Two-hot sensory compression from high-dimensional observations
- Processor: Multi-frequency temporal filtering with learnable decay rates
- Exponential moving average smoothing across multiple frequency channels
- L2 normalization with mean-centering per frequency channel
- Visualization of sensory processing pipeline across time

The LEC pathway implements the first stage of sensory processing in TEM:
  Observations (x) → Encoder → Compressed sensory (x_c) → Processor → Filtered sensory (x_f)

The Encoder compresses high-dimensional observations using a two-hot encoding scheme,
while the Processor applies frequency-specific exponential smoothing to create
multiple temporally-filtered views that help with credit assignment and temporal
stability in downstream hippocampal processing.

Architecture:
-------------
    LEC Encoder (lec.sensory.Encoder):
        - Input: One-hot observations [B, n_x]
        - Output: Two-hot compressed sensory [B, n_x_c]
        - Method: Lookup table mapping observation index to two-hot code

    LEC Processor (lec.processor.Processor):
        - Input: Compressed sensory [B, n_x_c]
        - Output: Multi-frequency filtered sensory List[n_f] of [B, n_x_c]
        - Method: Per-frequency exponential smoothing + L2 normalization
        - Learnable parameters: Decay rates (alpha) for each frequency channel

Usage Examples:
---------------
    # Default: 5×5 grid, 100 timesteps, 3 frequencies, save plots
    python examples/lec_components.py

    # Longer walk with more frequencies
    python examples/lec_components.py --walk_length 200

    # Different grid size and observation mode
    python examples/lec_components.py --grid_size 7 --observation_mode tiled

    # Custom frequency configuration
    python examples/lec_components.py --f_initial "[0.95, 0.7, 0.4, 0.15]"

    # Show plots interactively
    python examples/lec_components.py --show_plots true --save_plots false

    # Full help
    python examples/lec_components.py --help

Outputs:
--------
When save_plots=true, generates 4 visualizations in outputs/inference_sensory/:
    1. 01_frequency_bank.png - Frequency channel configuration
    2. 02_temporal_filtering.png - Temporal filtering heatmaps across all frequencies
    3. 03_frequency_comparison.png - Single feature comparison across frequencies
    4. 04_normalization_effects.png - Before/after normalization effects
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, lec, utils
from torch_tem.config import EnvironmentConfig, ModelConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for LEC sensory processing example.

    Defines environment setup, walk generation, and architecture parameters
    for the LEC (Lateral Entorhinal Cortex) sensory processing pipeline.
    Uses ModelConfig to initialize Encoder and Processor components.

    Attributes:
        grid_size: Size of the square grid environment (grid_size × grid_size)
        observation_mode: How observations are generated ("unique", "tiled", "random")
        walk_length: Number of timesteps in the generated trajectory
        f_initial: Base frequency values for each temporal filtering module
        n_g_subsampled: Number of grid cells per frequency module (unused in this example)
        n_x_c: Dimension of compressed sensory representation (two-hot encoding)
        output_dir: Directory for saving visualization outputs
        show_plots: Whether to display plots interactively
        save_plots: Whether to save plots to disk
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_sensory")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [10, 8, 6], description="Subsampled grid cells per frequency module")
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
    """Run the LEC sensory processing pipeline experiment with visualizations.

    This script demonstrates the complete sensory processing pathway:
      1. Generate a random walk trajectory in a grid environment
      2. Initialize LEC Encoder and Processor components
      3. Process observations through the sensory pipeline
      4. Generate visualizations of filtering and normalization effects
    """
    config = ExampleConfig()

    # Create model config using ModelConfig
    environment_config = EnvironmentConfig(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
    model_config = ModelConfig(n_x=config.n_x, n_x_c=config.n_x_c, f_initial=config.f_initial, n_g_subsampled=config.n_g_subsampled)

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
    # PHASE 2: Initialize Sensory Processing Components
    # =========================================================================
    print("Phase 2: Initializing LEC components...")

    # LEC Encoder: Compresses observations using two-hot encoding
    # Maps one-hot observations [n_x] to compressed representation [n_x_c]
    # Uses a lookup table computed from model_config.two_hot_table property
    encoder = lec.sensory.Encoder(model_config)

    # LEC Processor: Applies multi-frequency temporal filtering
    # Maintains n_f parallel exponential moving averages with learnable decay rates
    # Each frequency channel has its own alpha parameter (decay rate)
    processor = lec.processor.Processor(model_config)

    print(f"  ✓ Encoder: {model_config.n_x} → {model_config.n_x_c} (two-hot compression)")
    print(f"  ✓ Processor: {model_config.n_f} frequency channels (f = {model_config.f_initial})")
    print()

    # =========================================================================
    # PHASE 3: Run Sensory Processing Pipeline
    # =========================================================================
    print("Phase 3: Running sensory processing pipeline...")

    # Initialize history storage for visualization
    x_c_history = []  # Compressed sensory over time
    x_f_history = []  # Multi-frequency filtered sensory over time

    # Initialize previous filtered state (zero for first timestep)
    x_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Step 1: Encode observation → compressed sensory
        # Encoder maps one-hot observation to two-hot compressed representation
        x_t = observations[t].unsqueeze(0)  # Add batch dimension: [n_x] → [1, n_x]
        x_c = encoder(x_t)  # Two-hot lookup: [1, n_x] → [1, n_x_c]

        # Step 2: Temporal filtering → multi-frequency representation
        # Processor applies exponential smoothing at each frequency, then normalizes
        # x_f[f] = normalize(alpha[f] * x_c + (1 - alpha[f]) * x_prev[f])
        x_f = processor(x_c, x_prev)  # [1, n_x_c] → List[n_f] of [1, n_x_c]

        # Store history (remove batch dimension for single-trajectory storage)
        x_c_history.append(x_c[0])  # [1, n_x_c] → [n_x_c]
        x_f_history.append([x[0] for x in x_f])  # List[n_f] of [1, n_x_c] → List[n_f] of [n_x_c]

        # Update previous state for next timestep
        x_prev = x_f

    print(f"  ✓ Processed {config.walk_length} timesteps through sensory pipeline")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Generate demo data for normalization comparison (5 consecutive timesteps)
    observations_stacked = torch.stack(observations)  # List[T] of [n_x] → [T, n_x]
    midpoint = config.walk_length // 2
    x_c_demo = encoder(observations_stacked[midpoint : midpoint + 5])  # [5, n_x_c]
    x_prev_demo = [torch.zeros(5, model_config.n_x_c) for _ in range(model_config.n_f)]

    # Compare raw filtering vs normalized filtering
    x_f_raw = processor.filter_temporal(x_c_demo, x_prev_demo)  # Raw exponential smoothing only
    x_f_normalized = processor(x_c_demo, x_prev_demo)  # Full pipeline (smoothing + normalization)

    # Plot 1: Frequency bank configuration
    fig1 = figures.plot_frequency_bank(model_config.f_extended)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_frequency_bank.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_frequency_bank.png")

    # Plot 2: Temporal filtering across all frequencies
    fig2 = figures.plot_temporal_filtering(x_c_history, x_f_history, model_config.f_extended)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_temporal_filtering.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_temporal_filtering.png")

    # Plot 3: Single feature comparison
    fig3 = figures.plot_frequency_comparison(x_c_history, x_f_history, model_config.f_extended, feature_idx=0)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_frequency_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_frequency_comparison.png")

    # Plot 4: Normalization effects (single timestep)
    fig4 = figures.plot_normalization_effects(x_f_raw, x_f_normalized, model_config.f_extended)
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
