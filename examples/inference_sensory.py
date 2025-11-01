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

Usage:
    python examples/inference_sensory.py --walk-length 50 --n-frequencies 4
    python examples/inference_sensory.py --grid-size 6 --show-plots
    python examples/inference_sensory.py --help
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

from torch_tem import data, figures
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.inference.sensory import SensoryProcessor


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for sensory processing example.

    This config implements both EncoderParams and SensoryProcessorParams protocols,
    allowing direct instantiation of components without intermediate helper functions.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_sensory")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "shared"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")
    policy_type: Literal["random", "distance"] = Field(default="random", description="Policy for walk generation")

    # Sensory processor configuration
    n_frequencies: int = Field(default=5, ge=2, le=10, description="Number of frequency channels")
    f_min: float = Field(default=0.1, ge=0.01, le=0.5, description="Minimum frequency (longest memory)")
    f_max: float = Field(default=0.9, ge=0.5, le=1.0, description="Maximum frequency (shortest memory)")

    # Two-hot encoding
    n_x_c: int = Field(default=10, ge=2, le=50, description="Compressed sensory dimension (two-hot)")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference_sensory"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    # Internal fields (set after environment is created)
    n_x: int = Field(default=0, description="Number of distinct observations (set dynamically)")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    # ==============================================================================
    # EncoderParams Protocol Implementation
    # ==============================================================================

    @computed_field
    @property
    def two_hot_table_calculated(self) -> List[Tensor]:
        """Table for converting one-hot to two-hot compressed representation.

        Implements EncoderParams protocol requirement.
        """
        two_hot_table = []
        for i in range(self.n_x):
            code = torch.zeros(self.n_x_c)
            idx1 = i % self.n_x_c
            idx2 = (i + 1) % self.n_x_c
            code[idx1] = 1.0
            code[idx2] = 1.0
            two_hot_table.append(code)
        return two_hot_table

    # ==============================================================================
    # SensoryProcessorParams Protocol Implementation
    # ==============================================================================

    @computed_field
    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules.

        Implements SensoryProcessorParams protocol requirement.
        """
        return self.n_frequencies

    @computed_field
    @property
    def n_x_f_calculated(self) -> List[int]:
        """Neurons for temporally filtered sensory experience x per frequency.

        Implements SensoryProcessorParams protocol requirement.
        """
        return [self.n_x_c for _ in range(self.n_frequencies)]

    @computed_field
    @property
    def f_initial_extended(self) -> List[float]:
        """Extended f_initial with frequency values (logarithmic spacing).

        Implements SensoryProcessorParams protocol requirement.
        """
        if self.n_frequencies == 1:
            return [self.f_max]
        else:
            return np.logspace(np.log10(self.f_min), np.log10(self.f_max), self.n_frequencies).tolist()


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the sensory processing experiment with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Sensory Processing Example")
    print("=" * 80)

    # 1. Create environment and generate walk
    print("\n[1/5] Generating synthetic environment and walk sequence...")
    env = data.Environment.from_grid(config.grid_size, config.grid_size, config.observation_mode)
    env.validate()
    print(f"  Environment: {env.n_locations} locations, {env.n_observations} observations")

    # Update config with actual observation count from environment
    config.n_x = env.n_observations

    policy_gen = data.PolicyGenerator(env)
    if config.policy_type == "random":
        policy = policy_gen.random_policy()
    else:
        goal = env.n_locations - 1
        policy = policy_gen.distance_policy(goal, beta=2.0)

    walk_gen = data.WalkGenerator(env)
    walks = walk_gen.generate_walks(n_walks=1, walk_length=config.walk_length, policy=policy)
    walk = walks[0]

    # Extract observations and locations
    observations = torch.stack([obs.clone().detach() for obs in walk.observations])  # [T, n_x]
    locations = torch.tensor(walk.locations, dtype=torch.long)  # [T]
    print(f"  Generated walk: {config.walk_length} steps")

    # 2. Initialize sensory encoder and processor using config (implements protocols)
    print("\n[2/5] Initializing SensoryEncoder and SensoryProcessor...")
    encoder = SensoryEncoder(config)  # config satisfies EncoderParams protocol
    processor = SensoryProcessor(config)  # config satisfies SensoryProcessorParams protocol
    print(f"  Frequencies: {config.f_initial_extended}")

    # 3. Process observations through time
    print("\n[3/5] Processing observations through temporal filter...")
    x_c_history = []
    x_f_history = []
    x_prev = [torch.zeros(1, config.n_x_c) for _ in range(config.n_frequencies)]

    for t in range(config.walk_length):
        # Encode observation to compressed sensory
        x_t = observations[t : t + 1]  # [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Apply temporal filtering and normalization
        x_f = processor(x_c, x_prev)  # List[n_f] of [1, n_x_c]

        # Store history
        x_c_history.append(x_c.squeeze(0))
        x_f_history.append([x.squeeze(0) for x in x_f])

        # Update previous state
        x_prev = x_f

    x_c_history = torch.stack(x_c_history)  # [T, n_x_c]
    print(f"  Processed {len(x_f_history)} timesteps")

    # 4. Generate visualizations
    print("\n[4/5] Generating visualizations...")

    # Plot 1: Frequency bank configuration
    fig1 = figures.plot_frequency_bank(config.f_initial_extended, title="Frequency Bank Configuration")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_frequency_bank.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_frequency_bank.png'}")

    # Plot 2: Temporal filtering across all frequencies
    fig2 = figures.plot_temporal_filtering(x_c_history, x_f_history, config.f_initial_extended, title="Temporal Filtering Across Frequencies")
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_temporal_filtering.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_temporal_filtering.png'}")

    # Plot 3: Single feature comparison
    fig3 = figures.plot_frequency_comparison(x_c_history, x_f_history, config.f_initial_extended, feature_idx=0, title="Feature 0 Across Frequencies")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_frequency_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_frequency_comparison.png'}")

    # Plot 4: Normalization effects (single timestep)
    print("\n[5/5] Demonstrating normalization effects...")
    x_c_demo = encoder(observations[config.walk_length // 2 : config.walk_length // 2 + 5])  # [5, n_x_c]
    x_prev_demo = [torch.zeros(5, config.n_x_c) for _ in range(config.n_frequencies)]

    # Raw filtering (before normalization)
    x_f_raw = processor.filter_temporal(x_c_demo, x_prev_demo)
    # Full processing (with normalization)
    x_f_normalized = processor(x_c_demo, x_prev_demo)

    fig4 = figures.plot_normalization_effects(x_f_raw, x_f_normalized, config.f_initial_extended, title="L2 Normalization Effects")
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_normalization_effects.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '04_normalization_effects.png'}")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Environment: {env.n_locations} locations, {env.n_observations} observations")
    print(f"Walk length: {config.walk_length} steps")
    print(f"Frequency channels: {config.n_frequencies}")
    print(f"Compressed dim: {config.n_x_c}")
    print(f"Frequency range: [{config.f_min:.2f}, {config.f_max:.2f}]")
    print(f"Time constants: [τ_min≈{1/config.f_max:.1f}, τ_max≈{1/config.f_min:.1f}] steps")
    print("\nKey observations:")
    print("  - Low frequencies (f→0) preserve long-term history (high τ)")
    print("  - High frequencies (f→1) track current input closely (low τ)")
    print("  - L2 normalization stabilizes scale across channels")
    print("  - Multi-frequency representation aids temporal credit assignment")
    print("=" * 80)

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
