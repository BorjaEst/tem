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
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

from torch_tem import data
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.inference.sensory import SensoryProcessor


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for sensory processing example."""

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

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Visualization Functions
# ==============================================================================
def plot_frequency_bank(frequencies: List[float], title: str = "Frequency Bank Configuration") -> plt.Figure:
    """Visualize frequency bank showing memory time constants.

    Args:
        frequencies: List of frequency values in (0, 1]
        title: Plot title

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot frequencies
    ax1.bar(range(len(frequencies)), frequencies, color="steelblue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Frequency Channel", fontsize=11)
    ax1.set_ylabel("Frequency Value", fontsize=11)
    ax1.set_title("Frequency Values (0 = long memory, 1 = no memory)", fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(range(len(frequencies)))

    # Plot effective time constants (1 / frequency)
    time_constants = [1.0 / f if f > 0 else float("inf") for f in frequencies]
    ax2.bar(range(len(time_constants)), time_constants, color="darkorange", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Frequency Channel", fontsize=11)
    ax2.set_ylabel("Effective Time Constant (steps)", fontsize=11)
    ax2.set_title("Memory Decay Time Constants", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(range(len(frequencies)))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_temporal_filtering(
    x_c_history: Tensor,
    x_f_history: List[List[Tensor]],
    frequencies: List[float],
    locations: Tensor,
    title: str = "Temporal Filtering Across Frequencies",
) -> plt.Figure:
    """Visualize temporal filtering effects across frequency channels.

    Args:
        x_c_history: [T, n_x_c] compressed sensory over time
        x_f_history: List of T timesteps, each with n_f filtered tensors [n_x_c]
        frequencies: List of frequency values
        locations: [T] location indices for coloring
        title: Plot title

    Returns:
        matplotlib Figure
    """
    n_f = len(frequencies)
    T = x_c_history.shape[0]

    # Create subplot grid: original + all frequencies
    fig, axes = plt.subplots(n_f + 1, 1, figsize=(14, 2 * (n_f + 1)), sharex=True)

    # Plot original compressed sensory
    x_c_np = x_c_history.detach().cpu().numpy()
    im0 = axes[0].imshow(x_c_np.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_ylabel("Feature Dim", fontsize=10)
    axes[0].set_title("Original Compressed Sensory (x_c)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot each frequency channel
    for f_idx in range(n_f):
        # Stack filtered tensors over time: [T, n_x_c]
        x_f_t = torch.stack([x_f_history[t][f_idx] for t in range(T)])
        x_f_np = x_f_t.detach().cpu().numpy()

        im = axes[f_idx + 1].imshow(x_f_np.T, aspect="auto", cmap="viridis", interpolation="nearest")
        axes[f_idx + 1].set_ylabel("Feature Dim", fontsize=10)
        freq_val = frequencies[f_idx]
        tau = 1.0 / freq_val if freq_val > 0 else float("inf")
        axes[f_idx + 1].set_title(f"Frequency {f_idx}: f={freq_val:.2f} (τ≈{tau:.1f} steps)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[f_idx + 1], fraction=0.046, pad=0.04)

    axes[-1].set_xlabel("Time Step", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


def plot_frequency_comparison(
    x_c_history: Tensor, x_f_history: List[List[Tensor]], frequencies: List[float], feature_idx: int = 0, title: str = "Single Feature Across Frequencies"
) -> plt.Figure:
    """Plot a single feature dimension across all frequency channels over time.

    Args:
        x_c_history: [T, n_x_c] compressed sensory
        x_f_history: List of T timesteps with n_f filtered tensors
        frequencies: List of frequency values
        feature_idx: Which feature dimension to plot
        title: Plot title

    Returns:
        matplotlib Figure
    """
    n_f = len(frequencies)
    T = x_c_history.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Plot original
    x_c_feature = x_c_history[:, feature_idx].detach().cpu().numpy()
    ax.plot(range(T), x_c_feature, label="Original (x_c)", linewidth=2, color="black", linestyle="--", alpha=0.7)

    # Plot each frequency
    colors = plt.cm.viridis(np.linspace(0, 1, n_f))
    for f_idx in range(n_f):
        x_f_feature = torch.stack([x_f_history[t][f_idx][feature_idx] for t in range(T)])
        x_f_np = x_f_feature.detach().cpu().numpy()
        freq_val = frequencies[f_idx]
        tau = 1.0 / freq_val if freq_val > 0 else float("inf")
        ax.plot(range(T), x_f_np, label=f"f={freq_val:.2f} (τ≈{tau:.1f})", linewidth=1.5, color=colors[f_idx], alpha=0.8)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    ax.set_title(f"{title} (Feature {feature_idx})", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_normalization_effects(x_f_raw: List[Tensor], x_f_normalized: List[Tensor], frequencies: List[float], title: str = "Normalization Effects") -> plt.Figure:
    """Compare raw filtered outputs with normalized outputs.

    Args:
        x_f_raw: List of n_f raw filtered tensors [B, n_x_c]
        x_f_normalized: List of n_f normalized tensors [B, n_x_c]
        frequencies: List of frequency values
        title: Plot title

    Returns:
        matplotlib Figure
    """
    n_f = len(frequencies)
    fig, axes = plt.subplots(2, n_f, figsize=(3 * n_f, 6), sharex=True, sharey="row")

    for f_idx in range(n_f):
        # Raw
        raw_np = x_f_raw[f_idx].detach().cpu().numpy()
        im1 = axes[0, f_idx].imshow(raw_np.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
        axes[0, f_idx].set_title(f"f={frequencies[f_idx]:.2f} (Raw)", fontsize=10)
        plt.colorbar(im1, ax=axes[0, f_idx], fraction=0.046, pad=0.04)

        # Normalized
        norm_np = x_f_normalized[f_idx].detach().cpu().numpy()
        im2 = axes[1, f_idx].imshow(norm_np.T, aspect="auto", cmap="coolwarm", interpolation="nearest")
        axes[1, f_idx].set_title(f"f={frequencies[f_idx]:.2f} (Norm)", fontsize=10)
        plt.colorbar(im2, ax=axes[1, f_idx], fraction=0.046, pad=0.04)

        if f_idx == 0:
            axes[0, f_idx].set_ylabel("Feature Dim (Raw)", fontsize=10)
            axes[1, f_idx].set_ylabel("Feature Dim (Norm)", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ==============================================================================
# Helper: Create minimal params object
# ==============================================================================
def create_sensory_params(n_f: int, n_x_c: int, f_min: float, f_max: float):
    """Create a minimal parameter object for SensoryProcessor.

    Args:
        n_f: Number of frequency channels
        n_x_c: Compressed sensory dimension
        f_min: Minimum frequency
        f_max: Maximum frequency

    Returns:
        Simple namespace satisfying SensoryProcessorParams protocol
    """
    import types

    # Generate frequency bank (logarithmic spacing)
    if n_f == 1:
        frequencies = [f_max]
    else:
        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), n_f).tolist()

    return types.SimpleNamespace(
        n_f_calculated=n_f,
        n_x_c=n_x_c,
        n_x_f_calculated=n_f * n_x_c,
        f_initial_extended=frequencies,
    )


def create_encoder_params(n_x: int, n_x_c: int):
    """Create a minimal parameter object for SensoryEncoder.

    Args:
        n_x: Number of distinct observations
        n_x_c: Compressed sensory dimension

    Returns:
        Simple namespace satisfying EncoderParams protocol
    """
    import types

    # Generate two-hot encoding table
    two_hot_table = []
    for i in range(n_x):
        code = torch.zeros(n_x_c)
        idx1 = i % n_x_c
        idx2 = (i + 1) % n_x_c
        code[idx1] = 1.0
        code[idx2] = 1.0
        two_hot_table.append(code)

    return types.SimpleNamespace(
        n_x=n_x,
        n_x_c=n_x_c,
        two_hot_table_calculated=two_hot_table,
    )


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
    observations = torch.stack([torch.tensor(obs, dtype=torch.float32) for obs in walk.observations])  # [T, n_x]
    locations = torch.tensor(walk.locations, dtype=torch.long)  # [T]
    print(f"  Generated walk: {config.walk_length} steps")

    # 2. Initialize sensory encoder and processor
    print("\n[2/5] Initializing SensoryEncoder and SensoryProcessor...")
    encoder_params = create_encoder_params(n_x=env.n_observations, n_x_c=config.n_x_c)
    encoder = SensoryEncoder(encoder_params)

    sensory_params = create_sensory_params(n_f=config.n_frequencies, n_x_c=config.n_x_c, f_min=config.f_min, f_max=config.f_max)
    processor = SensoryProcessor(sensory_params)
    print(f"  Frequencies: {sensory_params.f_initial_extended}")

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
    fig1 = plot_frequency_bank(sensory_params.f_initial_extended, title="Frequency Bank Configuration")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_frequency_bank.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_frequency_bank.png'}")

    # Plot 2: Temporal filtering across all frequencies
    fig2 = plot_temporal_filtering(x_c_history, x_f_history, sensory_params.f_initial_extended, locations, title="Temporal Filtering Across Frequencies")
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_temporal_filtering.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_temporal_filtering.png'}")

    # Plot 3: Single feature comparison
    fig3 = plot_frequency_comparison(x_c_history, x_f_history, sensory_params.f_initial_extended, feature_idx=0, title="Feature 0 Across Frequencies")
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

    fig4 = plot_normalization_effects(x_f_raw, x_f_normalized, sensory_params.f_initial_extended, title="L2 Normalization Effects")
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
