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

from torch_tem import data, figures
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
        from torch_tem.utils import create_two_hot_table

        return create_two_hot_table(self.n_x, self.n_x_c)

    @computed_field(description="Kronecker repeat matrices for outer product")
    @property
    def W_repeat_calculated(self) -> List[Tensor]:
        from torch_tem.utils import create_W_repeat

        return create_W_repeat(self.n_g_subsampled_per_freq, [self.n_x_c] * self.n_frequencies)

    @computed_field(description="Kronecker tile matrices for outer product")
    @property
    def W_tile_calculated(self) -> List[Tensor]:
        from torch_tem.utils import create_W_tile

        return create_W_tile(self.n_g_subsampled_per_freq, [self.n_x_c] * self.n_frequencies)

    @computed_field(description="Downsampling matrices for grid cells")
    @property
    def g_downsample_calculated(self) -> List[Tensor]:
        from torch_tem.utils import create_g_downsample

        return create_g_downsample(self.n_g_calculated, self.n_g_subsampled_per_freq)


# ==============================================================================
# Helper Functions
# ==============================================================================
def generate_synthetic_grid_cells(n_steps: int, n_g: List[int], frequencies: List[float], batch_size: int = 1) -> List[Tensor]:
    """Generate synthetic grid cell activity patterns.

    Creates oscillating patterns with frequency-dependent dynamics to simulate
    grid cell responses during spatial navigation.

    Args:
        n_steps: Number of timesteps
        n_g: Grid cell dimensions per frequency
        frequencies: Frequency values per module
        batch_size: Batch size

    Returns:
        List of [T, B, n_g[f]] tensors with synthetic grid patterns
    """
    n_f = len(n_g)
    g_history = []

    for f in range(n_f):
        # Create oscillating patterns with random phase offsets
        t = torch.linspace(0, 10 * frequencies[f], n_steps).unsqueeze(1).unsqueeze(2)  # [T, 1, 1]
        phases = torch.randn(1, batch_size, n_g[f]) * 2 * np.pi  # [1, B, n_g[f]]

        # Combine multiple oscillations
        pattern = torch.sin(t + phases) + 0.3 * torch.sin(2 * t + phases * 0.5)
        pattern = pattern + 0.2 * torch.randn_like(pattern)  # Add noise

        g_history.append(pattern)

    return g_history


def plot_grounded_location_activity(p_history: List[List[Tensor]], observations: Tensor, locations: Tensor, frequencies: List[float], n_cells_per_freq: List[int]) -> plt.Figure:
    """Plot grounded location (place cell) activity over time.

    Args:
        p_history: List[T] of [List[n_f] of [B, n_p[f]]]
        observations: [T, n_x] observation indices
        locations: [T] location indices
        frequencies: Frequency values
        n_cells_per_freq: Number of place cells per frequency

    Returns:
        matplotlib Figure
    """
    n_f = len(p_history[0])
    T = len(p_history)

    fig, axes = plt.subplots(n_f + 2, 1, figsize=(14, 3 * (n_f + 2)), sharex=True)

    # Plot observations
    obs_indices = torch.argmax(observations, dim=1).numpy()
    axes[0].plot(obs_indices, "o-", linewidth=1, markersize=3, color="black")
    axes[0].set_ylabel("Observation\nIndex", fontsize=10)
    axes[0].set_title("Sensory Input (Observations)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot locations
    axes[1].plot(locations.numpy(), "s-", linewidth=1, markersize=3, color="darkblue")
    axes[1].set_ylabel("Location\nIndex", fontsize=10)
    axes[1].set_title("Spatial Location", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Plot place cell activity per frequency
    for f in range(n_f):
        # Extract activity over time [T, n_p[f]]
        activity = torch.stack([p_history[t][f][0] for t in range(T)])  # [T, n_p[f]]

        # Subsample cells for visualization if too many
        max_cells = 30
        if n_cells_per_freq[f] > max_cells:
            indices = torch.linspace(0, n_cells_per_freq[f] - 1, max_cells).long()
            activity = activity[:, indices]
            n_vis = max_cells
        else:
            n_vis = n_cells_per_freq[f]

        # Plot as heatmap
        im = axes[f + 2].imshow(activity.T.detach().numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
        axes[f + 2].set_ylabel(f"Place Cells\nFreq {f}\n(n={n_vis})", fontsize=9)
        axes[f + 2].set_title(f"Frequency {f} (f={frequencies[f]:.2f}) - Place Cell Activity", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[f + 2], label="Activation")

    axes[-1].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout()
    return fig


def plot_outer_product_structure(g_sample: List[Tensor], x_sample: List[Tensor], p_sample: List[Tensor], frequencies: List[float]) -> plt.Figure:
    """Visualize outer product structure: p = g ⊗ x.

    Args:
        g_sample: Grid cell activity [n_f] of [1, n_g_sub[f]]
        x_sample: Sensory activity [n_f] of [1, n_x_c]
        p_sample: Place cell activity [n_f] of [1, n_p[f]]
        frequencies: Frequency values

    Returns:
        matplotlib Figure
    """
    n_f = len(g_sample)
    fig, axes = plt.subplots(n_f, 3, figsize=(12, 3 * n_f))

    if n_f == 1:
        axes = axes.reshape(1, -1)

    for f in range(n_f):
        g_vec = g_sample[f][0].detach().numpy()  # [n_g_sub[f]]
        x_vec = x_sample[f][0].detach().numpy()  # [n_x_c]
        p_vec = p_sample[f][0].detach().numpy()  # [n_p[f]]

        # Plot grid cells
        axes[f, 0].bar(range(len(g_vec)), g_vec, color="steelblue")
        axes[f, 0].set_title(f"Freq {f} (f={frequencies[f]:.2f})\nGrid Cells g[{f}]", fontsize=10, fontweight="bold")
        axes[f, 0].set_xlabel("Grid Cell Index")
        axes[f, 0].set_ylabel("Activation")
        axes[f, 0].grid(True, alpha=0.3)

        # Plot sensory
        axes[f, 1].bar(range(len(x_vec)), x_vec, color="darkorange")
        axes[f, 1].set_title(f"Sensory x[{f}]", fontsize=10, fontweight="bold")
        axes[f, 1].set_xlabel("Sensory Feature Index")
        axes[f, 1].set_ylabel("Activation")
        axes[f, 1].grid(True, alpha=0.3)

        # Plot place cells (reshaped to show structure)
        n_g = len(g_vec)
        n_x = len(x_vec)
        p_matrix = p_vec.reshape(n_g, n_x)  # Reshape to show conjunctive structure

        im = axes[f, 2].imshow(p_matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
        axes[f, 2].set_title(f"Place Cells p[{f}] = g ⊗ x\n(Conjunctive Coding)", fontsize=10, fontweight="bold")
        axes[f, 2].set_xlabel("Sensory Feature")
        axes[f, 2].set_ylabel("Grid Cell")
        plt.colorbar(im, ax=axes[f, 2], label="Activation")

    plt.tight_layout()
    return fig


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the grounded location inference experiment with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Grounded Location Inference Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Grid size: {config.grid_size}x{config.grid_size}")
    print(f"  Observations: {config.n_x}")
    print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Grid cells per freq: {config.n_g_subsampled_per_freq}")
    print(f"  Place cells per freq: {config.n_p_calculated}")
    print(f"  Frequency values: {[f'{f:.2f}' for f in config.f_initial_extended]}")

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

    print(f"\nWalk generated:")
    print(f"  Length: {len(walk.observations)} steps")
    print(f"  Unique locations visited: {len(set(walk.locations))}")

    # Initialize components using config (implements all required protocols)
    encoder = SensoryEncoder(config)
    processor = SensoryProcessor(config)
    projection = ProjectionHead(config)
    grounded = GroundedLocationInference(config)

    print(f"\nComponents initialized:")
    print(f"  SensoryEncoder: {config.n_x} → {config.n_x_c}")
    print(f"  SensoryProcessor: {config.n_frequencies} frequency channels")
    print(f"  ProjectionHead: downsample + normalize")
    print(f"  GroundedLocationInference: g ⊗ x → p")

    # Generate synthetic grid cell patterns (simulating abstract location)
    g_history = generate_synthetic_grid_cells(config.walk_length, config.n_g_calculated, config.f_initial_extended, batch_size=1)

    print(f"\nSynthetic grid cells generated:")
    for f in range(config.n_frequencies):
        print(f"  Freq {f}: shape {tuple(g_history[f].shape)}")

    # Process observations through full inference pipeline
    x_c_history = []
    x_f_history = []
    p_history = []
    x_prev = [torch.zeros(1, config.n_x_c) for _ in range(config.n_frequencies)]

    print(f"\nRunning inference pipeline...")
    for t in range(config.walk_length):
        # 1. Encode observation
        x_t = observations[t : t + 1]  # [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # 2. Temporal filtering and normalization
        x_f = processor(x_c, x_prev)  # [n_f] of [1, n_x_c]

        # 3. Get grid cells at time t
        g_t = [g_history[f][t : t + 1, 0, :] for f in range(config.n_frequencies)]  # [n_f] of [1, n_g[f]]

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

    print(f"✓ Inference complete: {config.walk_length} timesteps processed")

    # Visualizations
    print(f"\nGenerating visualizations...")

    # Plot 1: Grounded location activity over time
    fig1 = plot_grounded_location_activity(p_history, observations, locations, config.f_initial_extended, config.n_p_calculated)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_place_cell_activity.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Place cell activity plot")

    # Plot 2: Outer product structure at specific timepoint
    mid_point = config.walk_length // 2
    g_mid = [g_history[f][mid_point : mid_point + 1, 0, :] for f in range(config.n_frequencies)]
    g_mid_transformed = projection.transform(g_mid)
    g_mid_downsampled = projection.downsample(g_mid_transformed)
    fig2 = plot_outer_product_structure(g_mid_downsampled, x_f_history[mid_point], p_history[mid_point], config.f_initial_extended)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_outer_product_structure.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Outer product structure plot")

    # Plot 3: Frequency comparison (single place cell across time)
    fig3, axes = plt.subplots(config.n_frequencies, 1, figsize=(12, 3 * config.n_frequencies), sharex=True)
    if config.n_frequencies == 1:
        axes = [axes]

    for f in range(config.n_frequencies):
        # Extract single place cell over time
        cell_idx = config.n_p_calculated[f] // 2  # Middle cell
        activity = torch.stack([p_history[t][f][0, cell_idx] for t in range(config.walk_length)])

        axes[f].plot(activity.detach().numpy(), linewidth=1.5, color=f"C{f}")
        axes[f].set_ylabel(f"Activation\nFreq {f}", fontsize=10)
        axes[f].set_title(f"Place Cell {cell_idx} (Freq {f}, f={config.f_initial_extended[f]:.2f})", fontsize=11, fontweight="bold")
        axes[f].grid(True, alpha=0.3)

        # Mark observation changes
        obs_changes = torch.where(torch.diff(torch.argmax(observations, dim=1)) != 0)[0] + 1
        for change in obs_changes:
            axes[f].axvline(change, color="red", alpha=0.2, linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout()
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_temporal_dynamics.png", dpi=150, bbox_inches="tight")
    print(f"  ✓ Temporal dynamics plot")

    # Summary statistics
    print(f"\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for f in range(config.n_frequencies):
        p_all = torch.stack([p_history[t][f] for t in range(config.walk_length)])
        print(f"Frequency {f} (f={config.f_initial_extended[f]:.2f}):")
        print(f"  Place cells: {config.n_p_calculated[f]}")
        print(f"  Mean activation: {p_all.mean():.4f}")
        print(f"  Std activation: {p_all.std():.4f}")
        print(f"  Max activation: {p_all.max():.4f}")
        print(f"  Min activation: {p_all.min():.4f}")

    print(f"\n" + "=" * 80)
    print(f"Plots saved to: {config.output_dir}")
    print("=" * 80)

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
