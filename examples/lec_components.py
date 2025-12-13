#!/usr/bin/env python3
"""LEC components example demonstrating sensory processing and decoding.

This example demonstrates the Lateral Entorhinal Cortex (LEC) components:
- Encoder: Two-hot sensory compression from high-dimensional observations
- Processor: Multi-frequency temporal filtering with learnable decay rates
- Projection: Tiling sensory to hippocampal space for conjunction with grid cells
- Decoder: Generating sensory predictions from grounded locations (place cells)

The LEC pipeline:
  Inference: x → Encoder → x_c → Processor → x_f → Projection → x̃
  Generative: p → Decoder → x̂

Note: This example uses a real environment with a random walk. For MEC spatial
processing, see mec_components.py.

Architecture:
-------------
    LEC Encoder (lec.encoder.Encoder):
        - Input: One-hot observations [B, n_x]
        - Output: Two-hot compressed sensory [B, n_x_c]
        - Method: Lookup table mapping observation index to two-hot code

    LEC Processor (lec.processor.Processor):
        - Input: Compressed sensory [B, n_x_c]
        - Output: Multi-frequency filtered sensory List[n_f] of [B, n_x_c]
        - Method: Per-frequency exponential smoothing + L2 normalization
        - Learnable parameters: Decay rates (alpha) for each frequency channel

    LEC Projection (lec.projection.Projection):
        - Input: Filtered sensory List[n_f] of [B, n_x_c]
        - Output: Tiled sensory List[n_f] of [B, n_p[f]]
        - Method: Normalize, tile to hippocampal dimension, frequency weighting
        - Prepares sensory for conjunction: p = g ⊗ x̃

    LEC Decoder (lec.decoder.Decoder):
        - Input: Grounded location (place cells) List[n_f] of [B, n_p[f]]
        - Output: Sensory predictions [B, n_x]
        - Method: Linear projection + MLP decoding
        - Generates expected observations from hippocampal activity

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
When save_plots=true, generates 7 visualizations in outputs/lec_components/:
    1. 01_frequency_bank.png - Frequency channel configuration and time constants
    2. 02_temporal_filtering.png - Temporal filtering heatmaps across all frequencies
    3. 03_frequency_comparison.png - Single feature comparison across frequencies
    4. 04_normalization_effects.png - Before/after normalization effects
    5. 05_sensory_projection.png - Projection to hippocampal p-space over time
    6. 06_decoder_predictions.png - Decoder sensory predictions from place cells
    7. 07_reconstruction_quality.png - Comparison of original vs decoded observations
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
    for demonstrating LEC components in isolation with real environment data.

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

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="lec_components")

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
    output_dir: Path = Field(default=Path("outputs/lec_components"), description="Directory for saving plots")
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

    This script demonstrates the sensory processing pathway:
      1. Generate a random walk trajectory in a grid environment
      2. Initialize LEC components (Encoder, Processor, Projection, Decoder)
      3. Process observations through the pipeline
      4. Generate visualizations of the complete LEC pathway
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
    # PHASE 2: Initialize LEC Components
    # =========================================================================
    print("Phase 2: Initializing LEC components...")

    # LEC Encoder: Compresses observations using two-hot encoding
    # x [B, n_x] → x_c [B, n_x_c]
    encoder = lec.encoder.Encoder(model_config)

    # LEC Processor: Multi-frequency temporal filtering
    # x_c → x_f (List[n_f] of [B, n_x_c])
    processor = lec.processor.Processor(model_config)

    # LEC Projection: Tiles sensory to hippocampal space
    # x_f → x̃ (List[n_f] of [B, n_p[f]])
    projection = lec.projection.Projection(model_config)

    # LEC Decoder: Generates sensory predictions from place cells
    # p → x̂ [B, n_x]
    decoder = lec.decoder.Decoder(model_config)

    # Create W_tile matrices for tiling operations
    # Enables conjunction: p = g ⊗ x̃
    W_tile = utils.create_W_tile(model_config.n_g_subsampled_combined, projection.n_x_f)

    print(f"  ✓ Encoder: {model_config.n_x} → {model_config.n_x_c} (two-hot compression)")
    print(f"  ✓ Processor: {model_config.n_f} frequency channels (f = {model_config.f_initial})")
    print(f"  ✓ Projection: {model_config.n_x_f} → {model_config.n_p} (hippocampal tiling)")
    print(f"  ✓ Decoder: place cells → sensory predictions")
    print()

    # =========================================================================
    # PHASE 3: Run LEC Pipeline
    # =========================================================================
    print("Phase 3: Running LEC pipeline (encoder → processor → projection → decoder)...")

    # Initialize history storage for visualization
    x_c_history = []  # Compressed sensory over time
    x_f_history = []  # Multi-frequency filtered sensory over time
    x__history = []  # Tiled projection to the hippocampus
    p_history = []  # Simulated place cell activity
    x_hat_history = []  # Decoded sensory predictions

    # Initialize processor state (previous filtered observations)
    x_f_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]

    for t in range(config.walk_length):
        # === INFERENCE PATHWAY ===
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # Add batch dimension: [n_x] → [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Step 2: Apply temporal filtering across frequencies
        x_f = processor(x_c, x_f_prev)  # List[n_f] of [1, n_x_c]

        # Step 3: Project to hippocampal p-space
        x_ = projection(x_f, W_tile)  # List[n_f] of [1, n_p[f]]

        # Update previous state
        x_f_prev = x_f

        # Store for visualization
        x_c_history.append(x_c[0])  # [1, n_x_c] → [n_x_c]
        x_f_history.append([x[0] for x in x_f])  # All frequencies, remove batch dim
        x__history.append([x[0] for x in x_])  # All frequencies, remove batch dim

        # === GENERATIVE PATHWAY ===
        # Simulate place cell activity (in real TEM, this comes from g ⊗ x̃)
        # Here we use the tiled sensory as a proxy for place cells
        p_t = x_  # Simulated grounded location from sensory
        p_history.append([p[0] for p in p_t])

        # Step 4: Decode place cells → sensory prediction
        x_hat = decoder(p_t, W_tile[0])  # Returns SensoryPrediction using first frequency's W_tile
        x_hat_history.append(x_hat.values[0][0])  # [1, n_x] → [n_x]

    print(f"  ✓ Processed {config.walk_length} timesteps through LEC pipeline")
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

    # Compare raw filtering vs normalized filtering (using projection.normalize)
    x_f_raw = processor.filter_temporal(x_c_demo, x_prev_demo)  # Raw exponential smoothing only
    x_f_normalized = projection.normalize(x_f_raw)  # Projection normalization (demean + ReLU + L2)

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

    # Plot 5: Projection to hippocampal p-space over time
    # Shows the tiled sensory representation ready for conjunction with grid cells
    fig5 = figures.plot_sensory_projection(
        x__history,
        n_p_per_freq=model_config.n_p,
        title="LEC → Hippocampus Projection: Sensory in p-space (x̃)",
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_sensory_projection.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_sensory_projection.png")

    # Plot 6: Decoder predictions
    # Show decoded sensory predictions over time
    fig6, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original observations
    obs_matrix = torch.stack(observations).detach().numpy()  # [T, n_x]
    axes[0].imshow(obs_matrix.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_title("Original Observations (x)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Observation Dimension", fontsize=10)
    axes[0].set_xlabel("Time Step", fontsize=10)

    # Decoded predictions
    x_hat_matrix = torch.stack(x_hat_history).detach().numpy()  # [T, n_x]
    axes[1].imshow(x_hat_matrix.T, aspect="auto", cmap="Oranges", interpolation="nearest")
    axes[1].set_title("Decoder Predictions (x̂ from p)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Observation Dimension", fontsize=10)
    axes[1].set_xlabel("Time Step", fontsize=10)

    # Reconstruction error
    error_matrix = obs_matrix - x_hat_matrix
    im = axes[2].imshow(error_matrix.T, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=-1, vmax=1)
    axes[2].set_title("Reconstruction Error (x - x̂)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Observation Dimension", fontsize=10)
    axes[2].set_xlabel("Time Step", fontsize=10)
    plt.colorbar(im, ax=axes[2], label="Error")

    fig6.suptitle("Decoder: Place Cells → Sensory Predictions", fontsize=14, y=0.995)
    fig6.tight_layout()

    if config.save_plots:
        fig6.savefig(config.output_dir / "06_decoder_predictions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_decoder_predictions.png")

    # Plot 7: Reconstruction quality metrics
    fig7, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE over time
    mse_per_step = ((obs_matrix - x_hat_matrix) ** 2).mean(axis=1)
    axes[0, 0].plot(mse_per_step, linewidth=2, color="crimson")
    axes[0, 0].set_title("Mean Squared Error Over Time", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Time Step")
    axes[0, 0].set_ylabel("MSE")
    axes[0, 0].grid(True, alpha=0.3)

    # Correlation over time
    correlations = []
    for t in range(len(observations)):
        corr = torch.corrcoef(torch.stack([observations[t], x_hat_history[t]]))[0, 1]
        correlations.append(corr.item())
    axes[0, 1].plot(correlations, linewidth=2, color="forestgreen")
    axes[0, 1].set_title("Correlation Between x and x̂", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Correlation")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="0.5 threshold")
    axes[0, 1].legend()

    # Distribution comparison
    axes[1, 0].hist(obs_matrix.flatten(), bins=50, alpha=0.5, label="Original (x)", color="blue")
    axes[1, 0].hist(x_hat_matrix.flatten(), bins=50, alpha=0.5, label="Decoded (x̂)", color="orange")
    axes[1, 0].set_title("Value Distribution Comparison", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Value")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Per-dimension reconstruction accuracy
    dim_mse = ((obs_matrix - x_hat_matrix) ** 2).mean(axis=0)
    axes[1, 1].bar(range(model_config.n_x), dim_mse, color="purple", alpha=0.7)
    axes[1, 1].set_title("Reconstruction Error Per Dimension", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Observation Dimension")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    fig7.suptitle("Reconstruction Quality Metrics", fontsize=14, y=0.995)
    fig7.tight_layout()

    if config.save_plots:
        fig7.savefig(config.output_dir / "07_reconstruction_quality.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 07_reconstruction_quality.png")

    print()
    print("=" * 80)
    print("LEC Pipeline Summary")
    print("=" * 80)
    print("\nINFERENCE PATHWAY (Sensory → Hippocampus):")
    print(f"  Input:  {model_config.n_x}-dim observations ({config.observation_mode} mode)")
    print(f"    ↓ LEC Encoder (two-hot compression)")
    print(f"  Stage 1: {model_config.n_x_c}-dim compressed sensory (x_c)")
    print(f"    ↓ LEC Processor ({model_config.n_f} frequencies: {model_config.f_initial})")
    print(f"  Stage 2: Multi-frequency filtered sensory (x_f) - List[{model_config.n_f}] of [batch, {model_config.n_x_c}]")
    print(f"    ↓ LEC Projection (tiling + weighting)")
    print(f"  Output: Hippocampal-ready sensory (x̃) - List[{model_config.n_f}] of [batch, n_p[f]]")
    print(f"          Dimensions per frequency: {model_config.n_p}")
    print(f"          Total hippocampal dimension: {sum(model_config.n_p)}")
    print("\nGENERATIVE PATHWAY (Hippocampus → Sensory):")
    print(f"  Input:  Grounded location (place cells p) - List[{model_config.n_f}] of [batch, n_p[f]]")
    print(f"    ↓ LEC Decoder (linear projection + MLP)")
    print(f"  Output: Sensory prediction (x̂) - [batch, {model_config.n_x}]")
    print("\nQUALITY METRICS:")
    print(f"  Average MSE: {mse_per_step.mean():.4f}")
    print(f"  Average Correlation: {sum(correlations)/len(correlations):.4f}")
    print("=" * 80)
    print()
    print("TEM Theory:")
    print("  • LEC provides sensory 'what' information to hippocampus")
    print("  • MEC provides spatial 'where' information (grid cells g)")
    print("  • Hippocampus forms conjunctive codes: p = g ⊗ x̃ ('where × what')")
    print("  • Decoder generates predictions: x̂ = decode(p) for planning and imagination")
    print("  • Multi-frequency filtering enables temporal credit assignment")
    print("=" * 80)
    print()
    if config.save_plots:
        print(f"All 7 visualizations saved to: {config.output_dir}")
    else:
        print("Plots not saved (use --save_plots true to save)")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
