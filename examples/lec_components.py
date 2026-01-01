#!/usr/bin/env python3
"""LEC components example demonstrating sensory processing and decoding.

This example demonstrates the Lateral Entorhinal Cortex (LEC) components:
- Encoder: Two-hot sensory compression from high-dimensional observations
- Processor: Multi-frequency temporal filtering with learnable decay rates
- Projection: Tiling sensory to hippocampal space for conjunction with grid cells
- Decoder: Generating sensory predictions from grounded locations (place cells)

The LEC pipeline:
  Inference: x → Encoder → o_c → Processor → x → Projection → x̃
  Generative: p → Decoder → x̂

Note: This example uses a real environment with a random walk. For MEC spatial
processing, see mec_components.py.

Architecture:
-------------
    LEC Encoder (lec.encoder.Encoder):
        - Input: One-hot observations [B, n_o]
        - Output: Two-hot compressed sensory [B, n_o_c]
        - Method: Lookup table mapping observation index to two-hot code

    LEC Processor (lec.processor.Processor):
        - Input: Compressed sensory [B, n_o_c]
        - Output: Multi-frequency filtered sensory List[n_f] of [B, n_o_c]
        - Method: Per-frequency exponential smoothing + L2 normalization
        - Learnable parameters: Decay rates (alpha) for each frequency channel

    LEC Projection (lec.projection.Projection):
        - Input: Filtered sensory List[n_f] of [B, n_o_c]
        - Output: Tiled sensory List[n_f] of [B, n_p[f]]
        - Method: Normalize, tile to hippocampal dimension, frequency weighting
        - Prepares sensory for conjunction: p = g ⊗ x̃

    LEC Decoder (lec.decoder.Decoder):
        - Input: Grounded location (place cells) List[n_f] of [B, n_p[f]]
        - Output: Sensory predictions [B, n_o]
        - Method: Linear projection + MLP decoding
        - Generates expected observations from hippocampal activity

Usage Examples:
---------------
    # Default: 5×5 grid, 100 timesteps, save plots
    python examples/lec_components.py

    # Configure LEC processor learning
    python examples/lec_components.py --processor.learn_alpha false

    # Configure tiling matrix learning
    python examples/lec_components.py --learn_w_tile true

    # Show plots interactively without saving
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

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.core import lec
from torch_tem.core.lec import DecoderConfig, EncoderConfig, ProcessorConfig, ProjectionConfig
from torch_tem.data.environment import EnvironmentConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for LEC sensory processing example.

    Defines submodule configurations and output settings for demonstrating
    LEC sensory processing components in isolation with real environment data.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="lec_components")

    # Lateral Entorhinal Cortex (LEC) parameters
    learn_W_tile: bool = Field(default=False, description="If True, tiling matrices W_tile are learnable")
    decoder: DecoderConfig = Field(default_factory=DecoderConfig, description="Decoder configuration")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig, description="Encoder configuration")
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig, description="Processor configuration")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig, description="Projection configuration")

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


# Model architecture; not configurable via CLI
N_X = 25  # Observation space size (5x5, one per cell)
N_X_C = 10  # Compressed dimension for two-hot encoding
N_P = [100, 80, 60, 50, 40]  # Place cells per frequency (5 modules)
N_WALKS = 1  # Number of walks to generate
WALK_LENGTH = 100  # Timesteps per walk
F_INITIAL = [0.95, 0.7, 0.4, 0.2, 0.1]  # Initial frequency values
DEVICE = torch.device("cpu")  # Change to "cuda" if GPU is available

# Create context for LEC components
W_tile = utils.create_tiling_matrices([N_X_C] * len(N_P), N_P)

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

    print("=" * 80)
    print("Sensory Processing Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Observation space: {N_X} ({int(N_X**0.5)}×{int(N_X**0.5)} grid)")
    print(f"  Compressed dimension: {N_X_C} (two-hot encoding)")
    print(f"  Frequencies: {len(F_INITIAL)} ({F_INITIAL})")
    print(f"  Place cells per frequency: {N_P}")
    print(f"  Walk length: {WALK_LENGTH} timesteps")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk Generation
    # =========================================================================
    print("Phase 1: Generating walk trajectory...")

    env_config = EnvironmentConfig(n_observations=N_X)
    env = data.Environment(env_config)
    env.validate()

    policy_gen = data.PolicyGenerator(env)
    walk_gen = data.WalkGenerator(env)
    walks = walk_gen.generate_walks(N_WALKS, WALK_LENGTH, policy=policy_gen.random_policy())
    walk = walks[0]

    observations = [obs.clone().detach() for obs in walk.observations]  # List[T] of [n_o]
    locations = torch.as_tensor(walk.locations, dtype=torch.long)  # [T]
    print(f"  ✓ Generated walk: {len(walk)} timesteps")
    print()

    # =========================================================================
    # PHASE 2: Initialize LEC Components
    # =========================================================================
    print("Phase 2: Initializing LEC components...")

    # LEC Encoder: Compresses observations using two-hot encoding
    # x [B, n_o] → o_c [B, n_o_c]
    encoder = lec.Encoder(N_X, N_X_C, config.encoder)

    # LEC Processor: Multi-frequency temporal filtering
    # o_c → x (List[n_f] of [B, n_o_c])
    processor = lec.Processor(F_INITIAL, config.processor)

    # LEC Projection: Tiles sensory to hippocampal space
    # x → x̃ (List[n_f] of [B, n_p[f]])
    projection = lec.Projection(W_tile, config.projection)

    # LEC Decoder: Generates sensory predictions from place cells
    # p → x̂ [B, n_o]
    decoder = lec.Decoder(N_X, W_tile, config.decoder)

    print(f"  ✓ Encoder: {N_X} → {N_X_C} (two-hot compression)")
    print(f"  ✓ Processor: {len(F_INITIAL)} frequency channels (f = {F_INITIAL})")
    print(f"  ✓ Projection: {N_X_C} → {N_P} (hippocampal tiling)")
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
    x_f_prev = [torch.zeros(1, N_X_C, device=DEVICE) for _ in range(len(F_INITIAL))]

    for t in range(WALK_LENGTH):
        # === INFERENCE PATHWAY ===
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # Add batch dimension: [n_o] → [1, n_o]
        o_c = encoder(x_t)  # [1, n_o_c]

        # Step 2: Apply temporal filtering across frequencies
        x = processor(o_c, x_f_prev)  # List[n_f] of [1, n_o_c]

        # Step 3: Project to hippocampal p-space
        x_ = projection(x)  # List[n_f] of [1, n_p[f]]

        # Update previous state
        x_f_prev = x

        # Store for visualization
        x_c_history.append(o_c[0])  # [1, n_o_c] → [n_o_c]
        x_f_history.append([x[0] for x in x])  # All frequencies, remove batch dim
        x__history.append([x[0] for x in x_])  # All frequencies, remove batch dim

        # === GENERATIVE PATHWAY ===
        # Simulate place cell activity (in real TEM, this comes from g ⊗ x̃)
        # Here we use the tiled sensory as a proxy for place cells
        p_t = x_  # Simulated grounded location from sensory
        p_history.append([p[0] for p in p_t])

        # Step 4: Decode place cells → sensory prediction
        x_hat = decoder(p_t)  # Returns SensoryPrediction
        x_hat_history.append(x_hat.values[0][0])  # [1, n_o] → [n_o]

    print(f"  ✓ Processed {WALK_LENGTH} timesteps through LEC pipeline")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Generate demo data for normalization comparison (5 consecutive timesteps)
    observations_stacked = torch.stack(observations)  # List[T] of [n_o] → [T, n_o]
    midpoint = WALK_LENGTH // 2
    x_c_demo = encoder(observations_stacked[midpoint : midpoint + 5])  # [5, n_o_c]
    x_prev_demo = [torch.zeros(5, N_X_C, device=DEVICE) for _ in range(len(F_INITIAL))]

    # Compare raw filtering vs normalized filtering (using projection.normalize)
    x_f_raw = processor.filter_temporal(x_c_demo, x_prev_demo)  # Raw exponential smoothing only
    x_f_normalized = projection.normalize(x_f_raw)  # Projection normalization (demean + ReLU + L2)

    # Plot 1: Frequency bank configuration
    fig1 = figures.plot_frequency_bank(F_INITIAL)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_frequency_bank.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_frequency_bank.png")

    # Plot 2: Temporal filtering across all frequencies
    fig2 = figures.plot_temporal_filtering(x_c_history, x_f_history, F_INITIAL)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_temporal_filtering.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_temporal_filtering.png")

    # Plot 3: Single feature comparison
    fig3 = figures.plot_frequency_comparison(x_c_history, x_f_history, F_INITIAL, feature_idx=0)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_frequency_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_frequency_comparison.png")

    # Plot 4: Normalization effects (single timestep)
    fig4 = figures.plot_normalization_effects(x_f_raw, x_f_normalized, F_INITIAL)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_normalization_effects.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_normalization_effects.png")

    # Plot 5: Projection to hippocampal p-space over time
    # Shows the tiled sensory representation ready for conjunction with grid cells
    fig5 = figures.plot_sensory_projection(
        x__history,
        n_p_per_freq=N_P,
        title="LEC → Hippocampus Projection: Sensory in p-space (x̃)",
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_sensory_projection.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_sensory_projection.png")

    # Plot 6: Decoder predictions
    # Show decoded sensory predictions over time
    fig6, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original observations
    obs_matrix = torch.stack(observations).detach().numpy()  # [T, n_o]
    axes[0].imshow(obs_matrix.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_title("Original Observations (x)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Observation Dimension", fontsize=10)
    axes[0].set_xlabel("Time Step", fontsize=10)

    # Decoded predictions
    x_hat_matrix = torch.stack(x_hat_history).detach().numpy()  # [T, n_o]
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
    fig7 = figures.plot_reconstruction_quality(observations, x_hat_history)
    if config.save_plots:
        fig7.savefig(config.output_dir / "07_reconstruction_quality.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 07_reconstruction_quality.png")

    # Calculate quality metrics for summary
    obs_matrix = torch.stack(observations).detach().numpy()
    x_hat_matrix = torch.stack(x_hat_history).detach().numpy()
    mse_per_step = ((obs_matrix - x_hat_matrix) ** 2).mean(axis=1)
    correlations = []
    for t in range(len(observations)):
        corr = torch.corrcoef(torch.stack([observations[t], x_hat_history[t]]))[0, 1]
        correlations.append(corr.item())

    print()
    print("=" * 80)
    print("LEC Pipeline Summary")
    print("=" * 80)
    print("\nINFERENCE PATHWAY (Sensory → Hippocampus):")
    print(f"  Input:  {N_X}-dim observations")
    print(f"    ↓ LEC Encoder (two-hot compression)")
    print(f"  Stage 1: {N_X_C}-dim compressed sensory (o_c)")
    print(f"    ↓ LEC Processor ({len(F_INITIAL)} frequencies: {F_INITIAL})")
    print(f"  Stage 2: Multi-frequency filtered sensory (x) - List[{len(F_INITIAL)}] of [batch, {N_X_C}]")
    print(f"    ↓ LEC Projection (tiling + weighting)")
    print(f"  Output: Hippocampal-ready sensory (x̃) - List[{len(F_INITIAL)}] of [batch, n_p[f]]")
    print(f"          Dimensions per frequency: {N_P}")
    print(f"          Total hippocampal dimension: {sum(N_P)}")
    print("\nGENERATIVE PATHWAY (Hippocampus → Sensory):")
    print(f"  Input:  Grounded location (place cells p) - List[{len(F_INITIAL)}] of [batch, n_p[f]]")
    print(f"    ↓ LEC Decoder (linear projection + MLP)")
    print(f"  Output: Sensory prediction (x̂) - [batch, {N_X}]")
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
