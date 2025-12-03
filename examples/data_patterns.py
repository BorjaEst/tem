#!/usr/bin/env python3
"""Pattern generation example with CLI configuration and visualizations.

This example demonstrates the torch_tem.data.patterns module capabilities:
- Place cell pattern generation with configurable sparsity
- Grid cell pattern generation across multiple frequencies
- Oscillatory grid cell patterns with temporal dynamics
- Paired pattern generation with correlation
- Temporal sequence generation and smoothness analysis

Usage:
    python examples/data_patterns.py --batch-size 8 --place-sparsity 0.15
    python examples/data_patterns.py --n-frequencies 4 --walk-length 100
    python examples/data_patterns.py --help
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures


# ==============================================================================
# Configuration
# ==============================================================================
class PatternConfig(BaseModel):
    """Neural architecture configuration for pattern generation."""

    n_g: List[int] = Field(default=[30, 25, 20], description="Grid cell dimensions per frequency")
    n_p: List[int] = Field(default=[240, 200, 160], description="Place cell dimensions per frequency")
    n_x_c: int = Field(default=8, description="Compressed sensory dimension")
    f_extended: List[float] = Field(default=[0.9, 0.5, 0.2], description="Frequency values for oscillatory patterns")


class ExampleConfig(BaseSettings):
    """Configuration for pattern generation example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="data_patterns")

    # Architecture
    n_frequencies: int = Field(default=3, ge=1, le=5, description="Number of frequency modules")
    grid_cells_per_freq: int = Field(default=30, ge=10, le=50, description="Base number of grid cells per frequency")
    place_cells_per_freq: int = Field(default=240, ge=50, le=500, description="Base number of place cells per frequency")

    # Pattern generation
    batch_size: int = Field(default=8, ge=1, le=32, description="Number of patterns to generate")
    place_sparsity: float = Field(default=0.1, ge=0.01, le=0.5, description="Place cell sparsity level (lower = sparser)")
    grid_noise_scale: float = Field(default=0.1, ge=0.0, le=1.0, description="Noise scale for grid cell patterns")
    place_noise_scale: float = Field(default=0.2, ge=0.0, le=1.0, description="Noise scale for place cell patterns")

    # Temporal patterns
    walk_length: int = Field(default=100, ge=10, le=500, description="Length of temporal sequences")
    temporal_smoothness: float = Field(default=0.7, ge=0.0, le=1.0, description="Temporal smoothness factor (higher = smoother)")

    # Oscillatory patterns
    time_scale: float = Field(default=10.0, ge=1.0, le=50.0, description="Time scaling for oscillations")
    harmonic_weight: float = Field(default=0.3, ge=0.0, le=1.0, description="Weight for second harmonic")
    oscillatory_noise: float = Field(default=0.2, ge=0.0, le=1.0, description="Noise scale for oscillatory patterns")

    # Paired patterns
    correlation: float = Field(default=0.3, ge=0.0, le=1.0, description="Correlation between grid and place patterns")

    # Output
    output_dir: Path = Field(default=Path("outputs/data_patterns"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    def create_pattern_config(self) -> PatternConfig:
        """Create PatternConfig from example settings."""
        # Create decreasing dimensions across frequencies
        n_g = [max(10, self.grid_cells_per_freq - f * 5) for f in range(self.n_frequencies)]
        n_p = [max(50, self.place_cells_per_freq - f * 40) for f in range(self.n_frequencies)]

        # Create decreasing frequencies
        f_extended = [0.9 - f * (0.7 / max(1, self.n_frequencies - 1)) for f in range(self.n_frequencies)]

        return PatternConfig(n_g=n_g, n_p=n_p, n_x_c=8, f_extended=f_extended)


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the pattern generation example with visualizations."""
    config = ExampleConfig()
    pattern_config = config.create_pattern_config()

    print("=" * 80)
    print("Pattern Generation Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Grid dimensions: {pattern_config.n_g}")
    print(f"  Place dimensions: {pattern_config.n_p}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Walk length: {config.walk_length}")

    # ==========================================================================
    # Example 1: Place Cell Patterns
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("1. Generating Place Cell Patterns")
    print(f"{'-'*80}")

    place_gen = data.PlaceCellPatternGenerator(pattern_config, sparsity=config.place_sparsity, noise_scale=config.place_noise_scale)
    p_batch = place_gen.generate(config.batch_size)

    print(f"  Generated: {p_batch.shape}")
    print(f"  Sparsity: {(p_batch > 0.01).float().mean():.2%}")
    print(f"  Mean activation: {p_batch.mean():.6f}")

    fig1 = figures.plot_place_cell_patterns(p_batch, title=f"Place Cell Patterns (sparsity={config.place_sparsity:.2f})", n_samples=min(4, config.batch_size))
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_place_cell_patterns.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 2: Grid Cell Patterns
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("2. Generating Grid Cell Patterns")
    print(f"{'-'*80}")

    grid_gen = data.GridCellPatternGenerator(pattern_config, noise_scale=config.grid_noise_scale)
    g_batch = grid_gen.generate(config.batch_size)

    print(f"  Generated: {len(g_batch)} frequency modules")
    for f, g_f in enumerate(g_batch):
        print(f"    Frequency {f}: shape={g_f.shape}, mean={g_f.mean():.6f}")

    fig2 = figures.plot_grid_cell_patterns(
        g_batch, pattern_config.f_extended, title=f"Grid Cell Patterns ({config.n_frequencies} frequencies)", n_samples=min(4, config.batch_size)
    )
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_grid_cell_patterns.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 3: Oscillatory Grid Cell Patterns
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("3. Generating Oscillatory Grid Cell Patterns")
    print(f"{'-'*80}")

    oscillatory_gen = data.OscillatoryGridGenerator(
        pattern_config, walk_length=config.walk_length, batch_size=1, time_scale=config.time_scale, harmonic_weight=config.harmonic_weight, noise_scale=config.oscillatory_noise
    )
    g_history = oscillatory_gen.generate()

    print(f"  Generated: {len(g_history)} timesteps")
    print(f"  Frequencies: {len(g_history[0])} modules")
    for f in range(len(g_history[0])):
        g_f_shape = g_history[0][f].shape
        print(f"    Frequency {f}: shape={g_f_shape}")

    fig3 = figures.plot_oscillatory_patterns(g_history, pattern_config.f_extended, title="Oscillatory Grid Cell Patterns", cell_indices=[0, 1, 2])
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_oscillatory_patterns.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 4: Paired Grid-Place Patterns
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("4. Generating Paired (Grid, Place) Patterns")
    print(f"{'-'*80}")

    paired_gen = data.PairedPatternGenerator(pattern_config, correlation=config.correlation, place_sparsity=config.place_sparsity, noise_scale=config.place_noise_scale)
    g_paired, p_paired = paired_gen.generate(config.batch_size)

    print(f"  Grid patterns: {len(g_paired)} modules")
    print(f"  Place patterns: {p_paired.shape}")
    print(f"  Correlation: {config.correlation:.2f}")

    fig4 = figures.plot_pattern_correlations(g_paired, p_paired, pattern_config.f_extended, title=f"Grid-Place Correlations (ρ={config.correlation:.2f})")
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_pattern_correlations.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 5: Temporal Place Cell Sequences
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("5. Generating Temporal Place Cell Sequences")
    print(f"{'-'*80}")

    p_sequence = place_gen.generate_sequence(config.walk_length, config.batch_size, temporal_smoothness=config.temporal_smoothness)

    print(f"  Sequence shape: {p_sequence.shape}")
    print(f"  Temporal smoothness: {config.temporal_smoothness:.2f}")

    fig5 = figures.plot_temporal_patterns(p_sequence, title=f"Temporal Place Cell Evolution (smoothness={config.temporal_smoothness:.2f})", n_cells_display=50)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_temporal_sequences.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 6: Temporal Grid Cell Sequences
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("6. Generating Temporal Grid Cell Sequences")
    print(f"{'-'*80}")

    g_sequences = grid_gen.generate_sequence(config.walk_length, config.batch_size, temporal_smoothness=config.temporal_smoothness)

    print(f"  Grid sequences: {len(g_sequences)} frequency modules")
    for f, g_f_seq in enumerate(g_sequences):
        print(f"    Frequency {f}: shape={g_f_seq.shape}")

    # Flatten grid sequences for visualization
    g_seq_flat = g_sequences[0]  # Show first frequency
    fig6 = figures.plot_temporal_patterns(g_seq_flat, title=f"Temporal Grid Cell Evolution (Freq 0, smoothness={config.temporal_smoothness:.2f})", n_cells_display=30)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_temporal_grid_sequences.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Example 7: Pattern Generator Comparison
    # ==========================================================================
    print(f"\n{'-'*80}")
    print("7. Comparing Different Pattern Types")
    print(f"{'-'*80}")

    # Generate patterns with different configurations
    sparse_place = data.PlaceCellPatternGenerator(pattern_config, sparsity=0.05, noise_scale=0.2).generate(config.batch_size)
    medium_place = data.PlaceCellPatternGenerator(pattern_config, sparsity=0.15, noise_scale=0.2).generate(config.batch_size)
    dense_place = data.PlaceCellPatternGenerator(pattern_config, sparsity=0.30, noise_scale=0.2).generate(config.batch_size)

    patterns_comparison = {
        "Sparse Place (5%)": sparse_place,
        "Medium Place (15%)": medium_place,
        "Dense Place (30%)": dense_place,
    }

    print(f"  Comparing {len(patterns_comparison)} pattern types")

    fig7 = figures.plot_pattern_comparison(patterns_comparison, title="Sparsity Comparison: Place Cell Patterns", n_samples=3)
    if config.save_plots:
        fig7.savefig(config.output_dir / "07_pattern_comparison.png", dpi=150, bbox_inches="tight")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print(f"\n{'='*80}")
    print("Pattern Generation Summary")
    print(f"{'='*80}")
    print(f"\nGenerated patterns:")
    print(f"  Place cells: {sum(pattern_config.n_p)} total across {config.n_frequencies} frequencies")
    print(f"  Grid cells: {sum(pattern_config.n_g)} total across {config.n_frequencies} frequencies")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Temporal length: {config.walk_length}")
    print(f"\nVisualizations saved to: {config.output_dir}")
    print(f"  Total plots: 7")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")

    print(f"\n{'='*80}")
    print("Pattern generation examples complete!")
    print(f"{'='*80}")
