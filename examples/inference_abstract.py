#!/usr/bin/env python3
"""Abstract location inference example demonstrating precision-weighted fusion.

This example demonstrates the torch_tem.inference.AbstractLocationInference capabilities:
- Precision-weighted fusion of multiple information sources
- Transition-based prediction (g_gen) with uncertainty
- Memory-based inference (p_x → g_mem) via learned MLPs
- Salient object ("shiny") signals integration
- Scheduled memory influence via p2g_scale_offset
- Uncertainty estimation from memory quality indicators
- Source contribution analysis and visualization

The abstract location inference combines predictive dynamics with episodic memory
to produce a unified abstract location representation (g_inf) that supports both
structural generalization and sensory anchoring.

Usage:
    python examples/inference_abstract.py --n-timesteps 100 --n-frequencies 3
    python examples/inference_abstract.py --use-p-inf --use-shiny --show-plots
    python examples/inference_abstract.py --help
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
from torch_tem.inference.abstract import AbstractLocationInference


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for abstract location inference example.

    This config implements AbstractInferenceParams protocol for direct
    component instantiation.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference_abstract")

    # Simulation configuration
    n_timesteps: int = Field(default=100, ge=20, le=500, description="Number of timesteps to simulate")
    batch_size: int = Field(default=4, ge=1, le=16, description="Batch size for parallel trajectories")

    # Architecture configuration
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of hierarchical frequency modules")
    n_g_per_module: List[int] = Field(default_factory=lambda: [10, 8, 6], description="Abstract location dimensions per frequency")
    n_g_subsampled_per_freq: List[int] = Field(default_factory=lambda: [6, 5, 4], description="Downsampled dimensions (for memory path)")

    # Source configuration
    use_p_inf: bool = Field(default=True, description="Enable memory-based inference path")
    use_shiny: bool = Field(default=True, description="Enable salient object signals")

    # Uncertainty and scheduling
    transition_sigma_base: float = Field(default=0.5, ge=0.1, le=2.0, description="Base uncertainty for transition prediction")
    memory_sigma_base: float = Field(default=0.3, ge=0.1, le=2.0, description="Base uncertainty for memory inference")
    shiny_sigma_base: float = Field(default=0.2, ge=0.05, le=1.0, description="Base uncertainty for shiny signals")
    p2g_schedule_start: float = Field(default=2.0, ge=0.0, le=5.0, description="Initial p2g scale offset (high = low memory influence)")
    p2g_schedule_end: float = Field(default=0.1, ge=0.0, le=1.0, description="Final p2g scale offset (low = high memory influence)")

    # Network initialization
    g_mem_std: float = Field(default=0.01, gt=0, description="Std for memory MLP weight initialization")
    g_init_std: float = Field(default=0.1, gt=0, description="Std for learnable g_init parameters")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference_abstract"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("n_g_per_module", "n_g_subsampled_per_freq")
    @classmethod
    def validate_list_length(cls, v: List[int], info) -> List[int]:
        """Ensure lists match n_frequencies if provided."""
        # During initialization, n_frequencies might not be set yet
        return v

    # ==============================================================================
    # AbstractInferenceParams Protocol Implementation
    # ==============================================================================

    @computed_field(description="Total number of frequency modules")
    @property
    def n_f_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Abstract location dimensions per frequency")
    @property
    def n_g_calculated(self) -> List[int]:
        # Adjust list length to match n_frequencies
        if len(self.n_g_per_module) < self.n_frequencies:
            # Extend by repeating last value
            return self.n_g_per_module + [self.n_g_per_module[-1]] * (self.n_frequencies - len(self.n_g_per_module))
        return self.n_g_per_module[: self.n_frequencies]

    @computed_field(description="Downsampled abstract dimensions per frequency")
    @property
    def n_g_subsampled_combined(self) -> List[int]:
        # Adjust list length to match n_frequencies
        if len(self.n_g_subsampled_per_freq) < self.n_frequencies:
            return self.n_g_subsampled_per_freq + [self.n_g_subsampled_per_freq[-1]] * (self.n_frequencies - len(self.n_g_subsampled_per_freq))
        return self.n_g_subsampled_per_freq[: self.n_frequencies]


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the abstract location inference experiment with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Abstract Location Inference Example: Precision-Weighted Fusion")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Timesteps: {config.n_timesteps}, Batch size: {config.batch_size}")
    print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Abstract dims per freq: {config.n_g_calculated}")
    print(f"  Downsampled dims per freq: {config.n_g_subsampled_combined}")
    print(f"  Memory path (use_p_inf): {config.use_p_inf}")
    print(f"  Shiny signals (use_shiny): {config.use_shiny}")
    print(f"  p2g schedule: {config.p2g_schedule_start:.2f} → {config.p2g_schedule_end:.2f}")
    print()

    # =========================================================================
    # PHASE 1: Initialize Abstract Inference Module
    # =========================================================================
    print("Initializing AbstractLocationInference module...")
    model = AbstractLocationInference(config)
    print(f"  Module initialized with {config.n_frequencies} frequency modules")
    print(f"  Memory path enabled: {config.use_p_inf}")
    print()

    # =========================================================================
    # PHASE 2: Generate Synthetic Data
    # =========================================================================
    print("Generating synthetic source signals...")

    # Transition predictions
    trans_gen = data.TransitionPredictionGenerator(config)
    g_gen_history, sigma_gen_history = trans_gen.generate()
    print(f"  ✓ Transition predictions: {len(g_gen_history)} timesteps")

    # Memory signals
    if config.use_p_inf:
        mem_gen = data.MemorySignalGenerator(config)
        p_x_history = mem_gen.generate()
    else:
        p_x_history = [None] * config.n_timesteps
    print(f"  ✓ Memory signals: {'enabled' if config.use_p_inf else 'disabled'}")

    # Shiny signals
    if config.use_shiny:
        shiny_gen = data.ShinySignalGenerator(config)
        mu_shiny_history, sigma_shiny_history = shiny_gen.generate()
    else:
        mu_shiny_history = [None] * config.n_timesteps
        sigma_shiny_history = [None] * config.n_timesteps
    n_shiny_active = sum(1 for x in mu_shiny_history if x is not None)
    print(f"  ✓ Shiny signals: {n_shiny_active}/{config.n_timesteps} timesteps active")

    # p2g schedule (linear decay)
    p2g_schedule = np.linspace(config.p2g_schedule_start, config.p2g_schedule_end, config.n_timesteps)
    print(f"  ✓ p2g schedule: {p2g_schedule[0]:.2f} → {p2g_schedule[-1]:.2f}")
    print()

    # =========================================================================
    # PHASE 3: Run Inference Over Time
    # =========================================================================
    print("Running precision-weighted fusion over trajectory...")

    g_inf_history = []
    precisions_history = []
    sigma_history_dict = {"transition": sigma_gen_history, "memory": [], "shiny": []}

    with torch.no_grad():
        for t in range(config.n_timesteps):
            # Prepare inputs
            g_gen = g_gen_history[t]
            sigma_gen = sigma_gen_history[t]
            p_x = p_x_history[t] if config.use_p_inf else None
            shiny = (mu_shiny_history[t], sigma_shiny_history[t]) if config.use_shiny and mu_shiny_history[t] is not None else None
            offset = p2g_schedule[t]

            # Forward pass
            g_inf = model(g_gen, sigma_gen, p_x, shiny, offset)
            g_inf_history.append(g_inf)

            # Extract precisions for analysis
            precisions = {"transition": [1.0 / (sigma_gen[f] ** 2 + 1e-8) for f in range(config.n_frequencies)]}

            if config.use_p_inf and p_x is not None:
                # Approximate memory uncertainty (would need to run through model internals)
                sigma_mem_approx = [torch.ones_like(g_gen[f]) * (config.memory_sigma_base + offset) for f in range(config.n_frequencies)]
                precisions["memory"] = [1.0 / (sigma_mem_approx[f] ** 2 + 1e-8) for f in range(config.n_frequencies)]
                sigma_history_dict["memory"].append(sigma_mem_approx)
            else:
                sigma_history_dict["memory"].append(None)

            if shiny is not None:
                sigma_shiny = shiny[1]
                precisions["shiny"] = [1.0 / (sigma_shiny[f] ** 2 + 1e-8) for f in range(config.n_frequencies)]
                sigma_history_dict["shiny"].append(sigma_shiny)
            else:
                sigma_history_dict["shiny"].append(None)

            precisions_history.append(precisions)

    print(f"  ✓ Inference completed for {config.n_timesteps} timesteps")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Generating visualizations...")

    # Select representative timesteps for source contribution plots
    timesteps_to_plot = [0, config.n_timesteps // 4, config.n_timesteps // 2, 3 * config.n_timesteps // 4, config.n_timesteps - 1]

    # Plot 1: Source precision contributions
    fig1 = figures.plot_source_contributions(precisions_history, timesteps_to_plot, config.n_frequencies)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_source_contributions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_source_contributions.png'}")

    # Plot 2: Uncertainty evolution
    fig2 = figures.plot_uncertainty_evolution(sigma_history_dict, config.n_frequencies)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_uncertainty_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_uncertainty_evolution.png'}")

    # Plot 3: g_inf evolution
    fig3 = figures.plot_g_inf_evolution(g_inf_history, config.n_frequencies, config.n_timesteps)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_g_inf_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_g_inf_evolution.png'}")

    # Plot 4: p2g schedule
    fig4 = figures.plot_schedule_effect(p2g_schedule)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_schedule_effect.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '04_schedule_effect.png'}")

    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
