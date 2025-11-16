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

from torch_tem import utils
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
# Synthetic Data Generation
# ==============================================================================
def generate_transition_predictions(config: ExampleConfig) -> tuple[List[List[Tensor]], List[List[Tensor]]]:
    """Generate synthetic transition predictions g_gen with uncertainty sigma_g_gen.

    Simulates predictive dynamics with slowly varying means and uncertainty.

    Returns:
        (g_gen_history, sigma_gen_history): Lists of length T, each containing
        lists of n_f tensors [B, n_g[f]]
    """
    T = config.n_timesteps
    B = config.batch_size
    n_f = config.n_frequencies
    n_g = config.n_g_calculated

    g_gen_history = []
    sigma_gen_history = []

    # Initialize with random state
    g_prev = [torch.randn(B, n_g[f]) for f in range(n_f)]

    for t in range(T):
        # Slowly evolving transition prediction (Ornstein-Uhlenbeck-like)
        g_gen = [0.95 * g_prev[f] + 0.05 * torch.randn(B, n_g[f]) for f in range(n_f)]

        # Uncertainty that varies over time (simulate varying prediction confidence)
        sigma_gen = [config.transition_sigma_base * (1.0 + 0.3 * torch.sin(torch.tensor(t / 20.0))) * torch.ones(B, n_g[f]) for f in range(n_f)]

        g_gen_history.append(g_gen)
        sigma_gen_history.append(sigma_gen)
        g_prev = g_gen

    return g_gen_history, sigma_gen_history


def generate_memory_signals(config: ExampleConfig) -> List[List[Tensor]]:
    """Generate synthetic memory-based signals p_x.

    Simulates memory retrieval patterns that could come from attractor dynamics.

    Returns:
        p_x_history: List of length T, each containing lists of n_f tensors [B, n_g_sub[f]]
    """
    T = config.n_timesteps
    B = config.batch_size
    n_f = config.n_frequencies
    n_g_sub = config.n_g_subsampled_combined

    p_x_history = []

    for t in range(T):
        # Memory patterns with temporal correlation
        p_x = [torch.randn(B, n_g_sub[f]) * (0.8 + 0.4 * np.sin(t / 15.0)) for f in range(n_f)]
        p_x_history.append(p_x)

    return p_x_history


def generate_shiny_signals(config: ExampleConfig) -> tuple[List[List[Tensor]], List[List[Tensor]]]:
    """Generate synthetic salient object signals with low uncertainty.

    Simulates strong localization cues that appear intermittently.

    Returns:
        (mu_shiny_history, sigma_shiny_history): Lists of length T, each containing
        lists of n_f tensors [B, n_g[f]] or None if no signal at that timestep
    """
    T = config.n_timesteps
    B = config.batch_size
    n_f = config.n_frequencies
    n_g = config.n_g_calculated

    mu_shiny_history = []
    sigma_shiny_history = []

    # Shiny signals appear intermittently (every ~20 timesteps)
    shiny_period = 20
    shiny_duration = 5

    for t in range(T):
        if (t % shiny_period) < shiny_duration:
            # Strong signal with low uncertainty
            mu_shiny = [torch.randn(B, n_g[f]) * 2.0 for f in range(n_f)]  # Stronger signal
            sigma_shiny = [config.shiny_sigma_base * torch.ones(B, n_g[f]) for f in range(n_f)]
            mu_shiny_history.append(mu_shiny)
            sigma_shiny_history.append(sigma_shiny)
        else:
            mu_shiny_history.append(None)
            sigma_shiny_history.append(None)

    return mu_shiny_history, sigma_shiny_history


# ==============================================================================
# Visualization Functions
# ==============================================================================
def plot_source_contributions(precisions_history: List[dict[str, List[Tensor]]], timesteps_to_plot: List[int], config: ExampleConfig, figsize: tuple = (14, 4)) -> plt.Figure:
    """Plot precision contributions from each source over selected timesteps.

    Shows how much each source (transition, memory, shiny) contributes to the
    final inference based on their relative precisions.
    """
    n_f = config.n_frequencies
    n_t = len(timesteps_to_plot)

    fig, axes = plt.subplots(n_f, n_t, figsize=(figsize[0], figsize[1] * n_f), squeeze=False)

    for freq_idx in range(n_f):
        for t_idx, t in enumerate(timesteps_to_plot):
            ax = axes[freq_idx, t_idx]

            # Extract precisions for this timestep and frequency
            precs = precisions_history[t]
            source_names = list(precs.keys())
            source_values = [precs[name][freq_idx].mean().item() for name in source_names]

            # Normalize to percentages
            total = sum(source_values)
            percentages = [100 * v / total if total > 0 else 0 for v in source_values]

            # Bar plot
            colors = ["steelblue", "darkorange", "green"][: len(source_names)]
            bars = ax.bar(source_names, percentages, color=colors, alpha=0.7, edgecolor="black")

            # Annotate with percentages
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{pct:.1f}%", ha="center", va="bottom", fontsize=9)

            ax.set_ylim(0, 105)
            ax.set_ylabel("Contribution (%)", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

            if freq_idx == 0:
                ax.set_title(f"t={t}", fontsize=11, fontweight="bold")
            if freq_idx == n_f - 1:
                ax.set_xlabel("Source", fontsize=10)
            if t_idx == 0:
                ax.set_ylabel(f"Freq {freq_idx}\nContribution (%)", fontsize=10)

    fig.suptitle("Source Precision Contributions Over Time", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_uncertainty_evolution(sigma_history: dict[str, List[List[Tensor]]], config: ExampleConfig, figsize: tuple = (12, 6)) -> plt.Figure:
    """Plot uncertainty (sigma) evolution for each source over time.

    Shows how confidence in each source varies across the trajectory.
    """
    T = config.n_timesteps
    n_f = config.n_frequencies

    fig, axes = plt.subplots(n_f, 1, figsize=figsize, sharex=True)
    if n_f == 1:
        axes = [axes]

    for freq_idx in range(n_f):
        ax = axes[freq_idx]

        for source_name, sigma_list in sigma_history.items():
            # Extract mean sigma over batch for this frequency
            sigma_vals = [sigma_list[t][freq_idx].mean().item() if sigma_list[t] is not None else np.nan for t in range(T)]

            ax.plot(range(T), sigma_vals, label=source_name, linewidth=2, alpha=0.8)

        ax.set_ylabel(f"Freq {freq_idx}\nσ (uncertainty)", fontsize=10)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle("Uncertainty Evolution by Source", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_g_inf_evolution(g_inf_history: List[List[Tensor]], config: ExampleConfig, figsize: tuple = (12, 6)) -> plt.Figure:
    """Plot inferred abstract location g_inf over time.

    Shows the temporal evolution of the fused abstract representation.
    """
    T = config.n_timesteps
    B = config.batch_size
    n_f = config.n_frequencies

    fig, axes = plt.subplots(n_f, 1, figsize=figsize, sharex=True)
    if n_f == 1:
        axes = [axes]

    for freq_idx in range(n_f):
        ax = axes[freq_idx]

        # Stack into [T, B, n_g[f]] and take first 3 dimensions for visualization
        g_inf_tensor = torch.stack([g_inf_history[t][freq_idx] for t in range(T)])  # [T, B, n_g]
        n_dims_to_plot = min(3, g_inf_tensor.shape[2])

        for dim in range(n_dims_to_plot):
            # Plot first batch trajectory
            g_vals = g_inf_tensor[:, 0, dim].detach().cpu().numpy()
            ax.plot(range(T), g_vals, label=f"dim {dim}", linewidth=1.5, alpha=0.8)

        ax.set_ylabel(f"Freq {freq_idx}\ng_inf", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=n_dims_to_plot)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle("Inferred Abstract Location (g_inf) Evolution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_schedule_effect(p2g_schedule: List[float], config: ExampleConfig, figsize: tuple = (10, 4)) -> plt.Figure:
    """Plot p2g schedule showing how memory influence changes over time."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(range(len(p2g_schedule)), p2g_schedule, linewidth=2, color="darkorange", marker="o", markersize=3, alpha=0.8)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("p2g_scale_offset", fontsize=11)
    ax.set_title("Memory Influence Schedule\n(higher offset = lower memory influence)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)

    # Annotate start and end
    ax.text(0, p2g_schedule[0] + 0.1, f"Start: {p2g_schedule[0]:.2f}", ha="left", fontsize=9, color="darkred")
    ax.text(len(p2g_schedule) - 1, p2g_schedule[-1] + 0.1, f"End: {p2g_schedule[-1]:.2f}", ha="right", fontsize=9, color="darkgreen")

    plt.tight_layout()
    return fig


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
    g_gen_history, sigma_gen_history = generate_transition_predictions(config)
    print(f"  ✓ Transition predictions: {len(g_gen_history)} timesteps")

    # Memory signals
    p_x_history = generate_memory_signals(config) if config.use_p_inf else [None] * config.n_timesteps
    print(f"  ✓ Memory signals: {'enabled' if config.use_p_inf else 'disabled'}")

    # Shiny signals
    mu_shiny_history, sigma_shiny_history = generate_shiny_signals(config) if config.use_shiny else ([None] * config.n_timesteps, [None] * config.n_timesteps)
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
    fig1 = plot_source_contributions(precisions_history, timesteps_to_plot, config)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_source_contributions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_source_contributions.png'}")

    # Plot 2: Uncertainty evolution
    fig2 = plot_uncertainty_evolution(sigma_history_dict, config)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_uncertainty_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_uncertainty_evolution.png'}")

    # Plot 3: g_inf evolution
    fig3 = plot_g_inf_evolution(g_inf_history, config)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_g_inf_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_g_inf_evolution.png'}")

    # Plot 4: p2g schedule
    fig4 = plot_schedule_effect(p2g_schedule, config)
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
