"""Abstract location inference visualization functions.

Provides plotting utilities for abstract location inference components including:
- Source precision contributions over time
- Uncertainty evolution across sources
- Inferred abstract location dynamics
- Memory influence scheduling effects

All functions use Protocol-based typing for flexibility and testability.
Each function returns a matplotlib Figure object for flexible display/saving.
"""

from typing import Dict, List, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..types import AbstractLocation, Vector


# ==============================================================================
# Protocols
# ==============================================================================
class AbstractInferenceConfig(Protocol):
    """Minimal configuration interface for abstract inference visualization."""

    n_timesteps: int
    n_frequencies: int
    n_g: List[int]


# ==============================================================================
# Source Contribution Analysis
# ==============================================================================
def plot_source_contributions(
    precisions_history: List[Dict[str, AbstractLocation]],
    timesteps_to_plot: List[int],
    n_frequencies: int,
    title: str = "Source Precision Contributions Over Time",
    figsize: tuple = (14, 4),
) -> plt.Figure:
    """Plot precision contributions from each source over selected timesteps.

    Shows how much each source (transition, memory, shiny) contributes to the
    final inference based on their relative precisions. Displays as percentage
    bar charts per frequency module.

    Args:
        precisions_history: List of dicts mapping source names to precision tensors
                           per frequency [n_f] of [B, n_g[f]]
        timesteps_to_plot: Timestep indices to visualize (typically 5-7 points)
        n_frequencies: Number of frequency modules
        title: Figure title
        figsize: Base figure size (width, height per frequency)

    Returns:
        matplotlib Figure with bar charts showing contribution percentages

    Example:
        >>> timesteps = [0, T//4, T//2, 3*T//4, T-1]
        >>> fig = plot_source_contributions(precisions_history, timesteps, n_f=3)
        >>> fig.savefig('source_contributions.png')
    """
    n_f = n_frequencies
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

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# Uncertainty Evolution
# ==============================================================================
def plot_uncertainty_evolution(
    sigma_history: Dict[str, List[AbstractLocation]],
    n_frequencies: int,
    title: str = "Uncertainty Evolution by Source",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot uncertainty (sigma) evolution for each source over time.

    Shows how confidence in each source varies across the trajectory,
    with separate panels per frequency module.

    Args:
        sigma_history: Dict mapping source names to lists of sigma tensors
                      per timestep [T] containing [n_f] of [B, n_g[f]]
        n_frequencies: Number of frequency modules
        title: Figure title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with line plots of uncertainty traces

    Example:
        >>> sigma_dict = {"transition": sigma_gen_list, "memory": sigma_mem_list}
        >>> fig = plot_uncertainty_evolution(sigma_dict, n_f=3)
        >>> fig.savefig('uncertainty_evolution.png')
    """
    # Determine timesteps from first non-None source
    T = len(next(iter(sigma_history.values())))
    n_f = n_frequencies

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
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# Abstract Location Evolution
# ==============================================================================
def plot_g_inf_evolution(
    g_inf_history: List[AbstractLocation],
    n_frequencies: int,
    n_dims_to_plot: int = 3,
    batch_idx: int = 0,
    title: str = "Inferred Abstract Location (g_inf) Evolution",
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot inferred abstract location g_inf over time.

    Shows the temporal evolution of the fused abstract representation,
    visualizing the first few dimensions per frequency module.

    Args:
        g_inf_history: List of timesteps [T], each containing [n_f] tensors [B, n_g[f]]
        n_frequencies: Number of frequency modules
        n_dims_to_plot: Number of dimensions to plot per frequency (default: 3)
        batch_idx: Which batch trajectory to visualize (default: 0)
        title: Figure title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with line plots of g_inf dimensions

    Example:
        >>> fig = plot_g_inf_evolution(g_inf_history, n_f=3, n_dims_to_plot=3)
        >>> fig.savefig('g_inf_evolution.png')
    """
    T = len(g_inf_history)
    n_f = n_frequencies

    fig, axes = plt.subplots(n_f, 1, figsize=figsize, sharex=True)
    if n_f == 1:
        axes = [axes]

    for freq_idx in range(n_f):
        ax = axes[freq_idx]

        # Stack into [T, B, n_g[f]] and take first n_dims dimensions
        g_inf_tensor = torch.stack([g_inf_history[t][freq_idx] for t in range(T)])  # [T, B, n_g]
        n_dims = min(n_dims_to_plot, g_inf_tensor.shape[2])

        for dim in range(n_dims):
            # Plot selected batch trajectory
            g_vals = g_inf_tensor[:, batch_idx, dim].detach().cpu().numpy()
            ax.plot(range(T), g_vals, label=f"dim {dim}", linewidth=1.5, alpha=0.8)

        ax.set_ylabel(f"Freq {freq_idx}\ng_inf", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, ncol=n_dims)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Timestep", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# Memory Influence Scheduling
# ==============================================================================
def plot_schedule_effect(
    p2g_schedule: List[float],
    title: str = "Memory Influence Schedule\n(higher offset = lower memory influence)",
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot p2g schedule showing how memory influence changes over time.

    Visualizes the curriculum learning schedule for memory pathway contribution,
    typically starting high (low influence) and decaying to low (high influence).

    Args:
        p2g_schedule: List of p2g_scale_offset values per timestep
        title: Figure title with explanation
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with schedule line plot

    Example:
        >>> schedule = np.linspace(2.0, 0.1, n_timesteps)
        >>> fig = plot_schedule_effect(schedule)
        >>> fig.savefig('schedule_effect.png')
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.plot(range(len(p2g_schedule)), p2g_schedule, linewidth=2, color="darkorange", marker="o", markersize=3, alpha=0.8)
    ax.set_xlabel("Timestep", fontsize=11)
    ax.set_ylabel("p2g_scale_offset", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)

    # Annotate start and end
    ax.text(0, p2g_schedule[0] + 0.1, f"Start: {p2g_schedule[0]:.2f}", ha="left", fontsize=9, color="darkred")
    ax.text(len(p2g_schedule) - 1, p2g_schedule[-1] + 0.1, f"End: {p2g_schedule[-1]:.2f}", ha="right", fontsize=9, color="darkgreen")

    plt.tight_layout()
    return fig
