"""Simulation visualization for the Tolman-Eichenbaum Machine (TEM).

This module provides visualization functions for TEM simulation state evolution:
- Abstract location (grid cell) activity heatmaps over time
- Grounded location (place cell) activity heatmaps over time
- Memory formation timeline showing Hebbian updates across timesteps
- Prediction accuracy metrics and error evolution

All functions follow TEM visualization conventions:
- Return matplotlib Figure objects (no plt.show() or plt.savefig())
- Accept standard Python types (lists, tensors)
- Use consistent styling and color schemes
- Provide customizable parameters with sensible defaults
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from torch import Tensor


def plot_abstract_location_heatmap(
    abstract_locations: List[List[Tensor]],
    n_modules_to_plot: int = 2,
    title: str = "Abstract Location Evolution (Grid Cells)",
    figsize: Tuple[float, float] = (14, 5),
    cmap: str = "viridis",
) -> Figure:
    """Plot temporal evolution of abstract location (grid cell) activations as heatmaps.

    Displays grid cell activation patterns over the course of a simulation, showing
    how abstract spatial representations evolve across timesteps. Each frequency
    module is plotted separately to reveal multi-scale coding.

    Args:
        abstract_locations: List of abstract locations per timestep.
            Each element is a list of tensors (one per frequency module).
            Shape per module: [batch_size, n_g[f]]
        n_modules_to_plot: Number of frequency modules to display (default: 2).
        title: Plot title.
        figsize: Figure size as (width, height).
        cmap: Colormap for heatmaps (default: 'viridis').

    Returns:
        Figure containing heatmaps (one per frequency module).

    Example:
        >>> # During simulation, collect states (extract batch dimension [0])
        >>> abstract_locs = []
        >>> for state in simulation:
        ...     abstract_locs.append([g[0].detach().cpu() for g in state.abstract_location])
        >>> # Visualize evolution
        >>> fig = plot_abstract_location_heatmap(abstract_locs)
    """
    n_freq = len(abstract_locations[0])
    n_modules = min(n_modules_to_plot, n_freq)

    fig, axes = plt.subplots(1, n_modules, figsize=figsize)
    if n_modules == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for f in range(n_modules):
        ax = axes[f]
        # Stack abstract locations over time [T, n_g[f]]
        g_history = torch.stack([g[f] for g in abstract_locations])  # [T, n_g[f]]
        im = ax.imshow(g_history.T, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(f"Module {f} (Frequency {f})")
        ax.set_xlabel("Timestep")
        ax.set_ylabel(f"Grid Cell Index")
        plt.colorbar(im, ax=ax, label="Activation")

    plt.tight_layout()
    return fig


def plot_memory_formation_timeline(
    memory_snapshots: List[Tuple[int, Tensor, Tensor]],
    title: str = "Memory Formation Timeline (Hebbian Updates)",
    figsize: Optional[Tuple[float, float]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
) -> Figure:
    """Plot memory matrix development over time showing Hebbian learning progression.

    Displays snapshots of generative and inference memory matrices at different
    timesteps to visualize how Hebbian learning shapes the memory structure during
    a simulation. Shows the temporal evolution of place cell associations.

    Args:
        memory_snapshots: List of (timestep, M_gen, M_inf) tuples.
            M_gen and M_inf are tensors of shape [batch_size, sum(n_p), sum(n_p)].
        title: Plot title.
        figsize: Figure size as (width, height). If None, auto-computed based on snapshots.
        vmin: Minimum value for colormap normalization.
        vmax: Maximum value for colormap normalization.
        cmap: Colormap for memory matrices (default: 'RdBu_r' for diverging).

    Returns:
        Figure with 2 rows (generative/inference) × N columns (snapshots).

    Example:
        >>> # During simulation, capture memory snapshots
        >>> snapshots = []
        >>> for t in [0, 25, 50, 75, 100]:
        ...     M_gen = model.memory.storage.M_gen.detach().cpu()
        ...     M_inf = model.memory.storage.M_inf.detach().cpu()
        ...     snapshots.append((t, M_gen, M_inf))
        >>> # Visualize formation
        >>> fig = plot_memory_formation_timeline(snapshots)
    """
    n_snapshots = len(memory_snapshots)
    if figsize is None:
        figsize = (4 * n_snapshots, 8)

    fig, axes = plt.subplots(2, n_snapshots, figsize=figsize)
    if n_snapshots == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, (t, M_gen, M_inf) in enumerate(memory_snapshots):
        # Generative memory (top row)
        im = axes[0, i].imshow(M_gen[0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"M_gen at t={t}")
        axes[0, i].set_xlabel("Pre-synaptic")
        axes[0, i].set_ylabel("Post-synaptic")
        if i == n_snapshots - 1:
            plt.colorbar(im, ax=axes[0, i], label="Weight")

        # Inference memory (bottom row)
        im = axes[1, i].imshow(M_inf[0].numpy(), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"M_inf at t={t}")
        axes[1, i].set_xlabel("Pre-synaptic")
        axes[1, i].set_ylabel("Post-synaptic")
        if i == n_snapshots - 1:
            plt.colorbar(im, ax=axes[1, i], label="Weight")

    plt.tight_layout()
    return fig


def plot_prediction_accuracy(
    predictions: List[Optional[List[Tensor]]],
    ground_truth: List[Tensor],
    title: str = "Sensory Prediction Accuracy",
    figsize: Tuple[float, float] = (12, 6),
) -> Figure:
    """Plot sensory prediction accuracy over time.

    Compares model predictions against ground truth observations to visualize
    how prediction quality evolves during simulation. Useful for evaluating
    the generative model's accuracy.

    Args:
        predictions: List of predictions per timestep.
            Each element is either None or a list of tensors per frequency module.
        ground_truth: List of ground truth observations.
            Each element is a tensor of shape [n_x].
        title: Plot title.
        figsize: Figure size as (width, height).

    Returns:
        Figure showing prediction error metrics over time.

    Example:
        >>> # During simulation
        >>> preds = []
        >>> for state in simulation:
        ...     if state.prediction:
        ...         preds.append([v.detach().cpu() for v in state.prediction.values])
        ...     else:
        ...         preds.append(None)
        >>> # Visualize accuracy
        >>> fig = plot_prediction_accuracy(preds, walk.observations)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Filter valid predictions
    valid_indices = [i for i, p in enumerate(predictions) if p is not None]
    if not valid_indices:
        axes[0].text(0.5, 0.5, "No predictions available", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No predictions available", ha="center", va="center", transform=axes[1].transAxes)
        return fig

    # Compute prediction errors
    errors = []
    for i in valid_indices:
        pred = predictions[i][0][0]  # First frequency module, first batch element
        truth = ground_truth[i]
        error = torch.nn.functional.mse_loss(pred, truth).item()
        errors.append(error)

    # Plot 1: Error over time
    axes[0].plot(valid_indices, errors, linewidth=2, color="steelblue")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Prediction Error Over Time")
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Error distribution
    axes[1].hist(errors, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].axvline(np.mean(errors), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(errors):.4f}")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Error Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
