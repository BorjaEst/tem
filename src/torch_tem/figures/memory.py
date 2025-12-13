"""Memory system visualization for the Tolman-Eichenbaum Machine (TEM).

This module provides visualization functions for TEM's Hebbian memory system:
- Memory matrices with hierarchical frequency structure
- Attractor dynamics convergence analysis
- Learning curves and memory evolution
- Hierarchical mask schedules
- Retrieval quality and robustness analysis

All functions follow TEM visualization conventions:
- Return matplotlib Figure objects (no plt.show() or plt.savefig())
- Accept Protocol-based interfaces for loose coupling
- Use consistent styling and color schemes
- Provide customizable parameters with sensible defaults
"""

from typing import Dict, List, Optional, Protocol

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..types import Matrix, Vector

# =======================================================================================
# PROTOCOL INTERFACES
# =======================================================================================


class MemoryParams(Protocol):
    """Minimal interface for memory configuration."""

    n_p: List[int]
    """Place cell counts per frequency module."""


# =======================================================================================
# MEMORY MATRIX VISUALIZATION
# =======================================================================================


def plot_memory_matrices(
    M_gen: Matrix,
    M_inf: Optional[Matrix] = None,
    n_p_per_freq: Optional[List[int]] = None,
    n_training_steps: Optional[int] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 5),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Visualize learned Hebbian memory matrices with frequency structure.

    Displays generative and/or inference memory matrices as heatmaps, with
    frequency module boundaries overlaid. The hierarchical structure shows
    how place cells at different spatial scales (frequencies) are connected
    via Hebbian learning.

    Args:
        M_gen: Generative memory matrix [sum(n_p), sum(n_p)]
        M_inf: Optional inference memory matrix [sum(n_p), sum(n_p)]
               If None, only M_gen is plotted
        n_p_per_freq: Place cell counts per frequency for boundary visualization
                      If None, no boundaries are drawn
        n_training_steps: Number of training steps (for title annotation)
        title: Optional custom title (overrides default)
        figsize: Figure size (width, height)
        cmap: Colormap name (default: "RdBu_r" for diverging red-blue)

    Returns:
        matplotlib Figure with memory matrix visualization(s)

    Example:
        >>> storage = MemoryStorage(params)
        >>> # ... training ...
        >>> fig = plot_memory_matrices(
        ...     storage.M_gen,
        ...     storage.M_inf,
        ...     n_p_per_freq=params.n_p,
        ...     n_training_steps=100
        ... )
        >>> fig.savefig('memory_matrices.png')
    """
    # Determine layout based on whether dual memory is used
    n_plots = 2 if M_inf is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Plot generative memory
    im1 = axes[0].imshow(M_gen.detach().cpu().numpy(), cmap=cmap, aspect="auto")
    axes[0].set_title("Generative Memory (M_gen)")
    axes[0].set_xlabel("To (Place Cell Index)")
    axes[0].set_ylabel("From (Place Cell Index)")
    plt.colorbar(im1, ax=axes[0], label="Connection Strength")

    # Add frequency boundaries if provided
    if n_p_per_freq is not None:
        boundaries = [0] + [sum(n_p_per_freq[: i + 1]) for i in range(len(n_p_per_freq))]
        for boundary in boundaries:
            axes[0].axhline(boundary, color="black", linestyle="--", alpha=0.3)
            axes[0].axvline(boundary, color="black", linestyle="--", alpha=0.3)

    # Plot inference memory if provided
    if M_inf is not None:
        im2 = axes[1].imshow(M_inf.detach().cpu().numpy(), cmap=cmap, aspect="auto")
        axes[1].set_title("Inference Memory (M_inf)")
        axes[1].set_xlabel("To (Place Cell Index)")
        axes[1].set_ylabel("From (Place Cell Index)")
        plt.colorbar(im2, ax=axes[1], label="Connection Strength")

        if n_p_per_freq is not None:
            for boundary in boundaries:
                axes[1].axhline(boundary, color="black", linestyle="--", alpha=0.3)
                axes[1].axvline(boundary, color="black", linestyle="--", alpha=0.3)

    # Set overall title
    if title is not None:
        fig.suptitle(title)
    elif n_training_steps is not None:
        fig.suptitle(f"Learned Memory Matrices (after {n_training_steps} updates)")
    else:
        fig.suptitle("Learned Memory Matrices")

    plt.tight_layout()
    return fig


# =======================================================================================
# ATTRACTOR DYNAMICS CONVERGENCE
# =======================================================================================


def plot_attractor_convergence(
    queries: List[Vector],
    retrievals: List[Vector],
    targets: List[Vector],
    n_samples: Optional[int] = None,
    query_labels: Optional[List[str]] = None,
    n_p_per_freq: Optional[List[int]] = None,
    title: str = "Attractor Dynamics: Query → Retrieval Convergence",
    figsize: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> plt.Figure:
    """Visualize attractor convergence for multiple query patterns.

    Displays query (noisy), retrieved (after attractor), and target (ground truth)
    patterns as overlapping grouped bars in the same subplot for each test case.
    This allows direct visual comparison of how attractor dynamics refine noisy
    or partial queries toward stored patterns.

    Accepts batched MultiScaleCode (per-frequency tensors) and automatically handles
    per-sample extraction for visualization. When n_p_per_freq is provided, frequency
    module structure is visualized with background colors, boundary lines, and labels.

    Args:
        queries: Batched MultiScaleCode - List of [n_samples, n_p[f]] tensors (one per frequency)
        retrievals: Batched MultiScaleCode - List of [n_samples, n_p[f]] tensors
        targets: Batched MultiScaleCode - List of [n_samples, n_p[f]] tensors
        n_samples: Number of samples to plot (default: all samples in batch)
        query_labels: Optional labels for each query (e.g., ["Query 1", "Query 2"])
        n_p_per_freq: Place cell counts per frequency. Enables frequency visualization
        title: Figure title
        figsize: Figure size (width, height). If None, auto-sized based on n_queries
        ylim: Y-axis limits for all subplots. If None, auto-scaled based on data range

    Returns:
        matplotlib Figure with convergence visualization showing grouped bars for
        direct comparison of query, retrieval, and target patterns

    Example:
        >>> # Batched MultiScaleCode format
        >>> queries_list = split_to_frequencies(test_queries, n_p)  # List of [5, n_p[f]]
        >>> retrievals_list = attractor(queries_list, M)  # List of [5, n_p[f]]
        >>> targets_list = split_to_frequencies(test_targets, n_p)
        >>> fig = plot_attractor_convergence(
        ...     queries_list, retrievals_list, targets_list,
        ...     n_p_per_freq=params.n_p
        ... )
    """
    # Auto-detect number of samples from first frequency tensor
    if n_samples is None:
        n_samples = queries[0].shape[0]

    # Concatenate frequencies for plotting
    queries_cat = torch.cat(queries, dim=-1)  # [n_samples, sum(n_p)]
    retrievals_cat = torch.cat(retrievals, dim=-1)
    targets_cat = torch.cat(targets, dim=-1)

    # Convert to list of individual samples
    queries = [queries_cat[i] for i in range(n_samples)]
    retrievals = [retrievals_cat[i] for i in range(n_samples)]
    targets = [targets_cat[i] for i in range(n_samples)]

    n_queries = n_samples

    # Auto-compute y-axis limits if not provided
    # Use the 95th percentile to avoid outliers dominating the scale
    if ylim is None:
        all_values = []
        for q, r, t in zip(queries, retrievals, targets):
            all_values.extend([q.max().item(), r.max().item(), t.max().item()])
        y_max = np.percentile(all_values, 95)
        y_margin = y_max * 0.1  # 10% margin
        ylim = (0, y_max + y_margin)

    # Auto-size figure based on number of queries
    if figsize is None:
        figsize = (10, 4 * n_queries)

    fig, axes = plt.subplots(n_queries, 1, figsize=figsize)
    if n_queries == 1:
        axes = [axes]

    # Bar width and positioning for grouped bars
    n_cells = len(queries[0])
    x = np.arange(n_cells)
    width = 0.25  # Width of each bar

    # Prepare frequency visualization if n_p_per_freq is provided
    freq_colors = ["#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a", "#ef3b2c", "#cb181d"]
    if n_p_per_freq is not None:
        boundaries = [0] + [sum(n_p_per_freq[: i + 1]) for i in range(len(n_p_per_freq))]
        freq_centers = [(boundaries[i] + boundaries[i + 1]) / 2 for i in range(len(boundaries) - 1)]

    for i in range(n_queries):
        # Generate label
        label = query_labels[i] if query_labels is not None else f"{i+1}"

        # Convert tensors to numpy arrays
        query_vals = queries[i].cpu().numpy()
        retrieval_vals = retrievals[i].cpu().numpy()
        target_vals = targets[i].cpu().numpy()

        # Add frequency module background colors
        if n_p_per_freq is not None:
            for freq_idx in range(len(n_p_per_freq)):
                axes[i].axvspan(boundaries[freq_idx], boundaries[freq_idx + 1], alpha=0.15, color=freq_colors[freq_idx % len(freq_colors)], zorder=0)
            # Add vertical boundary lines
            for boundary in boundaries[1:-1]:  # Skip first and last
                axes[i].axvline(boundary, color="gray", linestyle="--", alpha=0.4, linewidth=1.5)

        # Plot all three patterns as grouped bars with offset positions
        axes[i].bar(x - width, query_vals, width, alpha=0.7, label="Query (Noisy)", color="C0", zorder=3)
        axes[i].bar(x, retrieval_vals, width, alpha=0.7, label="Retrieved", color="green", zorder=3)
        axes[i].bar(x + width, target_vals, width, alpha=0.7, label="Target (Ground Truth)", color="orange", zorder=3)

        axes[i].set_title(f"Pattern {label}: Query → Retrieval vs Target")

        # Enhanced x-axis labeling for frequency modules
        if n_p_per_freq is not None:
            axes[i].set_xlabel("Frequency Module → Place Cell Index")
            axes[i].set_xticks(freq_centers)
            axes[i].set_xticklabels([f"F{j+1}" for j in range(len(n_p_per_freq))])
            # Add secondary x-axis with actual indices
            ax2 = axes[i].twiny()
            ax2.set_xlim(axes[i].get_xlim())
            ax2.set_xticks(x[:: max(1, n_cells // 10)])
            ax2.set_xticklabels(x[:: max(1, n_cells // 10)], fontsize=8, alpha=0.6)
            ax2.set_xlabel("Cell Index", fontsize=8, alpha=0.6)
        else:
            axes[i].set_xlabel("Place Cell Index")
            axes[i].set_xticks(x[:: max(1, n_cells // 10)])  # Show ~10 tick labels max

        axes[i].set_ylabel("Activation")
        axes[i].set_ylim(ylim)
        axes[i].legend(loc="upper right")
        axes[i].grid(True, alpha=0.2, axis="y", zorder=1)

    fig.suptitle(title)
    plt.tight_layout()
    return fig


# =======================================================================================
# LEARNING CURVES
# =======================================================================================


def plot_learning_curve(
    memory_strengths: List[float],
    cosine_sims: Optional[List[float]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """Plot Hebbian memory learning progression over training.

    Displays two metrics:
    1. Memory strength (Frobenius norm): Overall magnitude of learned connections
    2. Memory divergence (cosine similarity): Difference between M_gen and M_inf

    Args:
        memory_strengths: List of memory strength values (one per training step)
                         Computed as torch.norm(M_gen)
        cosine_sims: Optional list of cosine similarities between M_gen and M_inf
                    If None, only memory strength is plotted
        title: Optional custom title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with learning curve visualization

    Example:
        >>> memory_strengths = []
        >>> cosine_sims = []
        >>> for step in range(n_steps):
        ...     storage.update(p_inf, p_gen, eta, lamb)
        ...     memory_strengths.append(torch.norm(storage.M_gen).item())
        ...     cosine_sims.append(cosine_sim(storage.M_gen, storage.M_inf))
        >>> fig = plot_learning_curve(memory_strengths, cosine_sims)
    """
    n_plots = 2 if cosine_sims is not None and len(cosine_sims) > 0 else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Memory strength (Frobenius norm)
    axes[0].plot(memory_strengths, marker="o", linewidth=2)
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Memory Strength (Frobenius Norm)")
    axes[0].set_title("Hebbian Learning: Memory Growth")
    axes[0].grid(True, alpha=0.3)

    # Cosine similarity between M_gen and M_inf (if dual memory)
    if n_plots == 2:
        axes[1].plot(cosine_sims, marker="s", linewidth=2, color="green")
        axes[1].set_xlabel("Training Step")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Memory Divergence: M_gen vs M_inf")
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    return fig


# =======================================================================================
# HIERARCHICAL MASK VISUALIZATION
# =======================================================================================


def plot_hierarchical_masks(
    masks_inf: List[Vector],
    masks_gen: Optional[List[Vector]] = None,
    n_p_per_freq: Optional[List[int]] = None,
    f_initial: Optional[List[float]] = None,
    title: str = "Hierarchical Mask Schedule",
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Visualize hierarchical mask schedules comparing inference vs generative modes.

    Shows how different frequency modules are progressively enabled during
    attractor dynamics. Compares two retrieval strategies:
    - Inference: All frequencies always active (conservative, stable)
    - Generative: Coarse-to-fine early-stopping (hierarchical refinement)

    Args:
        masks_inf: Inference mode masks [n_p_total], one per attractor iteration
        masks_gen: Optional generative mode masks [n_p_total]
                   If None, only inference mode is plotted
        n_p_per_freq: Place cell counts per frequency for boundary visualization
                     If None, no boundaries are drawn
        f_initial: Initial frequency values for annotation
                   If None, frequencies are not labeled
        title: Figure title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure with dual-mode mask schedule visualization

    Example:
        >>> attractor = AttractorDynamics(params)
        >>> fig = plot_hierarchical_masks(
        ...     attractor.p_retrieve_mask_inf,
        ...     attractor.p_retrieve_mask_gen,
        ...     n_p_per_freq=params.n_p,
        ...     f_initial=params.f_initial
        ... )
        >>> fig.savefig('hierarchical_masks.png')
    """
    # Determine layout based on whether generative masks are provided
    n_plots = 2 if masks_gen is not None else 1
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    def plot_mask_schedule(ax, masks, mode_title, n_p_per_freq, f_initial):
        """Helper to plot a single mask schedule as heatmap."""
        n_masks = len(masks)
        n_p_total = sum(n_p_per_freq) if n_p_per_freq else len(masks[0])

        # Create matrix: rows = iterations, cols = neurons
        mask_matrix = torch.stack([m for m in masks]).cpu().numpy()  # [i_attractor, n_p_total]

        # Plot heatmap
        im = ax.imshow(mask_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")

        # Add frequency boundaries and labels
        if n_p_per_freq is not None:
            n_p_cumsum = [0] + torch.cumsum(torch.tensor(n_p_per_freq), dim=0).tolist()

            # Draw vertical boundaries between frequencies
            for i, boundary in enumerate(n_p_cumsum[1:-1], 1):
                ax.axvline(boundary - 0.5, color="black", linewidth=2, linestyle="--", alpha=0.5)

            # Add frequency labels at top
            for f, n_p in enumerate(n_p_per_freq):
                center = n_p_cumsum[f] + n_p / 2
                if f_initial is not None and f < len(f_initial):
                    freq_val = f_initial[f]
                    label = f"f{f}\n(α={freq_val:.1f})\n{n_p} cells"
                else:
                    label = f"f{f}\n{n_p} cells"
                ax.text(center, -0.5, label, ha="center", va="top", fontsize=9, fontweight="bold")

        ax.set_xlabel("Place Cell Index (grouped by frequency)", fontsize=11)
        ax.set_ylabel("Attractor Iteration (τ)", fontsize=11)
        ax.set_title(mode_title, fontsize=12, fontweight="bold", pad=20)
        ax.set_yticks(range(n_masks))
        ax.set_yticklabels([f"τ={i}" for i in range(n_masks)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="Active (1.0) / Frozen (0.0)")

        # Add summary: active neurons per iteration
        active_per_iter = mask_matrix.sum(axis=1)
        for i in range(n_masks):
            ax.text(n_p_total + 5, i, f"{int(active_per_iter[i])}/{n_p_total}", va="center", fontsize=9, color="navy", fontweight="bold")

    # Plot inference mode
    inf_title = f"Inference Mode: All {len(n_p_per_freq) if n_p_per_freq else 'N'} frequencies always active (stable)"
    plot_mask_schedule(axes[0], masks_inf, inf_title, n_p_per_freq, f_initial)

    # Plot generative mode if provided
    if masks_gen is not None:
        gen_title = "Generative Mode: Hierarchical early-stopping (coarse→fine refinement)"
        plot_mask_schedule(axes[1], masks_gen, gen_title, n_p_per_freq, f_initial)

        # Add explanation text for dual-mode comparison
        fig.text(
            0.5,
            0.02,
            "Hierarchical masking: Coarse frequencies (low f) converge first, providing stable foundation for fine details (high f)",
            ha="center",
            fontsize=10,
            style="italic",
            wrap=True,
        )
        fig.tight_layout(rect=[0, 0.03, 1, 1])
    else:
        fig.tight_layout()

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    return fig


# =======================================================================================
# RETRIEVAL QUALITY ANALYSIS
# =======================================================================================


def plot_retrieval_quality(
    errors_by_mode: Dict[str, List[float]],
    noise_levels: List[float],
    title: str = "Attractor Dynamics: Robustness to Noisy Queries",
    figsize: tuple = (10, 6),
    xlabel: str = "Query Noise Level",
    ylabel: str = "Retrieval Error (MSE)",
) -> plt.Figure:
    """Plot retrieval error vs. noise level for different memory modes.

    Analyzes how well attractor dynamics handle queries with varying amounts
    of noise. Typically compares inference vs. generative memory retrieval.

    Args:
        errors_by_mode: Dict mapping mode names to error lists
                       e.g., {"Inference": [0.01, 0.02, ...], "Generative": [...]}
        noise_levels: List of noise levels tested (x-axis values)
        title: Figure title
        figsize: Figure size (width, height)
        xlabel: X-axis label
        ylabel: Y-axis label

    Returns:
        matplotlib Figure with retrieval quality visualization

    Example:
        >>> noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
        >>> errors_by_mode = {"Inference": [], "Generative": []}
        >>> for noise in noise_levels:
        ...     # Test with noisy queries
        ...     errors_by_mode["Inference"].append(test_inference(noise))
        ...     errors_by_mode["Generative"].append(test_generative(noise))
        >>> fig = plot_retrieval_quality(errors_by_mode, noise_levels)
    """
    fig, ax = plt.subplots(figsize=figsize)

    for mode, errors in errors_by_mode.items():
        ax.plot(noise_levels, errors, marker="o", linewidth=2, label=mode)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# =======================================================================================
# MEMORY STATE COMPARISON
# =======================================================================================


def plot_memory_difference(
    M1: Matrix,
    M2: Matrix,
    labels: tuple = ("Memory 1", "Memory 2"),
    n_p_per_freq: Optional[List[int]] = None,
    title: str = "Memory Matrix Difference",
    figsize: tuple = (15, 5),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """Visualize difference between two memory matrices.

    Useful for comparing:
    - M_gen vs M_inf (inference vs generative memory)
    - Memory before vs after training
    - Different training configurations

    Args:
        M1: First memory matrix [sum(n_p), sum(n_p)]
        M2: Second memory matrix [sum(n_p), sum(n_p)]
        labels: Tuple of (label1, label2) for the two matrices
        n_p_per_freq: Place cell counts per frequency for boundary visualization
        title: Figure title
        figsize: Figure size (width, height)
        cmap: Colormap for difference plot

    Returns:
        matplotlib Figure with side-by-side comparison and difference

    Example:
        >>> M_before = storage.M_gen.clone()
        >>> # ... more training ...
        >>> M_after = storage.M_gen
        >>> fig = plot_memory_difference(M_before, M_after,
        ...     labels=("Before", "After"))
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Plot first matrix
    im1 = axes[0].imshow(M1.cpu().numpy(), cmap=cmap, aspect="auto")
    axes[0].set_title(labels[0])
    axes[0].set_xlabel("To")
    axes[0].set_ylabel("From")
    plt.colorbar(im1, ax=axes[0])

    # Plot second matrix
    im2 = axes[1].imshow(M2.cpu().numpy(), cmap=cmap, aspect="auto")
    axes[1].set_title(labels[1])
    axes[1].set_xlabel("To")
    axes[1].set_ylabel("From")
    plt.colorbar(im2, ax=axes[1])

    # Plot difference
    diff = (M2 - M1).cpu().numpy()
    im3 = axes[2].imshow(diff, cmap=cmap, aspect="auto")
    axes[2].set_title(f"Difference ({labels[1]} - {labels[0]})")
    axes[2].set_xlabel("To")
    axes[2].set_ylabel("From")
    plt.colorbar(im3, ax=axes[2])

    # Add frequency boundaries if provided
    if n_p_per_freq is not None:
        boundaries = [0] + [sum(n_p_per_freq[: i + 1]) for i in range(len(n_p_per_freq))]
        for ax in axes:
            for boundary in boundaries:
                ax.axhline(boundary, color="black", linestyle="--", alpha=0.3)
                ax.axvline(boundary, color="black", linestyle="--", alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()
    return fig
