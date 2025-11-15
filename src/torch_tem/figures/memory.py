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
from torch import Tensor

# =======================================================================================
# PROTOCOL INTERFACES
# =======================================================================================


class MemoryParams(Protocol):
    """Minimal interface for memory configuration."""

    n_p_calculated: List[int]
    """Place cell counts per frequency module."""


# =======================================================================================
# MEMORY MATRIX VISUALIZATION
# =======================================================================================


def plot_memory_matrices(
    M_gen: Tensor,
    M_inf: Optional[Tensor] = None,
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
        ...     n_p_per_freq=params.n_p_calculated,
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
    im1 = axes[0].imshow(M_gen.cpu().numpy(), cmap=cmap, aspect="auto")
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
        im2 = axes[1].imshow(M_inf.cpu().numpy(), cmap=cmap, aspect="auto")
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
    queries: List[Tensor],
    retrievals: List[Tensor],
    targets: List[Tensor],
    query_labels: Optional[List[str]] = None,
    title: str = "Attractor Dynamics: Query → Retrieval Convergence",
    figsize: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> plt.Figure:
    """Visualize attractor convergence for multiple query patterns.

    Displays query (noisy), retrieved (after attractor), and target (ground truth)
    patterns side-by-side for each test case. This shows how attractor dynamics
    refine noisy or partial queries toward stored patterns.

    Args:
        queries: List of query patterns [n_p_total] (one per test case)
        retrievals: List of retrieved patterns [n_p_total] (one per test case)
        targets: List of target patterns [n_p_total] (one per test case)
        query_labels: Optional labels for each query (e.g., ["Query 1", "Query 2"])
        title: Figure title
        figsize: Figure size (width, height). If None, auto-sized based on n_queries
        ylim: Y-axis limits for all subplots. If None, auto-scaled based on data range

    Returns:
        matplotlib Figure with convergence visualization

    Example:
        >>> queries = [noisy_pattern1, noisy_pattern2]
        >>> retrievals = [attractor.retrieve(q, M) for q in queries]
        >>> targets = [clean_pattern1, clean_pattern2]
        >>> fig = plot_attractor_convergence(queries, retrievals, targets)
        >>> fig.savefig('convergence.png')
    """
    n_queries = len(queries)

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
        figsize = (15, 3 * n_queries)

    fig, axes = plt.subplots(n_queries, 3, figsize=figsize)
    if n_queries == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_queries):
        # Generate label
        label = query_labels[i] if query_labels is not None else f"{i+1}"

        # Plot query pattern
        axes[i, 0].bar(range(len(queries[i])), queries[i].cpu().numpy(), alpha=0.7)
        axes[i, 0].set_title(f"Query {label} (Noisy)")
        axes[i, 0].set_xlabel("Place Cell Index")
        axes[i, 0].set_ylabel("Activation")
        axes[i, 0].set_ylim(ylim)

        # Plot retrieved pattern
        axes[i, 1].bar(range(len(retrievals[i])), retrievals[i].cpu().numpy(), alpha=0.7, color="green")
        axes[i, 1].set_title(f"Retrieved {label} (After Attractor)")
        axes[i, 1].set_xlabel("Place Cell Index")
        axes[i, 1].set_ylabel("Activation")
        axes[i, 1].set_ylim(ylim)

        # Plot target pattern (ground truth)
        axes[i, 2].bar(range(len(targets[i])), targets[i].cpu().numpy(), alpha=0.7, color="orange")
        axes[i, 2].set_title(f"Target {label} (Ground Truth)")
        axes[i, 2].set_xlabel("Place Cell Index")
        axes[i, 2].set_ylabel("Activation")
        axes[i, 2].set_ylim(ylim)

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
    masks: List[Tensor],
    n_p_per_freq: Optional[List[int]] = None,
    title: str = "Hierarchical Mask Schedule (Coarse-to-Fine Convergence)",
    figsize: Optional[tuple] = None,
    ylim: tuple = (0, 1.2),
) -> plt.Figure:
    """Visualize hierarchical mask schedule for attractor iterations.

    Shows how different frequency modules are progressively enabled during
    attractor dynamics. Early iterations update only low-frequency (coarse)
    components, with higher frequencies enabled in later iterations.

    Args:
        masks: List of mask tensors [n_p_total], one per attractor iteration
        n_p_per_freq: Place cell counts per frequency for boundary visualization
                     If None, no boundaries are drawn
        title: Figure title
        figsize: Figure size (width, height). If None, auto-sized based on n_masks
        ylim: Y-axis limits for all subplots

    Returns:
        matplotlib Figure with mask schedule visualization

    Example:
        >>> attractor = AttractorDynamics(params)
        >>> fig = plot_hierarchical_masks(
        ...     attractor.p_retrieve_mask_inf,
        ...     n_p_per_freq=params.n_p_calculated
        ... )
        >>> fig.savefig('hierarchical_masks.png')
    """
    n_masks = len(masks)

    # Auto-size figure based on number of masks
    if figsize is None:
        figsize = (4 * n_masks, 4)

    fig, axes = plt.subplots(1, n_masks, figsize=figsize)
    if n_masks == 1:
        axes = [axes]

    for it, mask in enumerate(masks):
        axes[it].bar(range(len(mask)), mask.cpu().numpy(), alpha=0.7, color=f"C{it}")
        axes[it].set_title(f"Iteration {it+1}")
        axes[it].set_xlabel("Place Cell Index")
        axes[it].set_ylabel("Mask Value (0=Frozen, 1=Active)")
        axes[it].set_ylim(ylim)

        # Add frequency boundaries if provided
        if n_p_per_freq is not None:
            boundaries = [sum(n_p_per_freq[:i]) for i in range(1, len(n_p_per_freq))]
            for boundary in boundaries:
                axes[it].axvline(boundary, color="black", linestyle="--", alpha=0.5)

    fig.suptitle(title)
    plt.tight_layout()
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
    M1: Tensor,
    M2: Tensor,
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
