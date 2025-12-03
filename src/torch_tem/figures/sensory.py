"""Sensory processing visualization functions.

Provides plotting utilities for sensory processor components including:
- Frequency bank configuration and time constants
- Temporal filtering effects across frequencies
- Feature evolution over time
- Normalization effects
- Multi-frequency representations

All functions use Protocol-based typing for flexibility and testability.
Each function returns a matplotlib Figure object for flexible display/saving.
"""

from typing import List, Optional, Protocol, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from ..types import MultiScaleCode, Vector


# ==============================================================================
# Protocols
# ==============================================================================
class SensoryProcessorProtocol(Protocol):
    """Minimal sensory processor interface for plotting."""

    n_f: int
    n_x_c: int
    f_initial: List[float]


# ==============================================================================
# Frequency Bank Visualization
# ==============================================================================
def plot_frequency_bank(frequencies: List[float], title: str = "Frequency Bank Configuration", figsize: Tuple[float, float] = (12, 4)) -> plt.Figure:
    """Visualize frequency bank showing memory time constants.

    Creates a dual-panel visualization:
    - Left: Frequency values (0=long memory, 1=no memory)
    - Right: Effective time constants (τ = 1/f in steps)

    Args:
        frequencies: List of frequency values in (0, 1]
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> frequencies = [0.1, 0.3, 0.5, 0.9]
        >>> fig = plot_frequency_bank(frequencies)
        >>> fig.savefig('frequency_bank.png')
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    n_f = len(frequencies)

    # Plot frequencies
    ax1.bar(range(n_f), frequencies, color="steelblue", alpha=0.7, edgecolor="black", linewidth=1.5)
    ax1.set_xlabel("Frequency Channel", fontsize=11)
    ax1.set_ylabel("Frequency Value", fontsize=11)
    ax1.set_title("Frequency Values (0 = long memory, 1 = no memory)", fontsize=12)
    ax1.set_ylim([0, 1.05])
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_xticks(range(n_f))

    # Add value labels on bars
    for i, freq in enumerate(frequencies):
        ax1.text(i, freq + 0.02, f"{freq:.2f}", ha="center", va="bottom", fontsize=9)

    # Plot effective time constants (1 / frequency)
    time_constants = [1.0 / f if f > 0 else float("inf") for f in frequencies]
    # Cap infinite values for visualization
    time_constants_capped = [min(tc, max(time_constants[:-1]) * 1.5) if not np.isinf(tc) else max(time_constants[:-1]) * 1.5 for tc in time_constants]

    ax2.bar(range(n_f), time_constants_capped, color="darkorange", alpha=0.7, edgecolor="black", linewidth=1.5)
    ax2.set_xlabel("Frequency Channel", fontsize=11)
    ax2.set_ylabel("Effective Time Constant (steps)", fontsize=11)
    ax2.set_title("Memory Decay Time Constants (τ = 1/f)", fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(range(n_f))

    # Add value labels on bars
    for i, tc in enumerate(time_constants):
        if np.isinf(tc):
            label = "∞"
        else:
            label = f"{tc:.1f}"
        ax2.text(i, time_constants_capped[i] + 0.5, label, ha="center", va="bottom", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ==============================================================================
# Temporal Filtering Visualization
# ==============================================================================
def plot_temporal_filtering(
    x_c_history: List[Vector],
    x_f_history: List[MultiScaleCode],
    frequencies: List[float],
    title: str = "Temporal Filtering Across Frequencies",
    figsize: Tuple[float, float] = (14, 2),
    cmap: str = "viridis",
) -> plt.Figure:
    """Visualize temporal filtering effects across frequency channels.

    Creates a multi-panel heatmap showing:
    - Top panel: Original compressed sensory input
    - Subsequent panels: Filtered output for each frequency channel

    Args:
        x_c_history: List of T timesteps, each a tensor [n_x_c] (single trajectory)
        x_f_history: List of T timesteps, each with n_f filtered tensors [n_x_c] (single trajectory)
        frequencies: List of frequency values
        title: Plot title
        figsize: Figure size per panel (width, height)
        cmap: Colormap name

    Returns:
        matplotlib Figure object

    Example:
        >>> x_c_history = [torch.randn(10) for _ in range(100)]
        >>> x_f_history = [[torch.randn(10) for _ in range(4)] for _ in range(100)]
        >>> frequencies = [0.1, 0.3, 0.5, 0.9]
        >>> fig = plot_temporal_filtering(x_c_history, x_f_history, frequencies)
    """
    n_f = len(frequencies)
    T = len(x_c_history)

    # Create subplot grid: original + all frequencies
    fig, axes = plt.subplots(n_f + 1, 1, figsize=(figsize[0], figsize[1] * (n_f + 1)), sharex=True)

    # Plot original compressed sensory - stack list into [T, n_x_c]
    x_c_stacked = torch.stack(x_c_history)
    x_c_np = x_c_stacked.detach().cpu().numpy()
    im0 = axes[0].imshow(x_c_np.T, aspect="auto", cmap=cmap, interpolation="nearest")
    axes[0].set_ylabel("Feature Dim", fontsize=10)
    axes[0].set_title("Original Compressed Sensory (x_c)", fontsize=11, fontweight="bold")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Plot each frequency channel
    for f_idx in range(n_f):
        # Stack filtered tensors over time: [T, n_x_c]
        x_f_t = torch.stack([x_f_history[t][f_idx] for t in range(T)])
        x_f_np = x_f_t.detach().cpu().numpy()

        im = axes[f_idx + 1].imshow(x_f_np.T, aspect="auto", cmap=cmap, interpolation="nearest")
        axes[f_idx + 1].set_ylabel("Feature Dim", fontsize=10)
        freq_val = frequencies[f_idx]
        tau = 1.0 / freq_val if freq_val > 0 else float("inf")
        tau_str = f"{tau:.1f}" if not np.isinf(tau) else "∞"
        axes[f_idx + 1].set_title(f"Frequency {f_idx}: f={freq_val:.2f} (τ≈{tau_str} steps)", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[f_idx + 1], fraction=0.046, pad=0.04)

    axes[-1].set_xlabel("Time Step", fontsize=11)
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    return fig


# ==============================================================================
# Feature Comparison Visualization
# ==============================================================================
def plot_frequency_comparison(
    x_c_history: List[Vector],
    x_f_history: List[MultiScaleCode],
    frequencies: List[float],
    feature_idx: int = 0,
    title: str = "Single Feature Across Frequencies",
    figsize: Tuple[float, float] = (14, 6),
) -> plt.Figure:
    """Plot a single feature dimension across all frequency channels over time.

    Shows how a single feature evolves differently across frequency channels,
    demonstrating the temporal smoothing effect.

    Args:
        x_c_history: List of T timesteps, each a tensor [n_x_c] (single trajectory)
        x_f_history: List of T timesteps with n_f filtered tensors [n_x_c] (single trajectory)
        frequencies: List of frequency values
        feature_idx: Which feature dimension to plot
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Example:
        >>> x_c_history = [torch.randn(10) for _ in range(100)]
        >>> fig = plot_frequency_comparison(x_c_history, x_f_history, frequencies, feature_idx=0)
        >>> fig.savefig('feature_comparison.png')
    """
    n_f = len(frequencies)
    T = len(x_c_history)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Plot original - extract feature from list of timesteps
    x_c_feature = torch.stack([x_c_history[t][feature_idx] for t in range(T)])
    x_c_feature_np = x_c_feature.detach().cpu().numpy()
    ax.plot(range(T), x_c_feature_np, label="Original (x_c)", linewidth=2, color="black", linestyle="--", alpha=0.7, zorder=n_f + 1)

    # Plot each frequency
    colors = plt.cm.viridis(np.linspace(0, 1, n_f))
    for f_idx in range(n_f):
        x_f_feature = torch.stack([x_f_history[t][f_idx][feature_idx] for t in range(T)])
        x_f_np = x_f_feature.detach().cpu().numpy()
        freq_val = frequencies[f_idx]
        tau = 1.0 / freq_val if freq_val > 0 else float("inf")
        tau_str = f"{tau:.1f}" if not np.isinf(tau) else "∞"
        ax.plot(range(T), x_f_np, label=f"f={freq_val:.2f} (τ≈{tau_str})", linewidth=1.5, color=colors[f_idx], alpha=0.8, zorder=n_f - f_idx)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Activation", fontsize=12)
    ax.set_title(f"{title} (Feature {feature_idx})", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ==============================================================================
# Normalization Effects Visualization
# ==============================================================================
def plot_normalization_effects(
    x_f_raw: MultiScaleCode,
    x_f_normalized: MultiScaleCode,
    frequencies: List[float],
    title: str = "L2 Normalization Effects",
    figsize: Tuple[float, float] = (3, 6),
    cmap: str = "coolwarm",
) -> plt.Figure:
    """Compare raw filtered outputs with normalized outputs.

    Creates a grid showing before/after normalization for each frequency channel,
    demonstrating the scale stabilization effect.

    Args:
        x_f_raw: List of n_f raw filtered tensors [B, n_x_c]
        x_f_normalized: List of n_f normalized tensors [B, n_x_c]
        frequencies: List of frequency values
        title: Plot title
        figsize: Figure size per subplot (width, height)
        cmap: Colormap name

    Returns:
        matplotlib Figure object

    Example:
        >>> x_f_raw = [torch.randn(5, 10) for _ in range(4)]
        >>> x_f_normalized = [torch.randn(5, 10) for _ in range(4)]
        >>> frequencies = [0.1, 0.3, 0.5, 0.9]
        >>> fig = plot_normalization_effects(x_f_raw, x_f_normalized, frequencies)
    """
    n_f = len(frequencies)
    fig, axes = plt.subplots(2, n_f, figsize=(figsize[0] * n_f, figsize[1] * 2), sharex=True, sharey="row")

    # Handle single frequency case
    if n_f == 1:
        axes = axes.reshape(2, 1)

    # Compute global min/max across all frequencies for shared colorbar
    all_raw_values = torch.cat([x_f_raw[f_idx].flatten() for f_idx in range(n_f)])
    all_norm_values = torch.cat([x_f_normalized[f_idx].flatten() for f_idx in range(n_f)])
    vmin = min(all_raw_values.min().item(), all_norm_values.min().item())
    vmax = max(all_raw_values.max().item(), all_norm_values.max().item())

    for f_idx in range(n_f):
        # Raw
        raw_np = x_f_raw[f_idx].detach().cpu().numpy()
        im1 = axes[0, f_idx].imshow(raw_np.T, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[0, f_idx].set_title(f"f={frequencies[f_idx]:.2f} (Raw)", fontsize=10)
        plt.colorbar(im1, ax=axes[0, f_idx], fraction=0.046, pad=0.04)

        # Normalized
        norm_np = x_f_normalized[f_idx].detach().cpu().numpy()
        im2 = axes[1, f_idx].imshow(norm_np.T, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        axes[1, f_idx].set_title(f"f={frequencies[f_idx]:.2f} (Norm)", fontsize=10)
        plt.colorbar(im2, ax=axes[1, f_idx], fraction=0.046, pad=0.04)

        if f_idx == 0:
            axes[0, f_idx].set_ylabel("Feature Dim\n(Raw)", fontsize=10)
            axes[1, f_idx].set_ylabel("Feature Dim\n(Normalized)", fontsize=10)

        axes[1, f_idx].set_xlabel("Batch Sample", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ==============================================================================
# Multi-Frequency Representation Visualization
# ==============================================================================
def plot_multi_frequency_representation(
    x_f_list: List[MultiScaleCode],
    frequencies: List[float],
    timesteps: List[int],
    title: str = "Multi-Frequency Representation",
    figsize: Tuple[float, float] = (4, 3),
    cmap: str = "viridis",
) -> plt.Figure:
    """Visualize multi-frequency representations at specific timesteps.

    Shows how the same input is represented differently across frequency channels
    at selected points in time.

    Args:
        x_f_list: List of timesteps, each with n_f filtered tensors [B, n_x_c]
        frequencies: List of frequency values
        timesteps: Which timesteps to display
        title: Plot title
        figsize: Figure size per subplot (width, height)
        cmap: Colormap name

    Returns:
        matplotlib Figure object

    Example:
        >>> x_f_list = [[torch.randn(1, 10) for _ in range(4)] for _ in range(100)]
        >>> frequencies = [0.1, 0.3, 0.5, 0.9]
        >>> timesteps = [0, 25, 50, 75]
        >>> fig = plot_multi_frequency_representation(x_f_list, frequencies, timesteps)
    """
    n_f = len(frequencies)
    n_t = len(timesteps)

    fig, axes = plt.subplots(n_t, n_f, figsize=(figsize[0] * n_f, figsize[1] * n_t), sharex=True, sharey=True)

    # Handle single row/column cases
    if n_t == 1 and n_f == 1:
        axes = np.array([[axes]])
    elif n_t == 1:
        axes = axes.reshape(1, -1)
    elif n_f == 1:
        axes = axes.reshape(-1, 1)

    for t_idx, t in enumerate(timesteps):
        for f_idx in range(n_f):
            x_f_np = x_f_list[t][f_idx].detach().cpu().numpy()

            im = axes[t_idx, f_idx].imshow(x_f_np.T, aspect="auto", cmap=cmap, interpolation="nearest")

            # Title for top row only
            if t_idx == 0:
                freq_val = frequencies[f_idx]
                tau = 1.0 / freq_val if freq_val > 0 else float("inf")
                tau_str = f"{tau:.1f}" if not np.isinf(tau) else "∞"
                axes[t_idx, f_idx].set_title(f"f={freq_val:.2f}\n(τ≈{tau_str})", fontsize=10)

            # Y-axis label for first column only
            if f_idx == 0:
                axes[t_idx, f_idx].set_ylabel(f"t={t}\nFeature", fontsize=9)

            # Colorbar
            plt.colorbar(im, ax=axes[t_idx, f_idx], fraction=0.046, pad=0.04)

    # X-axis labels for bottom row
    for f_idx in range(n_f):
        axes[-1, f_idx].set_xlabel("Batch", fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ==============================================================================
# Sensory Projection to Hippocampal p-space
# ==============================================================================
def plot_sensory_projection(
    x_p_history: Union[List[MultiScaleCode], MultiScaleCode, Vector],
    n_p_per_freq: Optional[List[int]] = None,
    title: str = "Sensory Projection to Hippocampal p-space (x → p)",
    figsize: Tuple[float, float] = (12, 4),
    cmap: str = "magma",
) -> plt.Figure:
    """Visualize sensory projection outputs in p-space over time.

    Renders a heatmap with time on the x-axis and place cell indices on the y-axis.
    If ``n_p_per_freq`` is provided, frequency boundaries are overlaid to highlight
    the hierarchical structure of p-space.

    Accepted input formats for ``x_p_history``:
    - List[List[Tensor]]: length T, each entry is a list of n_f tensors shaped
      [n_p[f]] or [B, n_p[f]] (first batch element is used).
    - List[Tensor]: length T, each entry is a flattened p-vector [sum(n_p)].
    - Tensor: [T, sum(n_p)] flattened over frequencies.

    Args:
        x_p_history: Projection outputs over time in one of the accepted formats.
        n_p_per_freq: Place cell counts per frequency (for boundary overlays).
        title: Plot title.
        figsize: Figure size (width, height).
        cmap: Matplotlib colormap name.

    Returns:
        matplotlib Figure with a heatmap of p-space activations over time.

    Example:
        >>> # x_f_t: [n_f, n_x_f] per timestep; apply projection per timestep
        >>> x_p_hist = []
        >>> for t in range(T):
        ...     # SensoryProjection expects a list of tensors per frequency with batch dim
        ...     x_list = [x_f_t[f].unsqueeze(0) for f in range(n_f)]  # [1, n_x_f[f]]
        ...     x_p_f = projection(x_list)  # List of [1, n_p[f]]
        ...     x_p_hist.append([x_p_f[f].squeeze(0) for f in range(n_f)])
        >>> fig = plot_sensory_projection(x_p_hist, n_p_per_freq=params.n_p)
    """

    # Helper to convert various inputs to a [T, sum(n_p)] numpy array
    def _to_time_by_p_numpy(inp: Union[List[List[Tensor]], List[Tensor], Tensor]) -> np.ndarray:
        if isinstance(inp, list):
            # Case A: List of lists (T x n_f)
            if len(inp) == 0:
                return np.zeros((0, 0), dtype=np.float32)
            if isinstance(inp[0], list):
                # Concatenate per-frequency tensors at each timestep
                flat_ts: List[Tensor] = []
                for t_list in inp:  # type: ignore[assignment]
                    parts: List[Tensor] = []
                    for part in t_list:
                        # Accept [n_p] or [B, n_p] → take first batch if present
                        if part.dim() == 2:
                            parts.append(part[0])
                        else:
                            parts.append(part)
                    flat_ts.append(torch.cat(parts, dim=-1))
                mat = torch.stack(flat_ts, dim=0)  # [T, sum(n_p)]
                return mat.detach().cpu().numpy()
            # Case B: List of flattened tensors (T x [sum(n_p)])
            elif isinstance(inp[0], Tensor):
                mat = torch.stack([t if t.dim() == 1 else t.view(-1) for t in inp], dim=0)
                return mat.detach().cpu().numpy()
        elif isinstance(inp, Tensor):
            # Case C: Tensor [T, sum(n_p)] (already flattened)
            if inp.dim() == 2:
                return inp.detach().cpu().numpy()
        raise ValueError("Unsupported x_p_history format. Provide List[List[Tensor]], List[Tensor], or Tensor [T, sum(n_p)].")

    x_time_p = _to_time_by_p_numpy(x_p_history)  # [T, sum(n_p)]
    if x_time_p.size == 0:
        # Create empty figure gracefully
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_axis_off()
        ax.set_title("No projection data to display")
        return fig

    # Plot heatmap (transpose so y-axis indexes place cells)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(x_time_p.T, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Place Cell Index", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Activation")

    # Overlay frequency boundaries if provided
    if n_p_per_freq is not None and len(n_p_per_freq) > 1:
        boundaries = [0] + [sum(n_p_per_freq[: i + 1]) for i in range(len(n_p_per_freq))]
        for b in boundaries:
            ax.axhline(b - 0.5, color="white", linestyle="--", alpha=0.6, linewidth=1)

    fig.tight_layout()
    return fig
