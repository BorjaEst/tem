"""Grounded location inference visualization.

Provides plotting functions for visualizing grounded location (place cell) inference,
including outer product structure and temporal dynamics. All functions follow the
torch_tem visualization conventions:

- Return matplotlib Figure objects for flexible saving/display
- Accept Protocol-based interfaces for loose coupling
- No side effects (caller controls show/save)
- Consistent aesthetics and styling
"""

from typing import List, Protocol

import matplotlib.pyplot as plt
import torch
from torch import Tensor


# ==============================================================================
# Protocols
# ==============================================================================
class PlaceCellHistoryProtocol(Protocol):
    """Protocol for place cell activity history."""

    def __len__(self) -> int:
        """Return number of timesteps."""
        ...

    def __getitem__(self, idx: int) -> List[Tensor]:
        """Return place cell activity at timestep idx.

        Returns:
            List[n_f] of [B, n_p[f]] tensors
        """
        ...


# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_grounded_location_activity(
    p_history: List[List[Tensor]],
    observations: List[Tensor],
    locations: Tensor,
    frequencies: List[float],
    n_cells_per_freq: List[int],
    max_cells: int = 30,
    figsize: tuple = None,
    title: str = None,
) -> plt.Figure:
    """Plot grounded location (place cell) activity over time.

    Visualizes the temporal evolution of place cell responses across multiple
    frequency modules, along with corresponding sensory observations and
    spatial locations.

    Args:
        p_history: List[T] of [List[n_f] of [B, n_p[f]]] - place cell activity over time
        observations: List of T timesteps, each a tensor [n_x] or [B, n_x] - one-hot observation vectors
        locations: [T] - location indices
        frequencies: Frequency values per module
        n_cells_per_freq: Number of place cells per frequency module
        max_cells: Maximum cells to visualize per frequency (subsamples if exceeded)
        figsize: Figure size (width, height). Auto-computed if None
        title: Optional main title for the figure

    Returns:
        matplotlib.figure.Figure: Figure with n_f+2 subplots showing observations,
            locations, and place cell activity heatmaps per frequency

    Example:
        >>> # After running grounded inference pipeline
        >>> fig = plot_grounded_location_activity(
        ...     p_history=p_history,
        ...     observations=observations,
        ...     locations=locations,
        ...     frequencies=[0.1, 0.3, 0.9],
        ...     n_cells_per_freq=[96, 80, 64]
        ... )
        >>> fig.savefig('place_cell_activity.png')
    """
    n_f = len(p_history[0])
    T = len(p_history)

    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (14, 3 * (n_f + 2))

    fig, axes = plt.subplots(n_f + 2, 1, figsize=figsize, sharex=True)

    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # Plot observations - stack list and extract indices
    obs_stacked = torch.stack([observations[t].squeeze() if observations[t].dim() > 1 else observations[t] for t in range(T)])
    obs_indices = torch.argmax(obs_stacked, dim=1).numpy()
    axes[0].plot(obs_indices, "o-", linewidth=1, markersize=3, color="black")
    axes[0].set_ylabel("Observation\nIndex", fontsize=10)
    axes[0].set_title("Sensory Input (Observations)", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Plot locations
    axes[1].plot(locations.numpy(), "s-", linewidth=1, markersize=3, color="darkblue")
    axes[1].set_ylabel("Location\nIndex", fontsize=10)
    axes[1].set_title("Spatial Location", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    # Plot place cell activity per frequency
    for f in range(n_f):
        # Extract activity over time [T, n_p[f]]
        activity = torch.stack([p_history[t][f][0] for t in range(T)])  # [T, n_p[f]]

        # Subsample cells for visualization if too many
        if n_cells_per_freq[f] > max_cells:
            indices = torch.linspace(0, n_cells_per_freq[f] - 1, max_cells).long()
            activity = activity[:, indices]
            n_vis = max_cells
        else:
            n_vis = n_cells_per_freq[f]

        # Plot as heatmap
        im = axes[f + 2].imshow(activity.T.detach().numpy(), aspect="auto", cmap="viridis", interpolation="nearest")
        axes[f + 2].set_ylabel(f"Place Cells\nFreq {f}\n(n={n_vis})", fontsize=9)
        axes[f + 2].set_title(f"Frequency {f} (f={frequencies[f]:.2f}) - Place Cell Activity", fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=axes[f + 2], label="Activation")

    axes[-1].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout()
    return fig


def plot_outer_product_structure(
    g_sample: List[Tensor],
    x_sample: List[Tensor],
    p_sample: List[Tensor],
    frequencies: List[float],
    figsize: tuple = None,
    title: str = None,
) -> plt.Figure:
    """Visualize outer product structure: p = g ⊗ x.

    Shows the conjunctive coding structure of place cells by displaying
    grid cell activity, sensory activity, and their outer product for
    each frequency module at a single timepoint.

    Args:
        g_sample: Grid cell activity [n_f] of [1, n_g_sub[f]]
        x_sample: Sensory activity [n_f] of [1, n_x_c]
        p_sample: Place cell activity [n_f] of [1, n_p[f]]
        frequencies: Frequency values per module
        figsize: Figure size (width, height). Auto-computed if None
        title: Optional main title for the figure

    Returns:
        matplotlib.figure.Figure: Figure with n_f rows × 3 columns showing
            grid cells, sensory features, and place cell matrix per frequency

    Example:
        >>> # Visualize outer product at specific timepoint
        >>> mid_point = len(g_history) // 2
        >>> g_mid = [g_history[f][mid_point:mid_point+1, 0, :] for f in range(n_f)]
        >>> fig = plot_outer_product_structure(
        ...     g_sample=g_mid_downsampled,
        ...     x_sample=x_f_history[mid_point],
        ...     p_sample=p_history[mid_point],
        ...     frequencies=[0.1, 0.3, 0.9]
        ... )
        >>> fig.savefig('outer_product_structure.png')
    """
    n_f = len(g_sample)

    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (12, 3 * n_f)

    fig, axes = plt.subplots(n_f, 3, figsize=figsize)

    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # Handle single frequency case
    if n_f == 1:
        axes = axes.reshape(1, -1)

    for f in range(n_f):
        g_vec = g_sample[f][0].detach().numpy()  # [n_g_sub[f]]
        x_vec = x_sample[f][0].detach().numpy()  # [n_x_c]
        p_vec = p_sample[f][0].detach().numpy()  # [n_p[f]]

        # Plot grid cells
        axes[f, 0].bar(range(len(g_vec)), g_vec, color="steelblue")
        axes[f, 0].set_title(f"Freq {f} (f={frequencies[f]:.2f})\nGrid Cells g[{f}]", fontsize=10, fontweight="bold")
        axes[f, 0].set_xlabel("Grid Cell Index")
        axes[f, 0].set_ylabel("Activation")
        axes[f, 0].grid(True, alpha=0.3)

        # Plot sensory
        axes[f, 1].bar(range(len(x_vec)), x_vec, color="darkorange")
        axes[f, 1].set_title(f"Sensory x[{f}]", fontsize=10, fontweight="bold")
        axes[f, 1].set_xlabel("Sensory Feature Index")
        axes[f, 1].set_ylabel("Activation")
        axes[f, 1].grid(True, alpha=0.3)

        # Plot place cells (reshaped to show structure)
        n_g = len(g_vec)
        n_x = len(x_vec)
        p_matrix = p_vec.reshape(n_g, n_x)  # Reshape to show conjunctive structure

        im = axes[f, 2].imshow(p_matrix, aspect="auto", cmap="RdYlBu_r", interpolation="nearest")
        axes[f, 2].set_title(f"Place Cells p[{f}] = g ⊗ x\n(Conjunctive Coding)", fontsize=10, fontweight="bold")
        axes[f, 2].set_xlabel("Sensory Feature")
        axes[f, 2].set_ylabel("Grid Cell")
        plt.colorbar(im, ax=axes[f, 2], label="Activation")

    plt.tight_layout()
    return fig


def plot_place_cell_dynamics(
    p_history: List[List[Tensor]],
    observations: List[Tensor],
    frequencies: List[float],
    n_cells_per_freq: List[int],
    cell_indices: List[int] = None,
    figsize: tuple = None,
    title: str = None,
) -> plt.Figure:
    """Plot temporal dynamics of individual place cells across frequencies.

    Tracks the activation of specific place cells over time, showing how
    they respond to changing sensory observations. Useful for analyzing
    frequency-dependent temporal dynamics.

    Args:
        p_history: List[T] of [List[n_f] of [B, n_p[f]]] - place cell activity
        observations: List of T timesteps, each a tensor [n_x] or [B, n_x] - one-hot observation vectors
        frequencies: Frequency values per module
        n_cells_per_freq: Number of place cells per frequency module
        cell_indices: Specific cell indices to plot (defaults to middle cell per freq)
        figsize: Figure size (width, height). Auto-computed if None
        title: Optional main title for the figure

    Returns:
        matplotlib.figure.Figure: Figure with n_f subplots showing temporal
            dynamics of selected place cells

    Example:
        >>> # Plot middle cell from each frequency module
        >>> fig = plot_place_cell_dynamics(
        ...     p_history=p_history,
        ...     observations=observations,
        ...     frequencies=[0.1, 0.3, 0.9],
        ...     n_cells_per_freq=[96, 80, 64]
        ... )
        >>> fig.savefig('place_cell_dynamics.png')
    """
    n_f = len(frequencies)
    T = len(p_history)

    # Default to middle cell per frequency
    if cell_indices is None:
        cell_indices = [n_cells // 2 for n_cells in n_cells_per_freq]

    # Auto-compute figure size if not provided
    if figsize is None:
        figsize = (12, 3 * n_f)

    fig, axes = plt.subplots(n_f, 1, figsize=figsize, sharex=True)

    # Add main title if provided
    if title:
        fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    # Handle single frequency case
    if n_f == 1:
        axes = [axes]

    for f in range(n_f):
        # Extract single place cell over time
        cell_idx = cell_indices[f]
        activity = torch.stack([p_history[t][f][0, cell_idx] for t in range(T)])

        axes[f].plot(activity.detach().numpy(), linewidth=1.5, color=f"C{f}")
        axes[f].set_ylabel(f"Activation\nFreq {f}", fontsize=10)
        axes[f].set_title(f"Place Cell {cell_idx} (Freq {f}, f={frequencies[f]:.2f})", fontsize=11, fontweight="bold")
        axes[f].grid(True, alpha=0.3)

        # Mark observation changes - stack list and compute changes
        obs_stacked = torch.stack([observations[t].squeeze() if observations[t].dim() > 1 else observations[t] for t in range(T)])
        obs_changes = torch.where(torch.diff(torch.argmax(obs_stacked, dim=1)) != 0)[0] + 1
        for change in obs_changes:
            axes[f].axvline(change, color="red", alpha=0.2, linestyle="--", linewidth=0.5)

    axes[-1].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout()
    return fig
