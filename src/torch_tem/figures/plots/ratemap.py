from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from torch_tem.figures.plots.map import plot_map
from torch_tem.figures.utils import aggregate_rate_map


def plot_ratemap_cell(
    ax: plt.Axes,
    world: object,
    cells: np.ndarray,
    location_ids: list[int],
    cell_idx: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    shape: str = "square",
    cmap: str = "copper_r",
) -> plt.Axes:
    """Plot a single cell's rate map on an existing axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_idx: Index of the cell to plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        shape: Shape for the background map markers.
        cmap: Colormap name for the rate map.

    Returns:
        The axes with the rate map rendered.
    """
    rate_map, _ = aggregate_rate_map(cells, location_ids, len(world.locations))
    values = rate_map[cell_idx]

    finite_mask = np.isfinite(values)
    if vmin is None:
        vmin = float(np.min(values[finite_mask])) if finite_mask.any() else 0.0
    if vmax is None:
        vmax = float(np.max(values[finite_mask])) if finite_mask.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    plot_map(world, values, ax=ax, vmin=vmin, vmax=vmax, shape=shape, cmap=cmap)
    return ax


def plot_ratemap_mosaic(
    ax: plt.Axes,
    world: object,
    cells: np.ndarray,
    location_ids: list[int],
    *,
    cell_indices: Optional[Sequence[int]] = None,
    vmin: float | None = None,
    vmax: float | None = None,
    shape: str = "square",
    cmap: str = "copper_r",
    min_cols: int = 2,
    max_cols: int = 36,
    wspace: float = 0.0,
    hspace: float = 0.0,
) -> plt.Axes:
    """Plot multiple cell rate maps in a mosaic layout on existing axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_indices: Optional cell indices to include.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        shape: Shape for the background map markers.
        cmap: Colormap name for the rate map.
    Returns:
        The axes with the rate map mosaic rendered.
    """
    pass
