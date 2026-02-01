from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from torch_tem.figures.plots.map import plot_map
from torch_tem.figures.utils import aggregate_rate_map, select_mosaic_grid
from torch_tem.figures.utils.rasterize import rasterize_locations


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

    plot_map(world, values, ax=ax, vmin=vmin, vmax=vmax, shape=shape, location_cm=cmap)
    return ax


def plot_ratematx_cell(
    ax: plt.Axes,
    world: object,
    cells: np.ndarray,
    location_ids: list[int],
    cell_idx: int,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
) -> plt.Axes:
    """Plot a single cell's rate matrix on an existing axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (T, B, C) or (T, C) with cell activations.
        location_ids: Ordered list of visited location indices.
        cell_idx: Index of the cell to plot.
        vmin: Minimum value for color scaling.
        vmax: Maximum value for color scaling.
        cmap: Colormap name for the rate map.

    Returns:
        The axes with the rate map matrix form rendered.
    """
    n_locations = len(getattr(world, "locations", []))
    rate_map, _ = aggregate_rate_map(cells, location_ids, n_locations)
    if rate_map.size == 0 or cell_idx < 0 or cell_idx >= rate_map.shape[0]:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    values = rate_map[cell_idx]
    grid, _, extent = rasterize_locations(world, values)

    if grid.size == 0 or not np.isfinite(grid).any():
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    finite = np.isfinite(grid)
    if vmin is None:
        vmin = float(np.nanmin(grid[finite])) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(grid[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("#d9d9d9")  # NaNs show as light gray

    ax.imshow(
        grid,
        origin="lower",
        interpolation="nearest",
        cmap=cm,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="equal",
    )
    ax.axis("off")
    return ax


def plot_ratematx_mosaic(
    ax: plt.Axes,
    world: object,
    cells_trace: NDArray,
    location_ids: list[int],
    *,
    cell_indices: Optional[Sequence[int]] = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "copper_r",
    min_cols: int = 2,
    max_cols: int = 36,
    wspace: float = 0.04,
    hspace: float = 0.04,
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
    cell_array = np.asarray(cells_trace)
    if cell_array.ndim < 2 or cell_array.shape[-1] == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    n_cells_total = int(cell_array.shape[-1])
    if cell_indices is None:
        indices = list(range(n_cells_total))
    else:
        indices = []
        for idx in cell_indices:
            idx_int = int(idx)
            if 0 <= idx_int < n_cells_total:
                indices.append(idx_int)

    if not indices:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return ax

    if hasattr(ax, "_tem_mosaic_axes"):
        for old_ax in list(getattr(ax, "_tem_mosaic_axes", [])):
            try:
                old_ax.remove()
            except (AttributeError, ValueError):
                continue

    fig = ax.figure
    bbox = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    slot_w = float(bbox.width * fig_w)
    slot_h = float(bbox.height * fig_h)
    wspace = max(float(wspace), 0.0)
    hspace = max(float(hspace), 0.0)
    nrows, ncols = select_mosaic_grid(
        len(indices),
        slot_w,
        slot_h,
        min_cols=min_cols,
        max_cols=max_cols,
        wspace=wspace,
        hspace=hspace,
    )

    denom_cols = ncols + max(ncols - 1, 0) * wspace
    denom_rows = nrows + max(nrows - 1, 0) * hspace
    if denom_cols <= 0 or denom_rows <= 0:
        ax.text(0.5, 0.5, "No layout", ha="center", va="center")
        ax.axis("off")
        return ax

    w = 1.0 / denom_cols
    h = 1.0 / denom_rows
    axes: list[plt.Axes] = []
    for plot_idx, cell_idx in enumerate(indices):
        r = plot_idx // ncols
        c = plot_idx % ncols
        x0 = c * (w + wspace * w)
        y0 = 1.0 - (r + 1) * h - r * (hspace * h)
        subax = ax.inset_axes([x0, y0, w, h])
        plot_ratematx_cell(
            subax,
            world,
            cells_trace,
            location_ids,
            cell_idx,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        axes.append(subax)

    ax.set_axis_off()
    ax._tem_mosaic_axes = axes
    return ax
