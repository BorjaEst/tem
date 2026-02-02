from __future__ import annotations

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def initialise_axes(
    ax: Optional[Axes] = None,
    *,
    environment: Optional[object] = None,
    radius: Optional[float] = None,
    padding_scale: float = 2.0,
) -> Axes:
    """Initialize or configure axes for environment map plotting.

    Sets up axes with:
        - Limits derived from environment coordinates when provided
        - Equal aspect ratio (square axes)
        - Inverted y-axis (graphics convention: y increases downward)
        - Hidden axis labels and ticks

    Args:
        ax: Existing axes to configure. If None, creates new figure and axes.
        environment: Optional environment with a .locations list of dicts
            containing "o" (x) and "y" (y) coordinates.
        radius: Optional marker radius used to pad axis limits.
        padding_scale: Multiplier applied to radius for axis padding.

    Returns:
        Configured matplotlib Axes object.
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()

    if environment is not None and getattr(environment, "locations", None):
        coords = np.array([[loc.get("o"), loc.get("y")] for loc in environment.locations], dtype=float)
        valid = np.isfinite(coords).all(axis=1)
        coords = coords[valid]
        if coords.size > 0:
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            if radius is None:
                radius = _default_radius(getattr(environment, "n_locations", 0))
            pad = (radius or 0.02) * padding_scale
            if x_min == x_max:
                x_min -= 1.0
                x_max += 1.0
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
            ax.set_xlim([x_min - pad, x_max + pad])
            ax.set_ylim([y_min - pad, y_max + pad])
        else:
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    else:
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    ax.set_aspect(1)
    ax.invert_yaxis()  # Y increases downward (standard graphics convention)
    ax.axis("off")

    return ax


def _default_radius(n_locations: int) -> float:
    if n_locations <= 0:
        return 0.05
    return 2 * (0.01 + 1 / (10 * np.sqrt(n_locations)))


def subdivide_axes(
    ax,
    nrows=1,
    ncols=1,
    *,
    wspace=0.0,
    hspace=0.0,
    left_pad=0.0,
    right_pad=0.0,
    top_pad=0.0,
    bottom_pad=0.0,
) -> Axes | np.ndarray[Any, np.dtype[Axes]]:
    """
    Subdivide the EXACT bbox of `ax` into a grid of new axes.

    Returns:
        - Axes (for 1x1)
        - 1D np.ndarray of Axes (for 1xn or nx1)
        - 2D np.ndarray of Axes (for nxm)
    """
    fig = ax.figure
    fig.canvas.draw()

    bbox = ax.get_position()
    ax.remove()

    # Apply outer padding
    x0 = bbox.x0 + left_pad
    y0 = bbox.y0 + bottom_pad
    w = bbox.width - left_pad - right_pad
    h = bbox.height - top_pad - bottom_pad

    # Cell size
    cell_w = (w - (ncols - 1) * wspace) / ncols
    cell_h = (h - (nrows - 1) * hspace) / nrows

    out = []
    for r in range(nrows):
        row = []
        for c in range(ncols):
            left = x0 + c * (cell_w + wspace)
            bottom = y0 + (nrows - 1 - r) * (cell_h + hspace)
            new_ax = fig.add_axes([left, bottom, cell_w, cell_h])
            row.append(new_ax)
        out.append(row)

    # Convert to Matplotlib-like output shape
    arr = np.array(out, dtype=object)

    if nrows == 1 and ncols == 1:
        return arr[0, 0]
    elif nrows == 1:  # 1 × N
        return arr[0]
    elif ncols == 1:  # N × 1
        return arr[:, 0]
    else:
        return arr


def mosaic_axes(
    ax: plt.Axes,
    n_items: int,
    *,
    wspace=0.0,
    hspace=0.0,
    left_pad=0.0,
    right_pad=0.0,
    top_pad=0.0,
    bottom_pad=0.0,
) -> np.ndarray[Any, np.dtype[Axes]]:
    bbox = ax.get_position()
    slot_w_in = bbox.width  * fig_w_in
    slot_h_in = bbox.height * fig_h_in
    A = slot_w_in / slot_h_in

    usable_w = 1 - left - right
    usable_h = 1 - bottom - top

    ncols0 = round(sqrt(n_items * A))
    candidates = {clamp(ncols0-1), clamp(ncols0), clamp(ncols0+1)}

    for ncols in candidates:
        nrows = ceil(n_items / ncols)

        cell_w = usable_w / (ncols + (ncols-1)*wspace)
        cell_h = usable_h / (nrows + (nrows-1)*hspace)

        score = min(cell_w * slot_w_in, cell_h * slot_h_in)
        keep best (score, squareness, empties)

    gap_w = wspace * best_cell_w
    gap_h = hspace * best_cell_h

    return subdivide_axes(
        ax, best_nrows, best_ncols,
        wspace=gap_w, hspace=gap_h,
        left_pad=left_pad, right_pad=right_pad, bottom_pad=bottom_pad, top_pad=top_pad
    )