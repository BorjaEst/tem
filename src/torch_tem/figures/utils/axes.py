from __future__ import annotations

import math
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
    ax: Axes,
    nrows: int = 1,
    ncols: int = 1,
    *,
    wspace: float = 0.0,
    hspace: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    top_pad: float = 0.0,
    bottom_pad: float = 0.0,
    hide_parent: bool = True,
) -> Axes | np.ndarray[Any, np.dtype[Axes]]:
    """
    Subdivide the bbox of `ax` into a grid of new child axes.

    IMPORTANT:
    - Does NOT remove `ax` (so titles/annotations can remain).
    - Child axes are created as inset axes anchored to `ax` in ax.transAxes,
      so they follow any later layout changes (e.g., colorbars shrinking the parent).

    Spacing/padding units:
    - `left_pad`, `right_pad`, `top_pad`, `bottom_pad`, `wspace`, `hspace`
      are in parent-axes fraction units (0..1).

    Returns:
        - Axes (for 1x1)
        - 1D np.ndarray of Axes (for 1xn or nx1)
        - 2D np.ndarray of Axes (for nxm)
    """
    if nrows <= 0 or ncols <= 0:
        raise ValueError(f"nrows and ncols must be positive, got nrows={nrows}, ncols={ncols}")

    if hide_parent:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.patch.set_alpha(0.0)
        ax.set_navigate(False)

    # Usable area inside the parent axes, in parent axes coordinates.
    usable_w = 1.0 - left_pad - right_pad
    usable_h = 1.0 - top_pad - bottom_pad
    if usable_w <= 0 or usable_h <= 0:
        return np.array([], dtype=object)

    # Cell size in parent axes coords.
    cell_w = (usable_w - (ncols - 1) * wspace) / ncols
    cell_h = (usable_h - (nrows - 1) * hspace) / nrows
    if cell_w <= 0 or cell_h <= 0:
        return np.array([], dtype=object)

    out: list[list[Axes]] = []
    for r in range(nrows):
        row: list[Axes] = []
        for c in range(ncols):
            x0 = left_pad + c * (cell_w + wspace)
            y0 = bottom_pad + (nrows - 1 - r) * (cell_h + hspace)
            child = ax.inset_axes([x0, y0, cell_w, cell_h], transform=ax.transAxes)
            row.append(child)
        out.append(row)

    arr = np.array(out, dtype=object)

    if nrows == 1 and ncols == 1:
        return arr[0, 0]
    if nrows == 1:  # 1 × N
        return arr[0]
    if ncols == 1:  # N × 1
        return arr[:, 0]
    return arr


def mosaic_axes(
    ax: plt.Axes,
    n_items: int,
    *,
    wspace: float = 0.0,
    hspace: float = 0.0,
    left_pad: float = 0.0,
    right_pad: float = 0.0,
    top_pad: float = 0.0,
    bottom_pad: float = 0.0,
) -> np.ndarray[Any, np.dtype[Axes]]:
    """Create a mosaic of axes sized to fit `n_items` plots.

    Args:
        ax: Slot axes whose bounding box defines the available space.
        n_items: Number of panels to layout.
        wspace: Horizontal spacing as a fraction of each cell width.
        hspace: Vertical spacing as a fraction of each cell height.
        left_pad: Left padding in figure fraction units.
        right_pad: Right padding in figure fraction units.
        top_pad: Top padding in figure fraction units.
        bottom_pad: Bottom padding in figure fraction units.

    Returns:
        Array of created axes laid out within the slot.
    """
    fig = ax.figure
    if fig is None:
        raise ValueError("Cannot create a mosaic for an Axes that is not attached to a Figure.")

    # Ensure layout has been computed before reading positions.
    if getattr(fig, "canvas", None) is not None:
        fig.canvas.draw()

    bbox = ax.get_position()

    if n_items <= 0:
        return np.array([], dtype=object)

    fig_w_in, fig_h_in = fig.get_size_inches()
    slot_w_in = bbox.width * fig_w_in
    slot_h_in = bbox.height * fig_h_in
    if slot_w_in <= 0 or slot_h_in <= 0:
        return np.array([], dtype=object)

    aspect = slot_w_in / slot_h_in
    usable_w = max(bbox.width - left_pad - right_pad, 1e-6)
    usable_h = max(bbox.height - top_pad - bottom_pad, 1e-6)

    def clamp_cols(value: int) -> int:
        return max(1, min(int(value), n_items))

    ncols0 = int(round(math.sqrt(n_items * aspect)))
    candidates = {clamp_cols(ncols0 - 1), clamp_cols(ncols0), clamp_cols(ncols0 + 1)}

    best: dict[str, float | int] | None = None
    for ncols in sorted(candidates):
        nrows = int(math.ceil(n_items / ncols))
        denom_cols = ncols + max(ncols - 1, 0) * wspace
        denom_rows = nrows + max(nrows - 1, 0) * hspace
        if denom_cols <= 0 or denom_rows <= 0:
            continue

        cell_w = usable_w / denom_cols
        cell_h = usable_h / denom_rows
        cell_w_in = cell_w * fig_w_in
        cell_h_in = cell_h * fig_h_in
        score = min(cell_w_in, cell_h_in)
        if score <= 0:
            continue

        squareness = 1.0 - abs(cell_w_in - cell_h_in) / max(cell_w_in, cell_h_in)
        empties = nrows * ncols - n_items
        key = (score, squareness, -empties)
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "nrows": nrows,
                "ncols": ncols,
                "cell_w": cell_w,
                "cell_h": cell_h,
            }

    if best is None:
        return np.array([], dtype=object)

    gap_w = wspace * float(best["cell_w"])
    gap_h = hspace * float(best["cell_h"])

    return subdivide_axes(
        ax,
        int(best["nrows"]),
        int(best["ncols"]),
        wspace=gap_w,
        hspace=gap_h,
        left_pad=left_pad,
        right_pad=right_pad,
        bottom_pad=bottom_pad,
        top_pad=top_pad,
    )
