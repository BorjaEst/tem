"""Autocorrelogram plotting panels."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from torch_tem.figures.utils.spatial import autocorr_2d


def plot_autocorr2d(
    ax: plt.Axes,
    world: object,
    values: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
    extent: tuple[float, float, float, float] | None = None,
    show_ticks: bool = False,
    x_label: str | None = None,
    y_label: str | None = None,
) -> plt.Axes:
    """Plot a 2D spatial autocorrelogram on an axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        values: Per-location values.
        title: Optional title string.
        cmap: Colormap for the autocorrelogram.
        vmin: Minimum colormap value.
        vmax: Maximum colormap value.
        extent: Optional imshow extent (xmin, xmax, ymin, ymax).
        show_ticks: Whether to show axis ticks for distance context.
        x_label: Optional x-axis label.
        y_label: Optional y-axis label.

    Returns:
        The axes with the autocorrelogram rendered.
    """
    acorr = autocorr_2d(values, world)
    if acorr.size == 0:
        ax.text(0.5, 0.5, "No autocorr data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return ax
    ax.imshow(acorr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower", extent=extent)
    if title:
        ax.set_title(title)
    if show_ticks:
        if extent is not None:
            ax.set_xticks([extent[0], 0.0, extent[1]])
            ax.set_yticks([extent[2], 0.0, extent[3]])
        else:
            mid_x = (acorr.shape[1] - 1) / 2
            mid_y = (acorr.shape[0] - 1) / 2
            ax.set_xticks([0.0, mid_x, acorr.shape[1] - 1])
            ax.set_yticks([0.0, mid_y, acorr.shape[0] - 1])
        ax.tick_params(labelsize=8)
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_aspect("equal")
    return ax
