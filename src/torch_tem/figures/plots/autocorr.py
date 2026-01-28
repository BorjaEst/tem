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

    Returns:
        The axes with the autocorrelogram rendered.
    """
    acorr = autocorr_2d(values, world)
    if acorr.size == 0:
        ax.text(0.5, 0.5, "No autocorr data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return ax
    ax.imshow(acorr, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    return ax
