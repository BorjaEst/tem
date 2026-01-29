from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def initialise_axes(
    ax: Optional[plt.Axes] = None,
    *,
    environment: Optional[object] = None,
    radius: Optional[float] = None,
    padding_scale: float = 2.0,
) -> plt.Axes:
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
