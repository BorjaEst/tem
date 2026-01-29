from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from numpy.typing import NDArray

from torch_tem.figures.utils.actions import action_patch
from torch_tem.figures.utils.axes import initialise_axes


def plot_map(
    environment,
    values: NDArray,
    ax: Optional[plt.Axes] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    num_cols: int = 100,
    location_cm: str = "viridis",
    action_cm: str = "Pastel1",
    do_plot_actions: bool = False,
    shape: str = "circle",
    radius: Optional[float] = None,
) -> plt.Axes:
    """Render an environment map with per-location scalar values.

    Draws locations as colored circles/squares where color represents the value.
    Optionally overlays action arrows. Shiny locations are highlighted with red outlines.

    Args:
        environment: Environment object with .locations list and .n_locations, .n_actions.
        values: Per-location scalar values (shape: n_locations,).
        ax: Axes to draw on. If None, initializes new axes.
        min_val: Minimum value for colormap normalization (None = auto from values).
        max_val: Maximum value for colormap normalization (None = auto from values).
        num_cols: Number of discrete colors in the colormap.
        location_cm: Colormap name for location coloring.
        action_cm: Colormap name for action arrows.
        do_plot_actions: Whether to draw action transition arrows.
        shape: Location marker shape ("circle" or "square").
        radius: Radius/half-width of location markers (None = auto-scale).

    Returns:
        The axes object with the environment map rendered.
    """
    values = np.asarray(values, dtype=float)
    has_finite = values.size > 0 and np.isfinite(values).any()

    # Handle NaN values by using nanmin/nanmax when possible
    min_val = (np.nanmin(values) if has_finite else 0.0) if min_val is None else min_val
    max_val = (np.nanmax(values) if has_finite else 1.0) if max_val is None else max_val

    location_cm = cm.get_cmap(location_cm, num_cols)
    action_cm = cm.get_cmap(action_cm, max(getattr(environment, "n_actions", 0), 1))

    # Normalize values to colormap indices, handling NaN
    if values.size == 0:
        plotvals = np.zeros(values.shape)
        nan_mask = np.zeros(values.shape, dtype=bool)
    else:
        if max_val != min_val:
            plotvals = np.floor((values - min_val) / (max_val - min_val) * num_cols)
        else:
            plotvals = np.zeros(values.shape)

        # Replace NaN with a sentinel for visualization (use 0 for blank/first color)
        nan_mask = np.isnan(values)
        plotvals = np.where(nan_mask, 0, plotvals)

    # Auto-scale radius based on environment density
    if radius is None:
        radius = _default_radius(getattr(environment, "n_locations", 0))

    ax = initialise_axes(ax, environment=environment, radius=radius)

    # Store a mappable for colorbar grouping when using patch-based maps.
    ax._tem_colorbar_mappable = cm.ScalarMappable(
        norm=Normalize(vmin=min_val, vmax=max_val),
        cmap=location_cm,
    )

    location_patches: List = []
    action_patches: List = []

    # Draw locations
    for i, location in enumerate(environment.locations):
        is_nan = nan_mask[i] if nan_mask.size else False
        if is_nan:
            color = "#d9d9d9"
            edgecolor = "#444444"
            alpha = 1.0
        else:
            color = location_cm(int(plotvals[i]))
            edgecolor = None
            alpha = 1.0

        if shape == "square":
            patch = plt.Rectangle(
                (location["o"] - radius / 2, location["y"] - radius / 2),
                radius,
                radius,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=0.6 if edgecolor else 0.0,
            )
        else:  # circle
            patch = plt.Circle(
                (location["o"], location["y"]),
                radius,
                color=color,
                alpha=alpha,
                edgecolor=edgecolor,
                linewidth=0.6 if edgecolor else 0.0,
            )
        location_patches.append(patch)

        # Draw action arrows if requested
        if do_plot_actions:
            for a, action in enumerate(location["actions"]):
                if action["probability"] > 0:
                    locations_to = [environment.locations[loc_to] for loc_to in np.where(np.array(action["transition"]) > 0)[0]]
                    for loc_to in locations_to:
                        action_patches.append(action_patch(location, loc_to, radius, action_cm(action["id"])))

    # Highlight shiny locations with red outline
    for location in environment.locations:
        if location.get("shiny", False):
            if shape == "square":
                outline = plt.Rectangle((location["o"] - radius / 2, location["y"] - radius / 2), radius, radius, linewidth=1, facecolor="none", edgecolor=[1, 0, 0])
            else:
                outline = plt.Circle((location["o"], location["y"]), radius, linewidth=1, facecolor="none", edgecolor=[1, 0, 0])
            location_patches.append(outline)

    # Add all patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)

    return ax


def _default_radius(n_locations: int) -> float:
    if n_locations <= 0:
        return 0.05
    return 2 * (0.01 + 1 / (10 * np.sqrt(n_locations)))
