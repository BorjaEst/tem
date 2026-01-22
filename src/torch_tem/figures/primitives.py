"""Reusable plotting primitives for TEM environment visualization.

Migrated from torch_tem.figures.__init__.py to establish clear separation
between primitives (low-level drawing utilities) and figure modules (multi-panel
compositions).

These utilities are environment-centric: they render spatial layouts, walks,
and per-location scalar fields. They operate on matplotlib axes objects and
can be composed to build complete figures.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from numpy.typing import NDArray


def initialise_axes(ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Initialize or configure axes for environment map plotting.
    
    Sets up axes with:
        - Limits [0, 1] x [0, 1] (normalized environment coordinates)
        - Equal aspect ratio (square axes)
        - Inverted y-axis (graphics convention: y increases downward)
        - Hidden axis labels and ticks
    
    Args:
        ax: Existing axes to configure. If None, creates new figure and axes.
    
    Returns:
        Configured matplotlib Axes object.
    """
    if ax is None:
        plt.figure()
        ax = plt.axes()
    
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect(1)
    ax.invert_yaxis()  # Y increases downward (standard graphics convention)
    ax.axis("off")
    
    return ax


def action_patch(location_from: dict, location_to: dict, radius: float, colour) -> plt.Polygon:
    """Create a triangular patch representing an action/transition arrow.
    
    Args:
        location_from: Source location dict with keys "o" (x), "y" (y), "id".
        location_to: Destination location dict with keys "o" (x), "y" (y), "id".
        radius: Radius of location circles (for arrow scaling).
        colour: Matplotlib color spec for the arrow.
    
    Returns:
        plt.Polygon representing an arrow from location_from toward location_to.
    """
    if location_to["id"] == location_from["id"]:
        # Self-transition: arrow points down (pi/2 in inverted y-axis)
        a_dir = np.pi / 2
        xdat = location_from["o"] + radius * np.array([
            2 * np.cos(a_dir - np.pi / 6),
            2 * np.cos(a_dir + np.pi / 6),
            3 * np.cos(a_dir)
        ])
        ydat = location_from["y"] - radius * 3 + radius * np.array([
            2 * np.sin(a_dir - np.pi / 6),
            2 * np.sin(a_dir + np.pi / 6),
            3 * np.sin(a_dir)
        ])
    else:
        # Directed transition: compute direction vector
        xvec = location_to["o"] - location_from["o"]
        yvec = location_from["y"] - location_to["y"]  # Inverted y
        a_dir = np.arctan2(-yvec, xvec)
        
        xdat = location_from["o"] + radius * np.array([
            2 * np.cos(a_dir - np.pi / 6),
            2 * np.cos(a_dir + np.pi / 6),
            3 * np.cos(a_dir)
        ])
        ydat = location_from["y"] + radius * np.array([
            2 * np.sin(a_dir - np.pi / 6),
            2 * np.sin(a_dir + np.pi / 6),
            3 * np.sin(a_dir)
        ])
    
    return plt.Polygon(np.stack([xdat, ydat], axis=1), color=colour)


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
    min_val = np.min(values) if min_val is None else min_val
    max_val = np.max(values) if max_val is None else max_val
    
    location_cm = cm.get_cmap(location_cm, num_cols)
    action_cm = cm.get_cmap(action_cm, environment.n_actions)
    
    # Normalize values to colormap indices
    if max_val != min_val:
        plotvals = np.floor((values - min_val) / (max_val - min_val) * num_cols)
    else:
        plotvals = np.ones(values.shape)
    
    # Auto-scale radius based on environment density
    if radius is None:
        radius = 2 * (0.01 + 1 / (10 * np.sqrt(environment.n_locations)))
    
    ax = initialise_axes(ax)
    
    location_patches: List = []
    action_patches: List = []
    
    # Draw locations
    for i, location in enumerate(environment.locations):
        if shape == "square":
            patch = plt.Rectangle(
                (location["o"] - radius / 2, location["y"] - radius / 2),
                radius, radius,
                color=location_cm(int(plotvals[i]))
            )
        else:  # circle
            patch = plt.Circle(
                (location["o"], location["y"]),
                radius,
                color=location_cm(int(plotvals[i]))
            )
        location_patches.append(patch)
        
        # Draw action arrows if requested
        if do_plot_actions:
            for a, action in enumerate(location["actions"]):
                if action["probability"] > 0:
                    locations_to = [
                        environment.locations[loc_to]
                        for loc_to in np.where(np.array(action["transition"]) > 0)[0]
                    ]
                    for loc_to in locations_to:
                        action_patches.append(
                            action_patch(location, loc_to, radius, action_cm(action["id"]))
                        )
    
    # Highlight shiny locations with red outline
    for location in environment.locations:
        if location.get("shiny", False):
            if shape == "square":
                outline = plt.Rectangle(
                    (location["o"] - radius / 2, location["y"] - radius / 2),
                    radius, radius,
                    linewidth=1, facecolor="none", edgecolor=[1, 0, 0]
                )
            else:
                outline = plt.Circle(
                    (location["o"], location["y"]),
                    radius,
                    linewidth=1, facecolor="none", edgecolor=[1, 0, 0]
                )
            location_patches.append(outline)
    
    # Add all patches to axes
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    
    return ax


def plot_walk(
    environment,
    walk: List,
    max_steps: Optional[int] = None,
    n_steps: int = 1,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Overlay a walk trajectory onto an environment map.
    
    Draws line segments connecting visited locations with color gradient
    indicating temporal progression (darker = earlier, lighter = later).
    
    Args:
        environment: Environment object with .locations list.
        walk: Walk trajectory as list of (location_dict, observation, action) tuples.
        max_steps: Maximum number of steps to plot (None = plot all).
        n_steps: Plot every n_steps-th step (1 = plot all).
        ax: Axes to draw on. If None, initializes new axes.
    
    Returns:
        The axes object with the walk overlay rendered.
    """
    max_steps = len(walk) if max_steps is None else min(max_steps, len(walk))
    
    if ax is None:
        ax = initialise_axes(ax)
    
    # Find circle patches on current axis to infer radius
    location_patches = [
        patch_i for patch_i, patch in enumerate(ax.patches)
        if isinstance(patch, (plt.Circle, plt.Rectangle))
    ]
    
    if len(location_patches) > 0:
        last_patch = ax.patches[location_patches[-1]]
        if isinstance(last_patch, plt.Circle):
            radius = last_patch.get_radius()
        else:  # Rectangle
            radius = last_patch.get_width()
    else:
        radius = 0.02  # Default fallback
    
    # Get initial position
    prev_loc = np.array([
        environment.locations[walk[0][0]["id"]]["o"],
        environment.locations[walk[0][0]["id"]]["y"]
    ])
    
    # Draw walk segments
    for step_i in range(1, max_steps, n_steps):
        new_loc = np.array([
            environment.locations[walk[step_i][0]["id"]]["o"],
            environment.locations[walk[step_i][0]["id"]]["y"]
        ])
        
        # Add jitter to prevent overlapping lines
        new_loc = new_loc + 0.8 * (-radius + 2 * radius * np.random.rand(*new_loc.shape))
        
        # Color gradient: darker at start, lighter at end
        color_intensity = step_i / max_steps
        plt.plot(
            [prev_loc[0], new_loc[0]],
            [prev_loc[1], new_loc[1]],
            color=[color_intensity] * 3
        )
        
        prev_loc = new_loc
    
    return ax


def plot_actions(
    environment,
    field: str = "probability",
    ax: Optional[plt.Axes] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    num_cols: int = 100,
    action_cm: str = "viridis",
) -> plt.Axes:
    """Visualize action properties across the environment.
    
    Draws action arrows colored by a specified field (e.g., "probability").
    Locations are rendered as black circles.
    
    Args:
        environment: Environment object with .locations list and .n_locations.
        field: Action property to visualize (e.g., "probability").
        ax: Axes to draw on. If None, initializes new axes.
        min_val: Minimum value for colormap normalization (None = auto).
        max_val: Maximum value for colormap normalization (None = auto).
        num_cols: Number of discrete colors in the colormap.
        action_cm: Colormap name for action coloring.
    
    Returns:
        The axes object with action visualization rendered.
    """
    # Auto-scale min/max from field values
    if min_val is None:
        min_val = min(
            action[field]
            for location in environment.locations
            for action in location["actions"]
        )
    if max_val is None:
        max_val = max(
            action[field]
            for location in environment.locations
            for action in location["actions"]
        )
    
    action_cm = cm.get_cmap(action_cm, num_cols)
    radius = 2 * (0.01 + 1 / (10 * np.sqrt(environment.n_locations)))
    
    ax = initialise_axes(ax)
    
    location_patches: List = []
    action_patches: List = []
    
    for location in environment.locations:
        # Draw location as black circle
        location_patches.append(
            plt.Circle((location["o"], location["y"]), radius, color=[0, 0, 0])
        )
        
        # Draw colored action arrows
        for action in location["actions"]:
            if action["probability"] > 0:
                # Normalize field value to colormap index
                if max_val != min_val:
                    color_idx = int(np.floor((action[field] - min_val) / (max_val - min_val) * num_cols))
                else:
                    color_idx = 0
                action_colour = action_cm(color_idx)
                
                # Find destination locations
                locations_to = [
                    environment.locations[loc_to]
                    for loc_to in np.where(np.array(action["transition"]) > 0)[0]
                ]
                
                for loc_to in locations_to:
                    action_patches.append(
                        action_patch(location, loc_to, radius, action_colour)
                    )
    
    for patch in location_patches + action_patches:
        ax.add_patch(patch)
    
    return ax
