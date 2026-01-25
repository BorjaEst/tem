"""Walk trajectories figure module.

Renders walk trajectories overlaid on the environment map with deterministic
positioning by default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import WorldTrace
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext


def plot(trace: WorldTrace, ctx: FigureContext, *, deterministic: bool = True, seed: int = 42, max_steps: int | None = None) -> Figure:
    """Generate walk trajectory figure.

    Renders walk trajectories overlaid on the environment map. By default,
    uses deterministic jitter for reproducible visualizations.

    Args:
        trace: WorldTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).
        deterministic: Whether to use deterministic jitter (default: True).
        seed: Random seed for deterministic jitter (default: 42).
        max_steps: Maximum walk steps to plot (None = plot all).

    Returns:
        matplotlib Figure showing walk trajectories.
    """
    # Select single environment if batch trace
    env = trace.environments[ctx.env_idx]
    walk = trace
    n_locs = env.n_locations

    # Extract location IDs for the selected environment
    location_ids = _extract_location_ids(walk, ctx.env_idx)
    if not location_ids:
        fig, ax = plt.subplots(figsize=ctx.figsize)
        ax.text(0.5, 0.5, "No location data available", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return fig

    # Set max_steps
    max_steps = len(location_ids) if max_steps is None else min(max_steps, len(location_ids))

    # Create figure
    fig, ax = plt.subplots(figsize=ctx.figsize)

    # Render environment base map
    values = np.ones(n_locs)  # Uniform coloring
    plot_map(env, values, ax=ax, do_plot_actions=False, shape="circle")

    # Plot walk trajectory with deterministic jitter
    _plot_walk_deterministic(env, location_ids, ax, max_steps, seed if deterministic else None)

    # Add title
    title = f"Walk Trajectory ({max_steps} steps)"
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    return fig


def _plot_walk_deterministic(
    environment,
    location_ids: list[int],
    ax: plt.Axes,
    max_steps: int,
    seed: int | None = None,
) -> None:
    """Plot walk trajectory with optional deterministic jitter.

    Args:
        environment: World object with location data.
        walk: Walk trajectory (list of [location_dict, observation, action]).
        ax: Axes to draw on.
        max_steps: Maximum steps to plot.
        seed: Random seed for jitter (None = use global random state).
    """
    # Initialize RNG for deterministic jitter
    rng = np.random.default_rng(seed) if seed is not None else np.random

    # Infer radius from existing patches
    location_patches = [patch for patch in ax.patches if isinstance(patch, (plt.Circle, plt.Rectangle))]
    if len(location_patches) > 0:
        last_patch = location_patches[-1]
        if isinstance(last_patch, plt.Circle):
            radius = last_patch.get_radius()
        else:  # Rectangle
            radius = last_patch.get_width()
    else:
        radius = 0.02  # Default fallback

    # Get initial position
    prev_loc = _location_coords(environment, location_ids[0])

    # Draw walk segments
    for step_i in range(1, max_steps):
        new_loc = _location_coords(environment, location_ids[step_i])

        # Add deterministic/seeded jitter to prevent overlapping lines
        jitter = 0.8 * (-radius + 2 * radius * rng.random(new_loc.shape))
        new_loc = new_loc + jitter

        # Color gradient: darker at start, lighter at end
        color_intensity = step_i / max_steps
        ax.plot([prev_loc[0], new_loc[0]], [prev_loc[1], new_loc[1]], color=[color_intensity] * 3, linewidth=1.5, alpha=0.7)

        prev_loc = new_loc


def _extract_location_ids(agent_trace, env_idx: int) -> list[int]:
    """Extract per-step location IDs for a single environment."""
    location_ids: list[int] = []
    for step in agent_trace:
        locations = step.locations
        if env_idx >= len(locations):
            continue
        location_ids.append(_coerce_location_id(locations[env_idx]))
    return location_ids


def _coerce_location_id(location) -> int:
    """Coerce location dicts/tensors/ints into a location id integer."""
    if isinstance(location, dict) and "id" in location:
        return int(location["id"])
    if hasattr(location, "item"):
        return int(location.item())
    return int(location)


def _location_coords(environment, location_id: int) -> np.ndarray:
    """Return (x, y) coordinates for a location id as a numpy array."""
    return np.array(
        [
            environment.locations[location_id]["o"],
            environment.locations[location_id]["y"],
        ]
    )
