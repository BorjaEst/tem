"""Walk trajectories figure module.

Renders walk trajectories overlaid on the environment map with deterministic
positioning by default.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.figures.core.data_trace import DataTrace
from torch_tem.figures.core.registry import FigureContext
from torch_tem.figures.primitives import initialise_axes, plot_map


def plot(
    trace: DataTrace,
    ctx: FigureContext,
    deterministic: bool = True,
    seed: int = 42,
    max_steps: int | None = None,
) -> Figure:
    """Generate walk trajectory figure.

    Renders walk trajectories overlaid on the environment map. By default,
    uses deterministic jitter for reproducible visualizations.

    Args:
        trace: DataTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).
        deterministic: Whether to use deterministic jitter (default: True).
        seed: Random seed for deterministic jitter (default: 42).
        max_steps: Maximum walk steps to plot (None = plot all).

    Returns:
        matplotlib Figure showing walk trajectories.
    """
    # Select single environment if batch trace
    if trace.batch_size > 1:
        trace = trace.select_env(ctx.env_idx)

    env = trace.worlds[0]
    walk = trace.walks[0]
    n_locs = env.n_locations

    # Set max_steps
    max_steps = len(walk) if max_steps is None else min(max_steps, len(walk))

    # Create figure
    fig, ax = plt.subplots(figsize=ctx.figsize)

    # Render environment base map
    values = np.ones(n_locs)  # Uniform coloring
    plot_map(env, values, ax=ax, do_plot_actions=False, shape="circle")

    # Plot walk trajectory with deterministic jitter
    _plot_walk_deterministic(env, walk, ax, max_steps, seed if deterministic else None)

    # Add title
    title = f"Walk Trajectory ({max_steps} steps)"
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    ax.set_title(title, fontsize=14, pad=10)

    plt.tight_layout()
    return fig


def _plot_walk_deterministic(
    environment,
    walk: list,
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
    location_patches = [
        patch for patch in ax.patches if isinstance(patch, (plt.Circle, plt.Rectangle))
    ]
    if len(location_patches) > 0:
        last_patch = location_patches[-1]
        if isinstance(last_patch, plt.Circle):
            radius = last_patch.get_radius()
        else:  # Rectangle
            radius = last_patch.get_width()
    else:
        radius = 0.02  # Default fallback

    # Get initial position
    prev_loc = np.array(
        [
            environment.locations[walk[0][0]["id"]]["o"],
            environment.locations[walk[0][0]["id"]]["y"],
        ]
    )

    # Draw walk segments
    for step_i in range(1, max_steps):
        new_loc = np.array(
            [
                environment.locations[walk[step_i][0]["id"]]["o"],
                environment.locations[walk[step_i][0]["id"]]["y"],
            ]
        )

        # Add deterministic/seeded jitter to prevent overlapping lines
        jitter = 0.8 * (-radius + 2 * radius * rng.random(new_loc.shape))
        new_loc = new_loc + jitter

        # Color gradient: darker at start, lighter at end
        color_intensity = step_i / max_steps
        ax.plot(
            [prev_loc[0], new_loc[0]],
            [prev_loc[1], new_loc[1]],
            color=[color_intensity] * 3,
            linewidth=1.5,
            alpha=0.7,
        )

        prev_loc = new_loc
