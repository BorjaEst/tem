"""Occupancy map figure module."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace, WorldTrace
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext


def plot(trace: WorldTrace | RolloutTrace, ctx: FigureContext) -> Figure:
    """Render an occupancy map for a selected environment.

    Args:
        trace: WorldTrace or RolloutTrace providing locations and environments.
        ctx: Figure context with env selection and styling.

    Returns:
        Matplotlib Figure containing the occupancy map.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        world_trace = _coerce_world_trace(trace)
        if world_trace.batch_size == 0 or len(world_trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = _validate_env_idx(world_trace, int(ctx.env_idx))
        world = _get_world(world_trace, env_idx)
        location_ids = _get_location_ids(world_trace, env_idx)
        n_locations = len(world.locations)

        occupancy = _compute_occupancy(location_ids, n_locations)
        values = occupancy.astype(float)
        values[occupancy == 0] = np.nan
        max_val = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0

        plot_map(
            world,
            values,
            ax=ax,
            min_val=0.0,
            max_val=max_val,
            shape="square",
        )

        coverage = np.isfinite(values).sum() / max(n_locations, 1)
        title = f"Occupancy Map (coverage={coverage:.0%})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig


def _compute_occupancy(location_ids: Iterable[int], n_locations: int) -> np.ndarray:
    """Compute visit counts per location.

    Args:
        location_ids: Sequence of per-step location ids.
        n_locations: Total number of locations in the environment.

    Returns:
        Array of visit counts per location.
    """
    occupancy = np.zeros(n_locations, dtype=int)
    for loc_id in location_ids:
        if 0 <= loc_id < n_locations:
            occupancy[loc_id] += 1
    return occupancy


def _coerce_world_trace(trace: WorldTrace | RolloutTrace) -> WorldTrace:
    """Normalize to a WorldTrace instance.

    Args:
        trace: Input trace (WorldTrace or RolloutTrace).

    Returns:
        WorldTrace view of the data.
    """
    if isinstance(trace, RolloutTrace):
        return trace.world_step
    if isinstance(trace, WorldTrace):
        return trace
    raise ValueError("Expected WorldTrace or RolloutTrace")


def _get_world(trace: WorldTrace, env_idx: int):
    """Get the World for the selected environment index."""
    if not trace.environments:
        raise ValueError("WorldTrace has no environments")
    return trace.environments[env_idx]


def _get_location_ids(trace: WorldTrace, env_idx: int) -> list[int]:
    """Get per-step location ids for the selected environment."""
    location_ids = trace.location_ids
    if not location_ids:
        raise ValueError("WorldTrace has no location ids")
    return location_ids[env_idx]


def _validate_env_idx(trace: WorldTrace, env_idx: int) -> int:
    """Validate the selected environment index."""
    if not (0 <= env_idx < trace.batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {trace.batch_size})")
    return env_idx


def _append_context(title: str, ctx: FigureContext) -> str:
    """Append optional split name and global step metadata."""
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    if ctx.global_step is not None:
        title += f" @ step {ctx.global_step}"
    return title


def _get_default_style():
    """Return the default style config if available."""
    try:
        from torch_tem.figures.style import StyleConfig

        return StyleConfig()
    except ImportError:
        return None


def _style_context(ctx: FigureContext):
    """Return a style context manager when style is provided."""
    if getattr(ctx, "style", None):
        return (ctx.style or _get_default_style()).apply_context()
    return _noop_context()


class _noop_context:
    """No-op context manager for style handling."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
