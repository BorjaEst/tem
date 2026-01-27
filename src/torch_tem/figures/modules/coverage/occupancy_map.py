"""Occupancy map figure module."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_world, validate_env_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
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

        if get_length(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        n_locations = len(world.locations)

        occupancy = _compute_occupancy(location_ids, n_locations)
        values = occupancy.astype(float)
        values[occupancy == 0] = np.nan
        max_val = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0

        plot_map(world, values, ax=ax, min_val=0.0, max_val=max_val, shape="square")

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
