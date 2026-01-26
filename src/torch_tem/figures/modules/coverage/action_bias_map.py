"""Action bias map figure module."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace, WorldTrace
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext


def plot(trace: WorldTrace | RolloutTrace, ctx: FigureContext) -> Figure:
    """Render an action bias map summarizing action entropy per location.

    Args:
        trace: WorldTrace or RolloutTrace providing actions and locations.
        ctx: Figure context with env selection and styling.

    Returns:
        Matplotlib Figure containing the action bias map.
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
        actions = _get_actions(world_trace, env_idx)

        bias = _compute_action_bias(location_ids, actions, world.n_locations, world.n_actions)
        max_val = float(np.nanmax(bias)) if np.isfinite(bias).any() else 1.0

        plot_map(
            world,
            bias,
            ax=ax,
            min_val=0.0,
            max_val=max_val,
            shape="square",
        )

        avg_bias = float(np.nanmean(bias)) if np.isfinite(bias).any() else 0.0
        title = f"Action Bias Map (mean bias={avg_bias:.2f})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig


def _compute_action_bias(
    location_ids: list[int],
    actions: list[int | None],
    n_locations: int,
    n_actions: int,
) -> np.ndarray:
    """Compute normalized action bias per location.

    Args:
        location_ids: Per-step location ids.
        actions: Per-step action ids.
        n_locations: Total number of locations.
        n_actions: Total number of actions.

    Returns:
        Array of bias values in [0, 1], with NaN for unvisited locations.
    """
    counts = np.zeros((n_locations, max(n_actions, 1)), dtype=int)
    for loc_id, action in zip(location_ids, actions):
        if action is None or not (0 <= loc_id < n_locations):
            continue
        if 0 <= action < n_actions:
            counts[loc_id, action] += 1

    totals = counts.sum(axis=1)
    bias = np.full(n_locations, np.nan, dtype=float)
    if n_actions <= 1:
        bias[totals > 0] = 1.0
        return bias

    log_base = math.log(n_actions)
    for loc_id, total in enumerate(totals):
        if total == 0:
            continue
        probs = counts[loc_id] / total
        entropy = -np.sum([p * math.log(p) for p in probs if p > 0]) / log_base
        bias[loc_id] = 1.0 - entropy
    return bias


def _coerce_world_trace(trace: WorldTrace | RolloutTrace) -> WorldTrace:
    """Normalize to a WorldTrace instance."""
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


def _get_actions(trace: WorldTrace, env_idx: int) -> list[int | None]:
    """Get per-step actions for the selected environment."""
    actions: list[int | None] = []
    for step_actions in trace.actions:
        if env_idx < len(step_actions):
            actions.append(step_actions[env_idx])
        else:
            actions.append(None)
    return actions


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
