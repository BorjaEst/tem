"""Memory retrieval error map figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_dense, get_length, get_location_ids_for_env, get_world, validate_env_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render retrieval error aggregated by location.

    Args:
        trace: RolloutTrace containing reconstruction predictions.
        ctx: Figure context with env selection.

    Returns:
        Matplotlib Figure with error and occupancy panels.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)

        try:
            pred_steps = get_dense(trace, "output/reconstruction/y_p_inf/prediction")
        except ValueError:
            fig.suptitle("No reconstruction predictions available")
            return fig

        obs_steps = get_dense(trace, "world_step/observation")

        n_steps = min(pred_steps.shape[0], obs_steps.shape[0])
        pred_steps = pred_steps[:n_steps, env_idx, :]
        obs_steps = obs_steps[:n_steps, env_idx, :]
        errors = ((pred_steps - obs_steps) ** 2).mean(axis=1)

        error_map, occupancy = _aggregate_by_location(errors, location_ids[:n_steps], len(world.locations))
        grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9])
        ax_error = fig.add_subplot(grid[0, 0])
        ax_occ = fig.add_subplot(grid[0, 1])

        vmin, vmax = _robust_min_max(error_map)
        plot_map(world, error_map, ax=ax_error, min_val=vmin, max_val=vmax, shape="square")
        ax_error.set_title("Retrieval Error", fontsize=11)

        occ_values = occupancy.astype(float)
        occ_values[occ_values == 0] = np.nan
        occ_max = float(np.nanmax(occ_values)) if np.isfinite(occ_values).any() else 1.0
        plot_map(world, occ_values, ax=ax_occ, min_val=0.0, max_val=occ_max, shape="square")
        coverage = np.isfinite(occ_values).sum() / max(len(occ_values), 1)
        ax_occ.set_title(f"Occupancy (coverage={coverage:.0%})", fontsize=11)

        title = f"Retrieval Error by Location (env={env_idx})"
        title = _append_context(title, ctx)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def _aggregate_by_location(errors: np.ndarray, location_ids: list[int], n_locations: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-step errors into per-location means."""
    error_map = np.full(n_locations, np.nan, dtype=float)
    occupancy = np.zeros(n_locations, dtype=int)
    loc_ids = np.asarray(location_ids, dtype=int)
    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            occupancy[loc_id] = int(mask.sum())
            error_map[loc_id] = float(np.mean(errors[mask]))
    return error_map, occupancy


def _robust_min_max(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
    """Compute robust min/max for colormap scaling."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lower, upper])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


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
