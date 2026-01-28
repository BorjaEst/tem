"""Spatial autocorrelogram figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.plots import plot_autocorr2d, plot_time_colored_trajectory
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.spatial import aggregate_rate_map, clip_unit_interval, robust_min_max, select_top_k_by_spatial_variance, summarize_radial_autocorr


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a multi-panel spatial summary for place-like activity.

    Panels include:
    - Time-colored trajectory.
    - 2D autocorrelogram for a representative cell.
    - Population radial autocorr summary (median + IQR band).
    - Top-k exemplar rate maps.

    Args:
        trace: TraceTree containing inference codes.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with spatial summary panels.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)
        grid = fig.add_gridspec(2, 3)
        ax_traj = fig.add_subplot(grid[0, 0])
        ax_autocorr2d = fig.add_subplot(grid[0, 1])
        ax_radial = fig.add_subplot(grid[0, 2])
        ax_maps = [fig.add_subplot(grid[1, idx]) for idx in range(3)]

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/inference/p_inf", int(ctx.freq_idx))

        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
        activity_env = activity_steps[:, env_idx, :]
        rate_map, occupancy = aggregate_rate_map(activity_env, location_ids, len(world.locations))

        if rate_map.size == 0:
            for ax in [ax_traj, ax_autocorr2d, ax_radial, *ax_maps]:
                _plot_missing(ax, "No rate map values available")
            return fig

        plot_time_colored_trajectory(ax_traj, world, location_ids)
        ax_traj.set_title(_append_context("Trajectory", ctx))

        top_cells = select_top_k_by_spatial_variance(rate_map, occupancy, k=3, min_coverage=0.1)
        if top_cells.size == 0:
            top_cells = np.arange(min(3, rate_map.shape[1]))

        if top_cells.size:
            cell_idx = int(top_cells[0])
            values = clip_unit_interval(rate_map[:, cell_idx])
            plot_autocorr2d(
                ax_autocorr2d,
                world,
                values,
                title=_append_context(f"2D Autocorr (cell {cell_idx})", ctx),
            )
        else:
            _plot_missing(ax_autocorr2d, "No valid cells")

        centers, median, q25, q75 = summarize_radial_autocorr(
            rate_map,
            world,
            n_bins=12,
            min_coverage=0.1,
        )
        if centers.size:
            ax_radial.fill_between(centers, q25, q75, color="#9ecae1", alpha=0.4)
            ax_radial.plot(centers, median, color="#3182bd", linewidth=2)
            ax_radial.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
            ax_radial.set_xlabel("Distance")
            ax_radial.set_ylabel("Autocorrelation")
            ax_radial.set_title(_append_context("Radial Autocorr (median ± IQR)", ctx))
        else:
            _plot_missing(ax_radial, "No radial autocorr")

        for ax, cell_idx in zip(ax_maps, top_cells, strict=False):
            values = clip_unit_interval(rate_map[:, int(cell_idx)])
            min_val, max_val = robust_min_max(values)
            plot_map(
                world,
                values,
                ax=ax,
                min_val=min_val,
                max_val=max_val,
                shape="square",
                location_cm="viridis",
            )
            ax.set_title(_append_context(f"Rate Map (cell {int(cell_idx)})", ctx))

        for ax in ax_maps[len(top_cells) :]:
            _plot_missing(ax, "No additional cells")

        fig.tight_layout()
        return fig


def _append_context(title: str, ctx: FigureContext) -> str:
    """Append optional split name and global step metadata."""
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    if ctx.global_step is not None:
        title += f" @ step {ctx.global_step}"
    return title


def _plot_missing(ax: plt.Axes, message: str) -> None:
    """Render a centered missing-data message."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.axis("off")


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
