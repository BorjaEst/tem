"""Spatial autocorrelogram figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from torch_tem.diagnostics.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.plots import plot_autocorr2d, plot_time_colored_trajectory
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.spatial import (
    aggregate_rate_map,
    autocorr_extent,
    clip_unit_interval,
    infer_distance_scale,
    robust_min_max,
    select_top_k_by_spatial_variance,
    summarize_radial_autocorr,
)


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a multi-panel spatial summary for g_gen activity.

    Panels include:
    - Time-colored trajectory (top-left).
    - Population radial autocorr summary (bottom-left).
    - Top-3 g_gen rate maps (top row, columns 2-4).
    - Matching 2D autocorrelograms (bottom row, columns 2-4).

    Args:
        trace: TraceTree containing generative codes.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with spatial summary panels.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)
        grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.1, hspace=0.04)
        ax_traj = fig.add_subplot(grid[0, 0])
        ax_radial = fig.add_subplot(grid[1, 0])
        ax_maps = [fig.add_subplot(grid[0, idx]) for idx in range(1, 3)]
        ax_autos = [fig.add_subplot(grid[1, idx]) for idx in range(1, 3)]
        ax_traj.set_box_aspect(1)
        ax_radial.set_box_aspect(1)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/generative/g_gen", int(ctx.freq_idx))

        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        activity_steps = get_multiscale(trace, "output/generative/g_gen", freq_idx)
        activity_env = activity_steps[:, env_idx, :]
        rate_map, occupancy = aggregate_rate_map(activity_env, location_ids, len(world.locations))

        if rate_map.size == 0:
            for ax in [ax_traj, ax_radial, *ax_maps, *ax_autos]:
                _plot_missing(ax, "No rate map values available")
            return fig

        # fig.suptitle(_append_context("g_gen autocorr", ctx))

        plot_time_colored_trajectory(ax_traj, world, location_ids)
        ax_traj.set_title("Trajectory", pad=2)

        top_cells = select_top_k_by_spatial_variance(rate_map, occupancy, k=3, min_coverage=0.1)
        if top_cells.size == 0:
            top_cells = np.arange(min(3, rate_map.shape[1]))

        distance_scale = infer_distance_scale(world)
        centers, median, q25, q75 = summarize_radial_autocorr(rate_map, world, n_bins=12, min_coverage=0.1)
        if centers.size:
            if distance_scale is not None and distance_scale > 0:
                centers_plot = centers / distance_scale
                distance_label = "Distance (cells)"
            else:
                centers_plot = centers
                distance_label = "Distance (world units)"

            ax_radial.fill_between(centers_plot, q25, q75, color="#9ecae1", alpha=0.4)
            ax_radial.plot(centers_plot, median, color="#3182bd", linewidth=2)
            ax_radial.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
            ax_radial.set_xlabel(distance_label)
            ax_radial.set_ylabel("Autocorrelation")
            ax_radial.set_title("Radial autocorr", pad=2)
            if centers_plot.size:
                ax_radial.set_xlim(0.0, float(centers_plot[-1]))
                ax_radial.set_xticks(np.linspace(0.0, float(centers_plot[-1]), num=4))
        else:
            _plot_missing(ax_radial, "No radial autocorr")

        if top_cells.size:
            selected_values = [clip_unit_interval(rate_map[:, int(cell_idx)]) for cell_idx in top_cells]
            stacked = np.concatenate([vals[np.isfinite(vals)] for vals in selected_values if np.isfinite(vals).any()])
            if stacked.size:
                shared_min, shared_max = robust_min_max(stacked)
            else:
                shared_min, shared_max = 0.0, 1.0
        else:
            selected_values = []
            shared_min, shared_max = 0.0, 1.0

        if distance_scale is not None and distance_scale > 0:
            extent = autocorr_extent(world, units="cells")
        else:
            extent = autocorr_extent(world, units="world")

        for ax_map, ax_auto, cell_idx in zip(ax_maps, ax_autos, top_cells, strict=False):
            values = clip_unit_interval(rate_map[:, int(cell_idx)])
            plot_map(world, values, ax=ax_map, min_val=shared_min, max_val=shared_max, shape="square", location_cm="viridis")
            ax_map.set_title(f"cell {int(cell_idx)}", pad=2)
            plot_autocorr2d(ax_auto, world, values, extent=extent)

        for ax in ax_maps[len(top_cells) :]:
            _plot_missing(ax, "No additional cells")

        for ax in ax_autos[len(top_cells) :]:
            _plot_missing(ax, "No additional cells")

        rate_norm = colors.Normalize(vmin=shared_min, vmax=shared_max)
        rate_sm = cm.ScalarMappable(norm=rate_norm, cmap="viridis")
        fig.colorbar(rate_sm, ax=ax_maps, fraction=0.046, pad=0.02, shrink=0.5, label="Firing rate")

        autocorr_norm = colors.Normalize(vmin=-1.0, vmax=1.0)
        autocorr_sm = cm.ScalarMappable(norm=autocorr_norm, cmap="RdBu_r")
        fig.colorbar(autocorr_sm, ax=ax_autos, fraction=0.046, pad=0.02, shrink=0.5, label="Autocorrelation")
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
