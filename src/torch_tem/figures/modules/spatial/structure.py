"""TEM Overview Rate Maps figure module.

Generates a multi-panel overview figure combining occupancy, spatial rate maps,
autocorrelograms, and a trajectory context panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import style_context
from torch_tem.figures.figures.insets import add_coverage_inset
from torch_tem.figures.plots import plot_autocorr2d, plot_time_colored_trajectory
from torch_tem.figures.plots.map import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.spatial import aggregate_rate_map, clip_unit_interval, select_feature_by_spatial_variance


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Generate TEM overview rate-maps figure from a combined rollout trace.

    Creates a 2x3 layout:
    - Top-left: Occupancy map with coverage
    - Top-middle: Spatial rate map for a selected g_inf feature
    - Top-right: Spatial rate map for a selected g_gen feature
    - Bottom-left: Time-colored trajectory with coverage inset
    - Bottom-middle: g_inf 2D spatial autocorrelogram
    - Bottom-right: g_gen 2D spatial autocorrelogram

    Args:
        trace: TraceTree with model outputs + world geometry + location IDs.
        ctx: Figure context (env_idx, freq_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with overview panels.
    """
    with style_context(ctx):
        fig = plt.figure(figsize=ctx.figsize)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = int(ctx.freq_idx)

        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        n_locations = len(world.locations)

        validate_freq_idx(trace, "output/inference/g_inf", freq_idx)
        validate_freq_idx(trace, "output/generative/g_gen", freq_idx)

        g_inf_time = get_multiscale(trace, "output/inference/g_inf", freq_idx)
        g_gen_time = get_multiscale(trace, "output/generative/g_gen", freq_idx)

        g_inf_env = g_inf_time[:, env_idx, :]
        g_gen_env = g_gen_time[:, env_idx, :]

        g_inf_rate_map, occupancy = aggregate_rate_map(g_inf_env, location_ids, n_locations)
        g_gen_rate_map, _ = aggregate_rate_map(g_gen_env, location_ids, n_locations)

        grid = fig.add_gridspec(2, 3)
        ax_occ = fig.add_subplot(grid[0, 0])
        ax_traj = fig.add_subplot(grid[1, 0])
        ax_inf_map = fig.add_subplot(grid[0, 1])
        ax_inf_auto = fig.add_subplot(grid[1, 1])
        ax_gen_map = fig.add_subplot(grid[0, 2])
        ax_gen_auto = fig.add_subplot(grid[1, 2])

        occupancy_values = occupancy.astype(float)
        occupancy_values[occupancy == 0] = np.nan
        occupancy_max = float(np.nanmax(occupancy_values)) if np.isfinite(occupancy_values).any() else 1.0
        plot_map(ax_occ, world, occupancy_values, min_val=0.0, max_val=occupancy_max, shape="square", location_cm="Greys")
        coverage = np.isfinite(occupancy_values).sum() / max(n_locations, 1)
        ax_occ.set_title(_append_context(f"Occupancy (coverage={coverage:.0%})", ctx))

        inf_feature_idx = select_feature_by_spatial_variance(g_inf_rate_map)
        gen_feature_idx = select_feature_by_spatial_variance(g_gen_rate_map)

        g_inf_values = clip_unit_interval(g_inf_rate_map[:, inf_feature_idx])
        g_gen_values = clip_unit_interval(g_gen_rate_map[:, gen_feature_idx])

        plot_map(ax_inf_map, world, g_inf_values, min_val=0.0, max_val=0.2, shape="square", location_cm="viridis")
        ax_inf_map.set_title(_append_context(f"g_inf[{inf_feature_idx}] Rate Map", ctx))

        plot_map(ax_gen_map, world, g_gen_values, min_val=0.0, max_val=0.2, shape="square", location_cm="viridis")
        ax_gen_map.set_title(_append_context(f"g_gen[{gen_feature_idx}] Rate Map", ctx))

        plot_autocorr2d(ax_inf_auto, world, g_inf_values, title=_append_context(f"g_inf[{inf_feature_idx}] 2D Autocorr", ctx))
        plot_autocorr2d(ax_gen_auto, world, g_gen_values, title=_append_context(f"g_gen[{gen_feature_idx}] 2D Autocorr", ctx))

        plot_time_colored_trajectory(ax_traj, world, location_ids)
        ax_traj.set_title(_append_context("Trajectory (time-colored)", ctx))
        add_coverage_inset(ax_traj, location_ids, n_locations)

        fig.tight_layout()
        return fig


def _append_context(title: str, ctx: FigureContext) -> str:
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    if ctx.global_step is not None:
        title += f" @ step {ctx.global_step}"
    return title
