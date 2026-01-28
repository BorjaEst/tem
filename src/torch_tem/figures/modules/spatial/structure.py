"""TEM Overview Rate Maps figure module.

Generates a multi-panel overview figure combining occupancy, spatial rate maps,
autocorrelograms, and a trajectory context panel.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx
from torch_tem.figures.utils.spatial import aggregate_rate_map, autocorr_2d


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
    style_ctx = _style_context(ctx)
    with style_ctx:
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
        plot_map(world, occupancy_values, ax=ax_occ, min_val=0.0, max_val=occupancy_max, shape="square", location_cm="Greys")
        coverage = np.isfinite(occupancy_values).sum() / max(n_locations, 1)
        ax_occ.set_title(_append_context(f"Occupancy (coverage={coverage:.0%})", ctx))

        inf_feature_idx = _select_feature_by_spatial_variance(g_inf_rate_map)
        gen_feature_idx = _select_feature_by_spatial_variance(g_gen_rate_map)

        g_inf_values = _clip_unit_interval(g_inf_rate_map[:, inf_feature_idx])
        g_gen_values = _clip_unit_interval(g_gen_rate_map[:, gen_feature_idx])

        plot_map(world, g_inf_values, ax=ax_inf_map, min_val=0.0, max_val=0.2, shape="square", location_cm="viridis")
        ax_inf_map.set_title(_append_context(f"g_inf[{inf_feature_idx}] Rate Map", ctx))

        plot_map(world, g_gen_values, ax=ax_gen_map, min_val=0.0, max_val=0.2, shape="square", location_cm="viridis")
        ax_gen_map.set_title(_append_context(f"g_gen[{gen_feature_idx}] Rate Map", ctx))

        _plot_autocorr_2d(ax_inf_auto, world, g_inf_values, f"g_inf[{inf_feature_idx}] 2D Autocorr", ctx)
        _plot_autocorr_2d(ax_gen_auto, world, g_gen_values, f"g_gen[{gen_feature_idx}] 2D Autocorr", ctx)

        _plot_time_colored_trajectory(world, location_ids, ax_traj)
        ax_traj.set_title(_append_context("Trajectory (time-colored)", ctx))
        _plot_coverage_inset(ax_traj, location_ids, n_locations)

        fig.tight_layout()
        return fig


def _plot_autocorr_2d(ax: plt.Axes, world, values: np.ndarray, title: str, ctx: FigureContext) -> None:
    acorr = autocorr_2d(values, world)
    if acorr.size == 0:
        ax.text(0.5, 0.5, "No autocorr data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return
    ax.imshow(acorr, cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower")
    ax.set_title(_append_context(title, ctx))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _plot_time_colored_trajectory(world, location_ids: list[int], ax: plt.Axes) -> None:
    if not location_ids:
        ax.text(0.5, 0.5, "No trajectory", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    values = np.full(len(world.locations), np.nan, dtype=float)
    plot_map(world, values, ax=ax, shape="square")

    coords = np.array([[world.locations[loc_id]["o"], world.locations[loc_id]["y"]] for loc_id in location_ids], dtype=float)
    if coords.shape[0] < 2:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, color="black")
        return

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    colors = np.linspace(0, 1, segments.shape[0])

    lc = LineCollection(segments, cmap="viridis", array=colors, linewidths=1.5)
    ax.add_collection(lc)
    ax.scatter(coords[0, 0], coords[0, 1], s=20, color="black", zorder=3)
    ax.scatter(coords[-1, 0], coords[-1, 1], s=20, color="white", edgecolor="black", zorder=3)

    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.axis("off")


def _plot_coverage_inset(ax: plt.Axes, location_ids: list[int], n_locations: int) -> None:
    if not location_ids or n_locations == 0:
        return
    coverage = _coverage_over_time(location_ids, n_locations)
    inset = inset_axes(ax, width="55%", height="35%", loc="lower left", borderpad=1.0)
    inset.plot(coverage, color="#333333", linewidth=1.2)
    inset.set_ylim(0.0, 1.0)
    inset.set_xlim(0, max(len(coverage) - 1, 1))
    inset.set_title("Coverage", fontsize=8)
    inset.set_xticks([])
    inset.set_yticks([0.0, 1.0])
    inset.set_yticklabels(["0", "1"], fontsize=7)


def _coverage_over_time(location_ids: list[int], n_locations: int) -> np.ndarray:
    visited = np.zeros(n_locations, dtype=bool)
    coverage = np.zeros(len(location_ids), dtype=float)
    for idx, loc_id in enumerate(location_ids):
        if 0 <= loc_id < n_locations:
            visited[loc_id] = True
        coverage[idx] = visited.sum() / max(n_locations, 1)
    return coverage


def _select_feature_by_spatial_variance(rate_map: np.ndarray) -> int:
    if rate_map.size == 0:
        return 0
    variances = np.nanvar(rate_map, axis=0)
    if not np.isfinite(variances).any():
        return 0
    return int(np.nanargmax(variances))


def _clip_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def _append_context(title: str, ctx: FigureContext) -> str:
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    if ctx.global_step is not None:
        title += f" @ step {ctx.global_step}"
    return title


def _get_default_style():
    try:
        from torch_tem.figures.style import StyleConfig

        return StyleConfig()
    except ImportError:
        return None


def _style_context(ctx: FigureContext):
    if getattr(ctx, "style", None):
        return (ctx.style or _get_default_style()).apply_context()
    return _noop_context()


class _noop_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
