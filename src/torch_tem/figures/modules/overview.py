"""TEM circuit overview figure module.

Provides a human-readable, multi-panel snapshot of LEC, MEC, and HPC behavior
for a single environment, plus observations and memory.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.trace_access import (
    get_hpc_memory,
    get_lec_cells,
    get_length,
    get_location_ids_for_env,
    get_mec_ovc_modules,
    get_multiscale,
    get_n_freq,
    get_observation_ids_for_env,
    get_world,
    validate_env_idx,
)
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.plots import plot_time_colored_trajectory
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.spatial import aggregate_rate_map, clip_unit_interval, select_feature_by_spatial_variance


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a multi-panel overview of LEC, MEC, and HPC signals.

    Args:
        trace: TraceTree with model outputs, state, and world steps.
        ctx: Figure context (env_idx, figsize, style, split_name, etc.).

    Returns:
        Matplotlib Figure with the overview panels.
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

        grid = fig.add_gridspec(2, 3)
        obs_lec = grid[0, 1:].subgridspec(2, 1, height_ratios=[2, 2], hspace=0.05)
        ax_mem = fig.add_subplot(grid[0, 0])
        ax_traj = fig.add_subplot(grid[1, 0])
        ax_obs = fig.add_subplot(obs_lec[0])
        ax_lec = fig.add_subplot(obs_lec[1], sharex=ax_obs)
        ax_mec = fig.add_subplot(grid[1, 1])
        ax_hpc = fig.add_subplot(grid[1, 2])

        _plot_observation_ids(trace, env_idx, ax_obs, ctx)
        _plot_lec_cells(trace, env_idx, ax_lec, ctx)
        plot_time_colored_trajectory(ax_traj, world, location_ids)
        ax_traj.set_title(_append_context("Trajectory (time-colored)", ctx))
        _plot_mec_rate_maps(trace, env_idx, world, location_ids, ax_mec, ctx)
        _plot_hpc_rate_maps(trace, env_idx, world, location_ids, ax_hpc, ctx)
        _plot_hpc_memory(trace, env_idx, ax_mem, ctx)

        fig.tight_layout()
        return fig


def _plot_observation_ids(trace: TraceTree, env_idx: int, ax: plt.Axes, ctx: FigureContext) -> None:
    """Plot observation ids over time for a selected environment."""
    obs_ids = get_observation_ids_for_env(trace, env_idx)
    if not obs_ids:
        _plot_missing(ax, "No observations")
        return
    unique_ids = sorted(set(obs_ids))
    row_map = {obs_id: idx for idx, obs_id in enumerate(unique_ids)}
    one_hot = np.zeros((len(unique_ids), len(obs_ids)), dtype=float)
    for t, obs_id in enumerate(obs_ids):
        one_hot[row_map[obs_id], t] = 1.0
    ax.imshow(one_hot, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_ylabel("Obs id")
    if len(unique_ids) <= 20:
        ax.set_yticks(np.arange(len(unique_ids)))
        ax.set_yticklabels([str(obs_id) for obs_id in unique_ids])
    else:
        ax.set_yticks([])
    ax.set_xticks([])
    title = f"Observations ids and LEC x_inf f{ctx.freq_idx} (all cells)"
    ax.set_title(_append_context(title, ctx))


def _plot_lec_cells(trace: TraceTree, env_idx: int, ax: plt.Axes, ctx: FigureContext) -> None:
    """Plot LEC cell activations for a selected frequency."""
    try:
        n_freq = get_n_freq(trace, "state/lec/cells")
    except ValueError:
        _plot_missing(ax, "Missing LEC cells")
        return

    freq_idx = int(ctx.freq_idx)
    if not (0 <= freq_idx < n_freq):
        _plot_missing(ax, f"freq_idx {freq_idx} out of range [0, {n_freq})")
        return

    try:
        activity = get_lec_cells(trace, freq_idx)[:, env_idx, :]
    except (ValueError, IndexError):
        _plot_missing(ax, "Missing LEC activity")
        return

    if activity.size == 0:
        _plot_missing(ax, "No LEC activity")
        return

    block = activity.T
    if block.size == 0:
        _plot_missing(ax, "No LEC activity")
        return

    ax.imshow(block, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Time")
    ax.set_ylabel("Cells")
    ax.set_yticks([])
    ax.set_title(_append_context(title=None, ctx=ctx))


def _plot_mec_rate_maps(trace: TraceTree, env_idx: int, world: object, location_ids: list[int], ax: plt.Axes, ctx: FigureContext) -> None:
    """Plot a MEC grid-cell rate map for the selected frequency."""
    try:
        n_freq = get_n_freq(trace, "output/inference/g_inf")
    except ValueError:
        _plot_missing(ax, "Missing MEC g_inf")
        return

    n_ovc_modules = get_mec_ovc_modules(trace) or 0
    n_grid = max(n_freq - n_ovc_modules, 0)
    freq_idx = int(ctx.freq_idx)
    if n_grid == 0:
        _plot_missing(ax, "No MEC grid modules")
        return
    if not (0 <= freq_idx < n_grid):
        _plot_missing(ax, f"freq_idx {freq_idx} out of grid range [0, {n_grid})")
        return

    try:
        activity = get_multiscale(trace, "output/inference/g_inf", freq_idx)[:, env_idx, :]
    except (ValueError, IndexError):
        _plot_missing(ax, "Missing MEC grid activity")
        return

    rate_map, _ = aggregate_rate_map(activity, location_ids, len(world.locations))
    cell_idx, values = _select_rate_map_cell(rate_map)
    if values is None:
        _plot_missing(ax, "No MEC grid activity")
        return

    max_val = float(np.max(values[~np.isnan(values)])) if np.isfinite(values).any() else 1.0
    max_val = max(max_val, 0.1)
    plot_map(world, values, ax=ax, min_val=0.0, max_val=max_val, shape="square", location_cm="cividis")
    ax.set_title(_append_context(f"MEC g_inf f{freq_idx} (cell {cell_idx}, OVC excluded)", ctx))


def _plot_hpc_rate_maps(trace: TraceTree, env_idx: int, world: object, location_ids: list[int], ax: plt.Axes, ctx: FigureContext) -> None:
    """Plot a HPC place-like rate map for the selected frequency."""
    try:
        n_freq = get_n_freq(trace, "output/inference/p_inf")
    except ValueError:
        _plot_missing(ax, "Missing HPC p_inf")
        return

    freq_idx = int(ctx.freq_idx)
    if not (0 <= freq_idx < n_freq):
        _plot_missing(ax, f"freq_idx {freq_idx} out of range [0, {n_freq})")
        return

    try:
        activity = get_multiscale(trace, "output/inference/p_inf", freq_idx)[:, env_idx, :]
    except (ValueError, IndexError):
        _plot_missing(ax, "Missing HPC activity")
        return

    rate_map, _ = aggregate_rate_map(activity, location_ids, len(world.locations))
    cell_idx, values = _select_rate_map_cell(rate_map)
    if values is None:
        _plot_missing(ax, "No HPC activity")
        return

    max_val = float(np.max(values[~np.isnan(values)])) if np.isfinite(values).any() else 1.0
    max_val = max(max_val, 0.1)
    plot_map(world, values, ax=ax, min_val=0.0, max_val=0.4, shape="square", location_cm="cividis")
    ax.set_title(_append_context(f"HPC p_inf f{freq_idx} (cell {cell_idx})", ctx))


def _plot_hpc_memory(trace: TraceTree, env_idx: int, ax: plt.Axes, ctx: FigureContext) -> None:
    """Plot the hierarchical HPC memory matrix at the final step."""
    try:
        memory = get_hpc_memory(trace, memory_idx=0)
    except (ValueError, IndexError):
        _plot_missing(ax, "Missing HPC memory")
        return

    if memory.size == 0:
        _plot_missing(ax, "Missing HPC memory")
        return

    matrix = memory[-1, env_idx]
    if matrix.size == 0:
        _plot_missing(ax, "Empty memory")
        return

    max_val = float(np.max(np.abs(matrix))) if np.isfinite(matrix).any() else 1.0
    max_val = max(max_val, 1e-6)
    ax.imshow(matrix, cmap="bwr", vmin=-max_val, vmax=max_val)
    ax.set_title(_append_context("HPC memory (hierarchical)", ctx))
    ax.set_xticks([])
    ax.set_yticks([])


def _select_rate_map_cell(rate_map: np.ndarray) -> tuple[int, Optional[np.ndarray]]:
    """Select the most spatially varying cell and return its values."""
    if rate_map.size == 0:
        return 0, None
    cell_idx = select_feature_by_spatial_variance(rate_map)
    values = clip_unit_interval(rate_map[:, cell_idx])
    if values.size == 0:
        return cell_idx, None
    return cell_idx, values


def _select_top_k_temporal(activity: np.ndarray, k: int) -> np.ndarray:
    """Select top-k cells by temporal variance."""
    if activity.size == 0:
        return np.array([], dtype=int)
    variances = np.var(activity, axis=0)
    k = max(0, min(k, variances.shape[0]))
    if k == 0:
        return np.array([], dtype=int)
    return np.argsort(-variances)[:k]


def _append_context(title: Optional[str], ctx: FigureContext) -> str:
    """Append split name and global step metadata to titles."""
    if title is None:
        return ""
    if ctx.split_name:
        return title + f" - {ctx.split_name}"
    if ctx.global_step is not None:
        return title + f" @ step {ctx.global_step}"


def _plot_missing(ax: plt.Axes, message: str) -> None:
    """Render a centered missing-data message."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.axis("off")


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
