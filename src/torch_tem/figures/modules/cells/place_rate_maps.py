"""Place-cell rate map figure modules."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_n_freq, get_world, validate_env_idx, validate_freq_idx


@dataclass(frozen=True)
class _RateMapSelection:
    cell_indices: np.ndarray
    rate_maps: np.ndarray
    occupancy: np.ndarray


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render single-frequency place-cell rate maps.

    Args:
        trace: TraceTree with world and inference outputs.
        ctx: Figure context (env_idx, freq_idx, figsize, style).

    Returns:
        matplotlib Figure.
    """
    style_ctx = (ctx.style or _get_default_style()).apply_context() if getattr(ctx, "style", None) else _noop_context()

    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = int(ctx.freq_idx)

        world = get_world(trace, env_idx)
        selection = _select_rate_maps(trace, env_idx, freq_idx, max_cells=40)

        n_cells = selection.cell_indices.size
        if n_cells == 0:
            fig.suptitle("No cells available for rate maps")
            return fig

        n_cols = min(8, n_cells)
        n_rows = int(ceil(n_cells / n_cols))
        grid = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.8] + [1.0] * n_rows)

        occupancy_values = selection.occupancy.astype(float)
        occupancy_values[occupancy_values == 0] = np.nan
        occupancy_max = np.nanmax(occupancy_values) if np.isfinite(occupancy_values).any() else 1.0

        occ_ax = fig.add_subplot(grid[0, :])
        plot_map(world, occupancy_values, ax=occ_ax, min_val=0.0, max_val=occupancy_max, shape="square")
        coverage = np.isfinite(occupancy_values).sum() / max(len(occupancy_values), 1)
        occ_ax.set_title(f"Occupancy (coverage={coverage:.0%})")

        vmin, vmax = _robust_min_max(selection.rate_maps)

        for idx, cell_id in enumerate(selection.cell_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = fig.add_subplot(grid[row + 1, col])
            plot_map(world, selection.rate_maps[:, idx], ax=ax, min_val=vmin, max_val=vmax, shape="square")
            ax.set_title(f"Cell {cell_id}", fontsize=9)

        title = f"Place-cell Rate Maps (freq={freq_idx}, env={env_idx})"
        if ctx.split_name:
            title += f" - {ctx.split_name}"
        if ctx.global_step is not None:
            title += f" @ step {ctx.global_step}"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def plot_frequencies(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render multi-frequency place-cell rate maps for the same cell set."""
    style_ctx = (ctx.style or _get_default_style()).apply_context() if getattr(ctx, "style", None) else _noop_context()

    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        base_freq = validate_freq_idx(trace, "output/inference/p_inf", int(ctx.freq_idx))

        world = get_world(trace, env_idx)
        n_freq = get_n_freq(trace, "output/inference/p_inf")

        n_cells_per_freq = []
        for freq_idx in range(n_freq):
            activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
            activity_env = activity_steps[:, env_idx, :]
            n_cells_per_freq.append(activity_env.shape[1] if activity_env.ndim == 2 else 0)

        min_cells = min(n_cells_per_freq) if n_cells_per_freq else 0
        if min_cells == 0:
            fig.suptitle("No cells available for rate maps")
            return fig

        rate_map, occupancy = _aggregate_rate_maps(trace, env_idx, base_freq)
        activity_steps = get_multiscale(trace, "output/inference/p_inf", base_freq)
        activity_env = activity_steps[:, env_idx, :]
        if activity_env.ndim == 1:
            activity_env = activity_env[:, None]
        if activity_env.shape[1] > min_cells:
            activity_env = activity_env[:, :min_cells]

        k = min(12, min_cells)
        cell_indices = _select_top_k(activity_env, k)
        selection = _RateMapSelection(cell_indices, rate_map[:, cell_indices], occupancy)
        n_cells = selection.cell_indices.size
        if n_cells == 0:
            fig.suptitle("No cells available for rate maps")
            return fig

        grid = fig.add_gridspec(n_freq, n_cells)
        all_maps = []

        rate_maps_by_freq = []
        for freq_idx in range(n_freq):
            rate_map, _ = _aggregate_rate_maps(trace, env_idx, freq_idx)
            selected = rate_map[:, selection.cell_indices]
            rate_maps_by_freq.append(selected)
            all_maps.append(selected)

        vmin, vmax = _robust_min_max(np.concatenate(all_maps, axis=1))

        for freq_idx, freq_maps in enumerate(rate_maps_by_freq):
            for col, cell_id in enumerate(selection.cell_indices):
                ax = fig.add_subplot(grid[freq_idx, col])
                plot_map(world, freq_maps[:, col], ax=ax, min_val=vmin, max_val=vmax, shape="square")
                if freq_idx == 0:
                    ax.set_title(f"Cell {cell_id}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"Freq {freq_idx}", fontsize=9)

        title = f"Place-cell Rate Maps Across Frequencies (env={env_idx})"
        if ctx.split_name:
            title += f" - {ctx.split_name}"
        if ctx.global_step is not None:
            title += f" @ step {ctx.global_step}"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def plot_pathways(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render place-cell rate maps for multiple pathways in a frequency."""
    style_ctx = (ctx.style or _get_default_style()).apply_context() if getattr(ctx, "style", None) else _noop_context()

    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)

        if get_length(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/inference/p_inf", int(ctx.freq_idx))

        world = get_world(trace, env_idx)

        selection = _select_rate_maps(trace, env_idx, freq_idx, max_cells=12)
        n_cells = selection.cell_indices.size
        if n_cells == 0:
            fig.suptitle("No cells available for rate maps")
            return fig

        pathways = [
            ("p_inf", get_multiscale(trace, "output/inference/p_inf", freq_idx)),
            ("p_gen_gi", get_multiscale(trace, "output/generative/p_gen_gi", freq_idx)),
            ("p_gen_gg", get_multiscale(trace, "output/generative/p_gen_gg", freq_idx)),
        ]

        grid = fig.add_gridspec(len(pathways), n_cells)
        all_maps = []
        pathway_maps = []

        location_ids = get_location_ids_for_env(trace, env_idx)
        n_locations = len(world.locations)

        for _, activity_steps in pathways:
            activity_env = activity_steps[:, env_idx, :]
            rate_map, _ = _aggregate_rate_maps_from_steps(
                activity_env,
                location_ids,
                n_locations,
            )
            selected = rate_map[:, selection.cell_indices]
            pathway_maps.append(selected)
            all_maps.append(selected)

        vmin, vmax = _robust_min_max(np.concatenate(all_maps, axis=1))

        for row, (pathway_name, freq_maps) in enumerate(zip([p[0] for p in pathways], pathway_maps)):
            for col, cell_id in enumerate(selection.cell_indices):
                ax = fig.add_subplot(grid[row, col])
                plot_map(world, freq_maps[:, col], ax=ax, min_val=vmin, max_val=vmax, shape="square")
                if row == 0:
                    ax.set_title(f"Cell {cell_id}", fontsize=9)
                if col == 0:
                    ax.set_ylabel(pathway_name, fontsize=9)

        title = "Place-cell Rate Maps Across Pathways " f"(freq={freq_idx}, env={env_idx})"
        if ctx.split_name:
            title += f" - {ctx.split_name}"
        if ctx.global_step is not None:
            title += f" @ step {ctx.global_step}"
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def _select_rate_maps(trace: TraceTree, env_idx: int, freq_idx: int, *, max_cells: int) -> _RateMapSelection:
    rate_map, occupancy = _aggregate_rate_maps(trace, env_idx, freq_idx)
    activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
    activity_env = activity_steps[:, env_idx, :]

    n_cells = activity_env.shape[1] if activity_env.ndim == 2 else 0
    if n_cells == 0:
        return _RateMapSelection(np.array([], dtype=int), np.empty((rate_map.shape[0], 0)), occupancy)

    k = min(max_cells, n_cells)
    cell_indices = _select_top_k(activity_env, k)
    return _RateMapSelection(cell_indices, rate_map[:, cell_indices], occupancy)


def _aggregate_rate_maps(trace: TraceTree, env_idx: int, freq_idx: int) -> tuple[np.ndarray, np.ndarray]:
    activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
    location_ids = get_location_ids_for_env(trace, env_idx)
    n_locations = len(get_world(trace, env_idx).locations)
    return _aggregate_rate_maps_from_steps(activity_steps[:, env_idx, :], location_ids, n_locations)


def _aggregate_rate_maps_from_steps(
    activity_env: np.ndarray,
    location_ids: list[int],
    n_locations: int,
) -> tuple[np.ndarray, np.ndarray]:
    if activity_env.ndim == 1:
        activity_env = activity_env[:, None]

    rate_map = np.full((n_locations, activity_env.shape[1]), np.nan, dtype=np.float32)
    occupancy = np.zeros(n_locations, dtype=int)
    loc_ids = np.asarray(location_ids, dtype=int)

    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            occupancy[loc_id] = int(mask.sum())
            rate_map[loc_id] = activity_env[mask].mean(axis=0)

    return rate_map, occupancy


def _select_top_k(activity_env: np.ndarray, k: int) -> np.ndarray:
    if activity_env.size == 0:
        return np.array([], dtype=int)
    variance = np.var(activity_env, axis=0)
    return np.argsort(-variance)[:k]


def _robust_min_max(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(flat, [lower, upper])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _get_default_style():
    try:
        from torch_tem.figures.style import StyleConfig

        return StyleConfig()
    except ImportError:
        return None


class _noop_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False
