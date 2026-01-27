"""Place-field summary figure module."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render summary statistics for place fields.

    Args:
        trace: TraceTree containing inference codes.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with histograms for place-field metrics.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, axes = plt.subplots(1, 3, figsize=ctx.figsize)

        if get_length(trace) == 0:
            axes[0].set_title("No trace data (empty rollout)")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/inference/p_inf", int(ctx.freq_idx))

        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
        activity_env = activity_steps[:, env_idx, :]

        rate_map = _aggregate_rate_maps(activity_env, location_ids, len(world.locations))
        sparsity, peaks, field_sizes = _compute_metrics(rate_map)

        axes[0].hist(sparsity[np.isfinite(sparsity)], bins=20, color="#4c72b0")
        axes[0].set_title("Sparsity")
        axes[0].set_xlabel("$s$")
        axes[0].set_ylabel("Count")

        axes[1].hist(peaks[np.isfinite(peaks)], bins=20, color="#55a868")
        axes[1].set_title("Peak Rate")
        axes[1].set_xlabel("Peak")

        axes[2].hist(field_sizes[np.isfinite(field_sizes)], bins=20, color="#c44e52")
        axes[2].set_title("Field Size")
        axes[2].set_xlabel("Fraction")

        title = f"Place-Field Summary (freq={freq_idx}, env={env_idx})"
        title = _append_context(title, ctx)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def _aggregate_rate_maps(activity_env: np.ndarray, location_ids: list[int], n_locations: int) -> np.ndarray:
    """Aggregate per-step activity into per-location rate maps.

    Args:
        activity_env: Per-step activity (T, C).
        location_ids: Per-step location ids (T,).
        n_locations: Number of locations.

    Returns:
        Rate map array with NaNs for unvisited locations (n_locations, C).
    """
    if activity_env.ndim == 1:
        activity_env = activity_env[:, None]

    rate_map = np.full((n_locations, activity_env.shape[1]), np.nan, dtype=float)
    loc_ids = np.asarray(location_ids, dtype=int)
    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            rate_map[loc_id] = activity_env[mask].mean(axis=0)
    return rate_map


def _compute_metrics(rate_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute sparsity, peak, and field size metrics per cell."""
    n_cells = rate_map.shape[1]
    sparsity = np.full(n_cells, np.nan, dtype=float)
    peaks = np.full(n_cells, np.nan, dtype=float)
    field_sizes = np.full(n_cells, np.nan, dtype=float)

    for cell_idx in range(n_cells):
        values = rate_map[:, cell_idx]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        vals = values[finite]
        peak = float(np.nanmax(vals))
        mean_val = float(np.nanmean(vals))
        mean_sq = float(np.nanmean(vals**2))
        sparsity[cell_idx] = (mean_val**2 / mean_sq) if mean_sq > 0 else np.nan
        peaks[cell_idx] = peak
        if peak > 0:
            field_sizes[cell_idx] = float(np.mean(vals >= 0.2 * peak))
    return sparsity, peaks, field_sizes


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
