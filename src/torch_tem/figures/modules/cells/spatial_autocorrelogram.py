"""Spatial autocorrelogram figure module."""

from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render a radial spatial autocorrelogram for place-like activity.

    Args:
        trace: TraceTree containing inference codes.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with autocorrelation curves.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        if get_length(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/inference/p_inf", int(ctx.freq_idx))

        world = get_world(trace, env_idx)
        location_ids = get_location_ids_for_env(trace, env_idx)
        activity_steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
        activity_env = activity_steps[:, env_idx, :]
        rate_map = _aggregate_rate_maps(activity_env, location_ids, len(world.locations))

        coords = _location_coords(world)
        if coords.size == 0:
            ax.set_title("No location coordinates available")
            ax.axis("off")
            return fig

        distances = _pairwise_distances(coords)
        bins, centers = _distance_bins(distances, n_bins=12)

        summary = _autocorr_summary(rate_map, distances, bins)
        for cell_idx, curve in summary["per_cell"][:6]:
            ax.plot(centers, curve, color="#9ecae1", alpha=0.4)
        ax.plot(centers, summary["mean"], color="#3182bd", linewidth=2)

        title = f"Spatial Autocorrelogram (freq={freq_idx}, env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Autocorrelation")
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        return fig


def _aggregate_rate_maps(
    activity_env: np.ndarray,
    location_ids: list[int],
    n_locations: int,
) -> np.ndarray:
    """Aggregate per-step activity into per-location means."""
    if activity_env.ndim == 1:
        activity_env = activity_env[:, None]
    rate_map = np.full((n_locations, activity_env.shape[1]), np.nan, dtype=float)
    loc_ids = np.asarray(location_ids, dtype=int)
    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            rate_map[loc_id] = activity_env[mask].mean(axis=0)
    return rate_map


def _location_coords(world) -> np.ndarray:
    """Extract (x, y) coordinates for all locations."""
    coords = [[float(loc["o"]), float(loc["y"])] for loc in world.locations]
    return np.asarray(coords, dtype=float)


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _distance_bins(distances: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute distance bin edges and centers."""
    max_dist = float(np.nanmax(distances)) if distances.size else 0.0
    bins = np.linspace(0.0, max_dist, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return bins, centers


def _autocorr_summary(
    rate_map: np.ndarray,
    distances: np.ndarray,
    bins: np.ndarray,
) -> dict:
    """Compute per-cell and mean autocorrelogram curves."""
    n_cells = rate_map.shape[1]
    per_cell = []
    for cell_idx in range(n_cells):
        values = rate_map[:, cell_idx]
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        z_vals = (values[finite] - np.mean(values[finite])) / np.std(values[finite])
        z_full = np.full(values.shape[0], np.nan)
        z_full[finite] = z_vals

        curve = _bin_autocorr(z_full, distances, bins)
        per_cell.append((cell_idx, curve))

    if not per_cell:
        mean_curve = np.zeros(len(bins) - 1)
    else:
        mean_curve = np.nanmean([curve for _, curve in per_cell], axis=0)
    return {"per_cell": per_cell, "mean": mean_curve}


def _bin_autocorr(values: np.ndarray, distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Bin autocorrelation products by distance."""
    n_bins = len(bins) - 1
    out = np.full(n_bins, np.nan, dtype=float)
    upper = np.triu_indices(values.shape[0], k=1)
    dists = distances[upper]
    vals_i = values[upper[0]]
    vals_j = values[upper[1]]
    valid = np.isfinite(vals_i) & np.isfinite(vals_j)
    if not valid.any():
        return out
    dists = dists[valid]
    products = vals_i[valid] * vals_j[valid]
    for bin_idx in range(n_bins):
        mask = (dists >= bins[bin_idx]) & (dists < bins[bin_idx + 1])
        if mask.any():
            out[bin_idx] = float(np.mean(products[mask]))
    return out


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
