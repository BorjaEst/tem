"""Location decoding error map figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext


def plot(trace: RolloutTrace, ctx: FigureContext) -> Figure:
    """Render a per-location decoding error map with occupancy.

    Args:
        trace: RolloutTrace containing inference codes and world data.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with error and occupancy panels.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig = plt.figure(figsize=ctx.figsize)

        if trace.batch_size == 0 or len(trace) == 0:
            fig.suptitle("No trace data (empty rollout)")
            return fig

        env_idx = _validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = _validate_freq_idx(trace, int(ctx.freq_idx))

        world = _get_world(trace, env_idx)
        location_ids = _get_location_ids(trace, env_idx)
        coords = _location_coords(world, location_ids)

        activity_steps = _get_multiscale_steps(trace.output.inference.p_inf, freq_idx)
        activity_env = activity_steps[:, env_idx, :].numpy()

        if activity_env.size == 0 or coords.size == 0:
            fig.suptitle("No decoding data available")
            return fig

        n_steps = min(activity_env.shape[0], coords.shape[0])
        activity_env = activity_env[:n_steps]
        coords = coords[:n_steps]

        errors = _ridge_decode_error(activity_env, coords)
        error_map, occupancy = _aggregate_by_location(
            errors,
            location_ids[:n_steps],
            len(world.locations),
        )

        grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.9])
        ax_error = fig.add_subplot(grid[0, 0])
        ax_occ = fig.add_subplot(grid[0, 1])

        vmin, vmax = _robust_min_max(error_map)
        plot_map(
            world,
            error_map,
            ax=ax_error,
            min_val=vmin,
            max_val=vmax,
            shape="square",
        )
        ax_error.set_title("Decoding Error", fontsize=11)

        occ_values = occupancy.astype(float)
        occ_values[occ_values == 0] = np.nan
        occ_max = float(np.nanmax(occ_values)) if np.isfinite(occ_values).any() else 1.0
        plot_map(
            world,
            occ_values,
            ax=ax_occ,
            min_val=0.0,
            max_val=occ_max,
            shape="square",
        )
        coverage = np.isfinite(occ_values).sum() / max(len(occ_values), 1)
        ax_occ.set_title(f"Occupancy (coverage={coverage:.0%})", fontsize=11)

        title = f"Location Decoding Error (freq={freq_idx}, env={env_idx})"
        title = _append_context(title, ctx)
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        return fig


def _ridge_decode_error(activity: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Fit a ridge decoder and return per-step Euclidean errors.

    Args:
        activity: Per-step activity matrix (T, C).
        coords: Per-step target coordinates (T, 2).

    Returns:
        Per-step decoding errors (T,).
    """
    if activity.ndim != 2 or coords.ndim != 2:
        raise ValueError("Expected activity (T, C) and coords (T, 2)")
    if activity.shape[0] < 2 or activity.shape[1] == 0:
        return np.full(activity.shape[0], np.nan, dtype=float)

    x_mat = activity
    y_mat = coords
    bias = np.ones((x_mat.shape[0], 1), dtype=x_mat.dtype)
    x_aug = np.hstack([x_mat, bias])

    alpha = 1e-3
    identity = np.eye(x_aug.shape[1], dtype=x_aug.dtype)
    weights = np.linalg.solve(x_aug.T @ x_aug + alpha * identity, x_aug.T @ y_mat)
    predictions = x_aug @ weights
    return np.linalg.norm(predictions - y_mat, axis=1)


def _aggregate_by_location(
    errors: np.ndarray,
    location_ids: list[int],
    n_locations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-step errors into per-location means.

    Args:
        errors: Per-step errors (T,).
        location_ids: Per-step location ids (T,).
        n_locations: Number of locations in the environment.

    Returns:
        Tuple of (error_map, occupancy_counts).
    """
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
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [lower, upper])
    if vmin == vmax:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def _location_coords(world, location_ids: list[int]) -> np.ndarray:
    """Map location ids to coordinate targets."""
    coords = []
    for loc_id in location_ids:
        if 0 <= loc_id < len(world.locations):
            loc = world.locations[loc_id]
            coords.append([float(loc["o"]), float(loc["y"])])
    return np.asarray(coords, dtype=float)


def _get_multiscale_steps(steps, freq_idx: int) -> torch.Tensor:
    """Stack multiscale steps for a single frequency."""
    if not steps:
        raise ValueError("RolloutTrace has no inference steps")
    if not (0 <= freq_idx < len(steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(steps[0])})")
    return torch.stack([step[freq_idx].detach().cpu() for step in steps], dim=0)


def _get_world(trace: RolloutTrace, env_idx: int):
    """Get the World for the selected environment index."""
    if not trace.world_step.environments:
        raise ValueError("RolloutTrace has no environments")
    return trace.world_step.environments[env_idx]


def _get_location_ids(trace: RolloutTrace, env_idx: int) -> list[int]:
    """Get per-step location ids for the selected environment."""
    location_ids = trace.world_step.location_ids
    if not location_ids:
        raise ValueError("RolloutTrace has no location ids")
    return location_ids[env_idx]


def _validate_env_idx(trace: RolloutTrace, env_idx: int) -> int:
    """Validate the selected environment index."""
    if not (0 <= env_idx < trace.batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {trace.batch_size})")
    return env_idx


def _validate_freq_idx(trace: RolloutTrace, freq_idx: int) -> int:
    """Validate the selected frequency index."""
    steps = trace.output.inference.p_inf
    if not steps:
        raise ValueError("RolloutTrace has no inference steps")
    n_freq = len(steps[0])
    if not (0 <= freq_idx < n_freq):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {n_freq})")
    return freq_idx


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
