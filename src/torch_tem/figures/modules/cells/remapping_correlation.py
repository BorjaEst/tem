"""Remapping correlation figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: RolloutTrace, ctx: FigureContext) -> Figure:
    """Render remapping correlations across environments.

    Args:
        trace: RolloutTrace containing inference codes for multiple envs.
        ctx: Figure context with frequency selection.

    Returns:
        Matplotlib Figure with environment correlation matrix.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        if trace.batch_size == 0 or len(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        freq_idx = _validate_freq_idx(trace, int(ctx.freq_idx))
        activity_steps = _get_multiscale_steps(trace.output.inference.p_inf, freq_idx)
        n_env = trace.batch_size

        if n_env < 2:
            ax.set_title("Need multiple environments for remapping")
            ax.axis("off")
            return fig

        rate_maps, n_locations = _collect_rate_maps(trace, activity_steps)
        selection = _select_cells(activity_steps[:, 0, :].numpy(), max_cells=12)

        corr_matrix = np.full((n_env, n_env), np.nan, dtype=float)
        for i in range(n_env):
            for j in range(n_env):
                if n_locations[i] != n_locations[j]:
                    continue
                corr_matrix[i, j] = _mean_cell_correlation(
                    rate_maps[i][:, selection],
                    rate_maps[j][:, selection],
                )

        im = ax.imshow(corr_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xlabel("Environment")
        ax.set_ylabel("Environment")
        ax.set_xticks(range(n_env))
        ax.set_yticks(range(n_env))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = f"Remapping Correlation (freq={freq_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig


def _collect_rate_maps(trace: RolloutTrace, activity_steps: torch.Tensor):
    """Collect per-environment rate maps and location counts."""
    rate_maps = []
    n_locations = []
    for env_idx in range(trace.batch_size):
        world = _get_world(trace, env_idx)
        location_ids = _get_location_ids(trace, env_idx)
        activity_env = activity_steps[:, env_idx, :].numpy()
        rate_map = _aggregate_rate_maps(activity_env, location_ids, len(world.locations))
        rate_maps.append(rate_map)
        n_locations.append(len(world.locations))
    return rate_maps, n_locations


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


def _select_cells(activity_env: np.ndarray, max_cells: int) -> np.ndarray:
    """Select top-variance cells for stable comparison."""
    if activity_env.size == 0:
        return np.array([], dtype=int)
    variance = np.var(activity_env, axis=0)
    k = min(max_cells, activity_env.shape[1])
    return np.argsort(-variance)[:k]


def _mean_cell_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Compute mean correlation across cells."""
    if left.size == 0 or right.size == 0:
        return float("nan")
    n_cells = min(left.shape[1], right.shape[1])
    corrs = []
    for cell_idx in range(n_cells):
        lvals = left[:, cell_idx]
        rvals = right[:, cell_idx]
        mask = np.isfinite(lvals) & np.isfinite(rvals)
        if mask.sum() < 2:
            continue
        corrs.append(np.corrcoef(lvals[mask], rvals[mask])[0, 1])
    return float(np.nanmean(corrs)) if corrs else float("nan")


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
