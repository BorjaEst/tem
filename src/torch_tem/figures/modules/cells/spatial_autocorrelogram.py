"""Spatial autocorrelogram figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_location_ids_for_env, get_multiscale, get_world, validate_env_idx, validate_freq_idx
from torch_tem.figures.utils.spatial import aggregate_rate_map, radial_autocorr


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
        rate_map, _ = aggregate_rate_map(activity_env, location_ids, len(world.locations))

        if rate_map.size == 0:
            ax.set_title("No rate map values available")
            ax.axis("off")
            return fig

        curves = []
        centers = None
        for cell_idx in range(rate_map.shape[1]):
            centers, curve = radial_autocorr(rate_map[:, cell_idx], world, n_bins=12)
            if curve.size:
                curves.append((cell_idx, curve))

        if not curves or centers is None or centers.size == 0:
            ax.set_title("No location coordinates available")
            ax.axis("off")
            return fig

        for cell_idx, curve in curves[:6]:
            ax.plot(centers, curve, color="#9ecae1", alpha=0.4)
        mean_curve = np.nanmean([curve for _, curve in curves], axis=0)
        ax.plot(centers, mean_curve, color="#3182bd", linewidth=2)

        title = f"Spatial Autocorrelogram (freq={freq_idx}, env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Distance")
        ax.set_ylabel("Autocorrelation")
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle="--")
        fig.tight_layout()
        return fig


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
