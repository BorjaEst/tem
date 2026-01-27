"""Uncertainty calibration figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_multiscale, validate_env_idx, validate_freq_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot calibration of uncertainty proxies versus error.

    Uses MEC uncertainty (if available) and path-integration error between
    g_inf and g_gen as a proxy error signal.

    Args:
        trace: RolloutTrace containing state uncertainty and codes.
        ctx: Figure context with env and frequency selection.

    Returns:
        Matplotlib Figure with calibration curve.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        if get_length(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        freq_idx = validate_freq_idx(trace, "output/inference/g_inf", int(ctx.freq_idx))

        try:
            uncert = get_multiscale(trace, "state/mec/location/uncertainty", freq_idx)
        except (ValueError, IndexError):
            ax.set_title("No uncertainty data available")
            ax.axis("off")
            return fig

        g_inf = get_multiscale(trace, "output/inference/g_inf", freq_idx)
        g_gen = get_multiscale(trace, "output/generative/g_gen", freq_idx)

        n_steps = min(uncert.shape[0], g_inf.shape[0], g_gen.shape[0])
        if n_steps == 0:
            ax.set_title("No calibration data available")
            ax.axis("off")
            return fig

        uncert_env = uncert[:n_steps, env_idx, :]
        error_env = np.linalg.norm(g_inf[:n_steps, env_idx, :] - g_gen[:n_steps, env_idx, :], axis=1)
        uncertainty_value = np.nanmean(uncert_env, axis=1)

        bins = np.quantile(uncertainty_value, np.linspace(0.0, 1.0, 11))
        centers = 0.5 * (bins[:-1] + bins[1:])
        mean_error, counts = _bin_errors(uncertainty_value, error_env, bins)

        ax.plot(centers, mean_error, marker="o", color="#4c72b0")
        ax.set_xlabel("Uncertainty (binned)")
        ax.set_ylabel("Mean Error")
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=np.diff(bins), alpha=0.2, color="#4c72b0")
        ax2.set_ylabel("Count")

        title = f"Uncertainty Calibration (freq={freq_idx}, env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig


def _bin_errors(uncertainty: np.ndarray, errors: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate errors within uncertainty bins."""
    mean_error = np.full(len(bins) - 1, np.nan, dtype=float)
    counts = np.zeros(len(bins) - 1, dtype=int)
    for idx in range(len(bins) - 1):
        mask = (uncertainty >= bins[idx]) & (uncertainty <= bins[idx + 1])
        if mask.any():
            counts[idx] = int(mask.sum())
            mean_error[idx] = float(np.mean(errors[mask]))
    return mean_error, counts


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
