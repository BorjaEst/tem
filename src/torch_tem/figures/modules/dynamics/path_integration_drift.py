"""Path integration drift figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_multiscale, get_n_freq, validate_env_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot path integration drift over time for all frequencies.

    Drift is computed as the L2 distance between g_inf and g_gen per step.

    Args:
        trace: RolloutTrace containing inference and generative codes.
        ctx: Figure context with env selection.

    Returns:
        Matplotlib Figure containing drift curves.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        if get_length(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        n_freq = get_n_freq(trace, "output/inference/g_inf")

        if n_freq == 0:
            ax.set_title("No frequency modules available")
            ax.axis("off")
            return fig

        for freq_idx in range(n_freq):
            g_inf = get_multiscale(trace, "output/inference/g_inf", freq_idx)
            g_gen = get_multiscale(trace, "output/generative/g_gen", freq_idx)

            n_steps = min(g_inf.shape[0], g_gen.shape[0])
            if n_steps == 0:
                continue

            g_inf_env = g_inf[:n_steps, env_idx, :]
            g_gen_env = g_gen[:n_steps, env_idx, :]
            drift = np.linalg.norm(g_inf_env - g_gen_env, axis=1)
            ax.plot(drift, label=f"freq {freq_idx}")

        title = f"Path Integration Drift (env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("L2 Drift")
        ax.legend(loc="upper right", fontsize=8, ncol=2)
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
