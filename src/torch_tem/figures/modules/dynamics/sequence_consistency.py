"""Sequence consistency figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_multiscale, get_n_freq, validate_env_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot consistency between generated and inferred sequences.

    Consistency is computed as cosine similarity between g_gen at time t
    and g_inf at time t+1 for each frequency module.

    Args:
        trace: RolloutTrace containing inference and generative codes.
        ctx: Figure context with env selection.

    Returns:
        Matplotlib Figure containing consistency curves.
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
            g_gen = get_multiscale(trace, "output/generative/g_gen", freq_idx)
            g_inf = get_multiscale(trace, "output/inference/g_inf", freq_idx)

            n_steps = min(g_gen.shape[0], g_inf.shape[0]) - 1
            if n_steps <= 0:
                continue

            gen_env = g_gen[:n_steps, env_idx, :]
            inf_env = g_inf[1 : n_steps + 1, env_idx, :]
            sim = _cosine_similarity(gen_env, inf_env)
            ax.plot(sim, label=f"freq {freq_idx}")

        title = f"Sequence Consistency (env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Time")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        return fig


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute cosine similarity row-wise."""
    left_norm = np.linalg.norm(left, axis=1, keepdims=True)
    right_norm = np.linalg.norm(right, axis=1, keepdims=True)
    denom = np.where(left_norm * right_norm == 0, 1.0, left_norm * right_norm)
    return (left * right).sum(axis=1) / denom.squeeze(-1)


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
