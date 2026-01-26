"""Sequence consistency figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: RolloutTrace, ctx: FigureContext) -> Figure:
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

        if trace.batch_size == 0 or len(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = _validate_env_idx(trace, int(ctx.env_idx))
        n_freq = _get_n_freq(trace)
        if n_freq == 0:
            ax.set_title("No frequency modules available")
            ax.axis("off")
            return fig

        for freq_idx in range(n_freq):
            g_gen = _get_multiscale_steps(trace.output.generative.g_gen, freq_idx)
            g_inf = _get_multiscale_steps(trace.output.inference.g_inf, freq_idx)

            n_steps = min(g_gen.shape[0], g_inf.shape[0]) - 1
            if n_steps <= 0:
                continue

            gen_env = g_gen[:n_steps, env_idx, :].numpy()
            inf_env = g_inf[1 : n_steps + 1, env_idx, :].numpy()
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


def _get_multiscale_steps(steps, freq_idx: int) -> torch.Tensor:
    """Stack multiscale steps for a single frequency."""
    if not steps:
        raise ValueError("RolloutTrace has no steps for this pathway")
    if not (0 <= freq_idx < len(steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(steps[0])})")
    return torch.stack([step[freq_idx].detach().cpu() for step in steps], dim=0)


def _get_n_freq(trace: RolloutTrace) -> int:
    """Return the number of frequency modules."""
    steps = trace.output.inference.g_inf
    if not steps:
        return 0
    return len(steps[0])


def _validate_env_idx(trace: RolloutTrace, env_idx: int) -> int:
    """Validate the selected environment index."""
    if not (0 <= env_idx < trace.batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {trace.batch_size})")
    return env_idx


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
