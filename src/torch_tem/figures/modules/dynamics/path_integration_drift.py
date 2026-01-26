"""Path integration drift figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: RolloutTrace, ctx: FigureContext) -> Figure:
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
            g_inf = _get_multiscale_steps(trace.output.inference.g_inf, freq_idx)
            g_gen = _get_multiscale_steps(trace.output.generative.g_gen, freq_idx)

            n_steps = min(g_inf.shape[0], g_gen.shape[0])
            if n_steps == 0:
                continue

            g_inf_env = g_inf[:n_steps, env_idx, :].numpy()
            g_gen_env = g_gen[:n_steps, env_idx, :].numpy()
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
