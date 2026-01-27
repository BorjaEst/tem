"""Frequency similarity figure module."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_length, get_multiscale, get_n_freq, validate_env_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Render representational similarity across frequency modules.

    Args:
        trace: RolloutTrace containing inference codes.
        ctx: Figure context with env selection.

    Returns:
        Matplotlib Figure with a frequency similarity heatmap.
    """
    style_ctx = _style_context(ctx)
    with style_ctx:
        fig, ax = plt.subplots(figsize=ctx.figsize)

        if get_length(trace) == 0:
            ax.set_title("No trace data (empty rollout)")
            ax.axis("off")
            return fig

        env_idx = validate_env_idx(trace, int(ctx.env_idx))
        n_freq = get_n_freq(trace, "output/inference/p_inf")
        if n_freq <= 1:
            ax.set_title("Not enough frequency modules for comparison")
            ax.axis("off")
            return fig

        vectors = []
        for freq_idx in range(n_freq):
            steps = get_multiscale(trace, "output/inference/p_inf", freq_idx)
            activity = steps[:, env_idx, :]
            if activity.shape[0] < 2:
                vectors.append(np.array([]))
                continue
            sim_vector = _time_similarity_vector(activity)
            vectors.append(sim_vector)

        sim_matrix = np.full((n_freq, n_freq), np.nan, dtype=float)
        for i in range(n_freq):
            for j in range(n_freq):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    sim_matrix[i, j] = _corrcoef_safe(vectors[i], vectors[j])

        im = ax.imshow(sim_matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Frequency")
        ax.set_xticks(range(n_freq))
        ax.set_yticks(range(n_freq))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = f"Frequency Similarity (env={env_idx})"
        title = _append_context(title, ctx)
        ax.set_title(title, fontsize=12)
        fig.tight_layout()
        return fig


def _time_similarity_vector(activity: np.ndarray) -> np.ndarray:
    """Compute upper-triangle cosine similarity of time steps.

    Args:
        activity: Per-step activity (T, C).

    Returns:
        Flattened upper-triangle similarity vector.
    """
    activity = activity - activity.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(activity, axis=1, keepdims=True)
    normed = activity / np.where(norms == 0, 1.0, norms)
    sim = normed @ normed.T
    return sim[np.triu_indices(sim.shape[0], k=1)]


def _corrcoef_safe(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation for equal-length vectors."""
    if left.size == 0 or right.size == 0:
        return float("nan")
    if left.size != right.size:
        min_size = min(left.size, right.size)
        left = left[:min_size]
        right = right[:min_size]
    return float(np.corrcoef(left, right)[0, 1])


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
