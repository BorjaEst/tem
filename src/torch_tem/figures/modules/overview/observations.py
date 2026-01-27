"""TEM Overview figure module.

Generates a multi-panel overview figure showing key model outputs over time.
This is a placeholder/minimal implementation demonstrating the registry pattern.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.trace_access import get_action_ids, get_length, get_multiscale, validate_env_idx, validate_freq_idx


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Generate TEM overview figure from a model rollout trace.

    Creates a multi-panel figure showing:
    - Inference abstract codes (g_inf) over time
    - Generative abstract codes (g_gen) over time
    - Actions taken over time

    Args:
        trace: TraceTree with model outputs and world steps.
        ctx: Figure context (env_idx, freq_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with overview panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=ctx.figsize)

    if get_length(trace) == 0:
        axes[0].set_title("No trace data (empty rollout)")
        return fig

    env_idx = int(ctx.env_idx)
    validate_env_idx(trace, env_idx)

    freq_idx = int(ctx.freq_idx)

    # Panel 1: g_inf over time (selected frequency scale, all features)
    validate_freq_idx(trace, "output/inference/g_inf", freq_idx)
    g_inf_time = get_multiscale(trace, "output/inference/g_inf", freq_idx)
    g_inf_env = g_inf_time[:, env_idx, :].T
    axes[0].imshow(g_inf_env, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"Inference Abstract Codes (g_inf) - Freq {freq_idx}")
    axes[0].set_ylabel("Feature")
    axes[0].set_xlabel("Time Step")

    # Panel 2: g_gen over time (selected frequency scale, all features)
    validate_freq_idx(trace, "output/generative/g_gen", freq_idx)
    g_gen_time = get_multiscale(trace, "output/generative/g_gen", freq_idx)
    g_gen_env = g_gen_time[:, env_idx, :].T
    axes[1].imshow(g_gen_env, aspect="auto", cmap="plasma", interpolation="nearest")
    axes[1].set_title(f"Generative Abstract Codes (g_gen) - Freq {freq_idx}")
    axes[1].set_ylabel("Feature")
    axes[1].set_xlabel("Time Step")

    # Panel 3: Actions over time (NumPy array with -1 sentinel)
    actions_env = get_action_ids(trace)[:, env_idx]
    axes[2].plot(actions_env, marker="o", linestyle="-", markersize=4)
    axes[2].set_title("Actions Over Time")
    axes[2].set_ylabel("Action ID")
    axes[2].set_xlabel("Time Step")
    axes[2].axhline(y=-1, color="red", linestyle="--", alpha=0.5, label="Episode Boundary")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
