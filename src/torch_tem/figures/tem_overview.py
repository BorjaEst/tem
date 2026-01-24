"""TEM Overview figure module.

Generates a multi-panel overview figure showing key model outputs over time.
This is a placeholder/minimal implementation demonstrating the registry pattern.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import ModelTrace
from torch_tem.figures.core.registry import FigureContext


def plot(trace: ModelTrace, ctx: FigureContext) -> Figure:
    """Generate TEM overview figure from a model rollout trace.

    Creates a multi-panel figure showing:
    - Inference abstract codes (g_inf) over time
    - Generative abstract codes (g_gen) over time
    - Actions taken over time

    Args:
        trace: ModelTrace with model outputs (CPU/NumPy).
        ctx: Figure context (env_idx, freq_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with overview panels.
    """
    # Select single environment if batch trace
    if trace.batch_size > 1:
        trace = trace.select_env(ctx.env_idx)

    fig, axes = plt.subplots(3, 1, figsize=ctx.figsize)

    # Panel 1: g_inf over time (selected frequency scale, all features)
    # trace.g_inf is list of [batch, n_steps, n_features_f] per scale
    g_inf = trace.g_inf[ctx.freq_idx][0, :, :]  # [n_steps, n_features]
    axes[0].imshow(g_inf.T, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"Inference Abstract Codes (g_inf) - Freq {ctx.freq_idx}")
    axes[0].set_ylabel("Feature")
    axes[0].set_xlabel("Time Step")

    # Panel 2: g_gen over time (selected frequency scale, all features)
    g_gen = trace.g_gen[ctx.freq_idx][0, :, :]  # [n_steps, n_features]
    axes[1].imshow(g_gen.T, aspect="auto", cmap="plasma", interpolation="nearest")
    axes[1].set_title(f"Generative Abstract Codes (g_gen) - Freq {ctx.freq_idx}")
    axes[1].set_ylabel("Feature")
    axes[1].set_xlabel("Time Step")

    # Panel 3: Actions over time (NumPy array with -1 sentinel)
    actions = trace.actions[0, :]  # [n_steps] int array
    axes[2].plot(actions, marker="o", linestyle="-", markersize=4)
    axes[2].set_title("Actions Over Time")
    axes[2].set_ylabel("Action ID")
    axes[2].set_xlabel("Time Step")
    axes[2].axhline(y=-1, color="red", linestyle="--", alpha=0.5, label="Episode Boundary")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
