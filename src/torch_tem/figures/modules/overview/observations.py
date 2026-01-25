"""TEM Overview figure module.

Generates a multi-panel overview figure showing key model outputs over time.
This is a placeholder/minimal implementation demonstrating the registry pattern.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TEMStateTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: TEMStateTrace, ctx: FigureContext) -> Figure:
    """Generate TEM overview figure from a model rollout trace.

    Creates a multi-panel figure showing:
    - Inference abstract codes (g_inf) over time
    - Generative abstract codes (g_gen) over time
    - Actions taken over time

    Args:
        trace: TEMStateTrace with model outputs (CPU/NumPy).
        ctx: Figure context (env_idx, freq_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with overview panels.
    """
    fig, axes = plt.subplots(3, 1, figsize=ctx.figsize)

    if len(trace.actions) == 0:
        axes[0].set_title("No trace data (empty rollout)")
        return fig

    # Infer batch size from the first action list
    batch_size = len(trace.actions[0])
    env_idx = int(ctx.env_idx)
    if not (0 <= env_idx < batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {batch_size})")

    freq_idx = int(ctx.freq_idx)

    # Panel 1: g_inf over time (selected frequency scale, all features)
    g_inf_steps = trace.output.inference.g_inf  # list[T] of MultiScaleCode
    if len(g_inf_steps) == 0:
        raise ValueError("TEMStateTrace has no inference steps")
    if not (0 <= freq_idx < len(g_inf_steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(g_inf_steps[0])})")

    g_inf_time = torch.stack([step[freq_idx].detach().cpu() for step in g_inf_steps], dim=0)  # (T, B, C)
    g_inf_env = g_inf_time[:, env_idx, :].T.numpy()  # (C, T)
    axes[0].imshow(g_inf_env, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0].set_title(f"Inference Abstract Codes (g_inf) - Freq {freq_idx}")
    axes[0].set_ylabel("Feature")
    axes[0].set_xlabel("Time Step")

    # Panel 2: g_gen over time (selected frequency scale, all features)
    g_gen_steps = trace.output.generative.g_gen
    if len(g_gen_steps) == 0:
        raise ValueError("TEMStateTrace has no generative steps")
    if not (0 <= freq_idx < len(g_gen_steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(g_gen_steps[0])})")

    g_gen_time = torch.stack([step[freq_idx].detach().cpu() for step in g_gen_steps], dim=0)  # (T, B, C)
    g_gen_env = g_gen_time[:, env_idx, :].T.numpy()  # (C, T)
    axes[1].imshow(g_gen_env, aspect="auto", cmap="plasma", interpolation="nearest")
    axes[1].set_title(f"Generative Abstract Codes (g_gen) - Freq {freq_idx}")
    axes[1].set_ylabel("Feature")
    axes[1].set_xlabel("Time Step")

    # Panel 3: Actions over time (NumPy array with -1 sentinel)
    actions_time = np.array([[(-1 if a is None else int(a)) for a in step_actions] for step_actions in trace.actions], dtype=int)  # (T, B)
    actions_env = actions_time[:, env_idx]
    axes[2].plot(actions_env, marker="o", linestyle="-", markersize=4)
    axes[2].set_title("Actions Over Time")
    axes[2].set_ylabel("Action ID")
    axes[2].set_xlabel("Time Step")
    axes[2].axhline(y=-1, color="red", linestyle="--", alpha=0.5, label="Episode Boundary")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig
