"""TEM Overview Rate Maps figure module.

Generates a multi-panel overview figure combining model time-series heatmaps
with spatial rate maps (square-grid) showing per-location activity aggregations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import RolloutTrace
from torch_tem.figures.primitives import plot_map
from torch_tem.figures.registry import FigureContext


def plot(trace: RolloutTrace, ctx: FigureContext) -> Figure:
    """Generate TEM overview rate-maps figure from a combined rollout trace.

    Creates a 2x2 layout:
    - Top-left: g_inf time heatmap (selected env, selected freq)
    - Top-right: g_gen time heatmap
    - Bottom-left: Spatial rate map for a single g_inf feature
    - Bottom-right: Spatial rate map for a single g_gen feature

    Args:
        trace: RolloutTrace with model outputs + world geometry + location IDs.
        ctx: Figure context (env_idx, freq_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with overview panels.
    """
    fig, axes = plt.subplots(2, 3, figsize=ctx.figsize)

    # Validate trace
    if trace.batch_size == 0 or len(trace) == 0:
        axes[0, 0].set_title("No trace data (empty rollout)")
        return fig

    env_idx = int(ctx.env_idx)
    if not (0 <= env_idx < trace.batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {trace.batch_size})")

    freq_idx = int(ctx.freq_idx)

    # Get selected environment's world and location IDs
    if not trace.world_step.environments:
        raise ValueError("RolloutTrace has no environments")
    world = trace.world_step.environments[env_idx]
    location_ids = trace.world_step.location_ids[env_idx]
    n_locations = len(world.locations)

    # Panel 1 (top-left): g_inf time heatmap
    g_inf_steps = trace.output.inference.g_inf
    if len(g_inf_steps) == 0:
        raise ValueError("RolloutTrace has no inference steps")
    if not (0 <= freq_idx < len(g_inf_steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(g_inf_steps[0])})")

    g_inf_time = torch.stack([step[freq_idx].detach().cpu() for step in g_inf_steps], dim=0)  # (T, B, C)
    g_inf_env = g_inf_time[:, env_idx, :].T.numpy()  # (C, T)
    axes[0, 0].imshow(g_inf_env, aspect="auto", cmap="viridis", interpolation="nearest")
    axes[0, 0].set_title(_append_context(f"g_inf Time (Freq {freq_idx})", ctx))
    axes[0, 0].set_ylabel("Feature")
    axes[0, 0].set_xlabel("Time")

    # Panel 2 (top-right): g_gen time heatmap
    g_gen_steps = trace.output.generative.g_gen
    if len(g_gen_steps) == 0:
        raise ValueError("RolloutTrace has no generative steps")
    if not (0 <= freq_idx < len(g_gen_steps[0])):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {len(g_gen_steps[0])})")

    g_gen_time = torch.stack([step[freq_idx].detach().cpu() for step in g_gen_steps], dim=0)  # (T, B, C)
    g_gen_env = g_gen_time[:, env_idx, :].T.numpy()  # (C, T)
    axes[0, 1].imshow(g_gen_env, aspect="auto", cmap="plasma", interpolation="nearest")
    axes[0, 1].set_title(_append_context(f"g_gen Time (Freq {freq_idx})", ctx))
    axes[0, 1].set_ylabel("Feature")
    axes[0, 1].set_xlabel("Time")

    # Panel 3 (bottom-left): Spatial rate map for first g_inf feature
    # Select first feature channel for simplicity
    feature_idx = 0
    g_inf_activity = g_inf_time[:, env_idx, feature_idx].numpy()  # (T,)
    g_inf_rate_map = compute_rate_map(g_inf_activity, location_ids, n_locations)

    plot_map(world, g_inf_rate_map, ax=axes[1, 0], shape="square", location_cm="viridis")
    axes[1, 0].set_title(_append_context(f"g_inf[{feature_idx}] Rate Map", ctx))

    # Panel 4 (bottom-right): Spatial rate map for first g_gen feature
    g_gen_activity = g_gen_time[:, env_idx, feature_idx].numpy()  # (T,)
    g_gen_rate_map = compute_rate_map(g_gen_activity, location_ids, n_locations)

    plot_map(world, g_gen_rate_map, ax=axes[1, 1], shape="square", location_cm="plasma")
    axes[1, 1].set_title(_append_context(f"g_gen[{feature_idx}] Rate Map", ctx))

    # Panel 5 (top-right): Location id vs time strip
    _plot_location_strip(location_ids, axes[0, 2])
    axes[0, 2].set_title(_append_context("Location ID vs Time", ctx))

    # Panel 6 (bottom-right): Time-colored trajectory companion
    _plot_time_colored_trajectory(world, location_ids, axes[1, 2])
    axes[1, 2].set_title(_append_context("Trajectory (time-colored)", ctx))

    plt.tight_layout()
    return fig


def compute_rate_map(activity: np.ndarray, location_ids: list[int], n_locations: int) -> np.ndarray:
    """Compute per-location rate map from per-step activity values.

    Aggregates activity values by visited location using mean pooling.
    Unvisited locations are represented as NaN.

    Args:
        activity: Per-step activity values (T, C) or (T,).
        location_ids: Per-step visited location IDs (T,).
        n_locations: Total number of locations in the environment.

    Returns:
        Per-location aggregated values (n_locations, C) or (n_locations,).
        Unvisited locations contain NaN.
    """
    loc_ids = np.array(location_ids)
    shape = (n_locations, activity.shape[1]) if activity.ndim > 1 else (n_locations,)
    rate_map = np.full(shape, np.nan, dtype=np.float32)

    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            rate_map[loc_id] = np.nanmean(activity[mask], axis=0)

    return rate_map


def _plot_location_strip(location_ids: list[int], ax: plt.Axes) -> None:
    if not location_ids:
        ax.text(0.5, 0.5, "No locations", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    strip = np.array(location_ids, dtype=float)[np.newaxis, :]
    ax.imshow(strip, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_ylabel("Location")
    ax.set_xlabel("Time")


def _plot_time_colored_trajectory(world, location_ids: list[int], ax: plt.Axes) -> None:
    if not location_ids:
        ax.text(0.5, 0.5, "No trajectory", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return

    values = np.full(len(world.locations), np.nan, dtype=float)
    plot_map(world, values, ax=ax, shape="square")

    coords = np.array([[world.locations[loc_id]["o"], world.locations[loc_id]["y"]] for loc_id in location_ids], dtype=float)
    if coords.shape[0] < 2:
        ax.scatter(coords[:, 0], coords[:, 1], s=10, color="black")
        return

    segments = np.stack([coords[:-1], coords[1:]], axis=1)
    colors = np.linspace(0, 1, segments.shape[0])

    lc = LineCollection(segments, cmap="viridis", array=colors, linewidths=1.5)
    ax.add_collection(lc)
    ax.scatter(coords[0, 0], coords[0, 1], s=20, color="black", zorder=3)
    ax.scatter(coords[-1, 0], coords[-1, 1], s=20, color="white", edgecolor="black", zorder=3)

    ax.set_aspect(1)
    ax.invert_yaxis()
    ax.axis("off")


def _append_context(title: str, ctx: FigureContext) -> str:
    if ctx.split_name:
        title += f" - {ctx.split_name}"
    if ctx.global_step is not None:
        title += f" @ step {ctx.global_step}"
    return title
