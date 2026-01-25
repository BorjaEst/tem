"""Walk statistics figure module.

Generates summary statistics and histograms for walk characteristics.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import WorldTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: WorldTrace, ctx: FigureContext) -> Figure:
    """Generate walk statistics figure.

    Creates a multi-panel figure showing:
    - Walk length distribution
    - Action frequency histogram
    - Location revisit counts
    - Shiny object hit statistics (if applicable)

    Args:
        trace: WorldTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with walk statistics panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=ctx.figsize)
    axes = axes.flatten()

    # Panel 1: Walk length distribution (per environment in batch)
    walk_lengths = [len(trace)] * trace.batch_size
    axes[0].hist(walk_lengths, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_title("Walk Length Distribution")
    axes[0].set_xlabel("Walk Length (steps)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Panel 2: Action frequency (flattened across all steps and environments)
    all_actions = _flatten_actions(trace)

    action_counts = Counter(all_actions)
    n_actions = _infer_action_count(trace.environments, action_counts)
    action_freqs = [action_counts.get(i, 0) for i in range(n_actions)]

    axes[1].bar(range(n_actions), action_freqs, edgecolor="black", alpha=0.7)
    axes[1].set_title("Action Frequency")
    axes[1].set_xlabel("Action ID")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.3, axis="y")

    # Panel 3: Location revisit statistics
    if trace.visited:
        visit_counts = [sum(visited) for visited in trace.visited]
        axes[2].hist(visit_counts, bins=20, edgecolor="black", alpha=0.7, color="green")
        axes[2].set_title("Locations Visited per Walk")
        axes[2].set_xlabel("Unique Locations Visited")
        axes[2].set_ylabel("Frequency")
        axes[2].grid(alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, "Visit tracking\nnot available", ha="center", va="center", fontsize=12)
        axes[2].axis("off")

    # Panel 4: Shiny object hits (if shiny environment)
    shiny_hits = _collect_shiny_hits(trace)

    if shiny_hits:
        axes[3].hist(shiny_hits, bins=20, edgecolor="black", alpha=0.7, color="gold")
        axes[3].set_title("Shiny Object Hits")
        axes[3].set_xlabel("Number of Hits")
        axes[3].set_ylabel("Frequency")
        axes[3].grid(alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, "No shiny objects\nin environment", ha="center", va="center", fontsize=12)
        axes[3].axis("off")

    # Add overall title
    fig.suptitle(f"Walk Statistics ({trace.batch_size} walks)", fontsize=16, y=0.995)
    plt.tight_layout()
    return fig


def _flatten_actions(agent_trace: Iterable[Any]) -> list[int]:
    """Flatten actions across time and batch, ignoring missing actions."""
    actions: list[int] = []
    for step in agent_trace:
        for action in step.action:
            if action is None:
                continue
            actions.append(int(action))
    return actions


def _infer_action_count(environments: list[Any], action_counts: Counter) -> int:
    """Infer action count from environments or observed actions."""
    env_actions = [getattr(env, "n_actions", 0) for env in environments]
    max_env_actions = max(env_actions) if env_actions else 0
    if action_counts:
        return max(max_env_actions, max(action_counts.keys()) + 1)
    return max(max_env_actions, 1)


def _collect_shiny_hits(trace: WorldTrace) -> list[int]:
    """Count shiny hits per environment over the trace duration."""
    if not trace.environments or len(trace) == 0:
        return []

    shiny_hits: list[int] = []
    for env_idx, env in enumerate(trace.environments):
        if env.shiny is None:
            continue
        hits = 0
        for step in trace:
            locations = step.locations
            if env_idx >= len(locations):
                continue
            loc = locations[env_idx]
            if isinstance(loc, dict) and loc.get("shiny", False):
                hits += 1
        shiny_hits.append(hits)
    return shiny_hits
