"""Walk statistics figure module.

Generates summary statistics and histograms for walk characteristics.
"""

from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import DataTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: DataTrace, ctx: FigureContext) -> Figure:
    """Generate walk statistics figure.

    Creates a multi-panel figure showing:
    - Walk length distribution
    - Action frequency histogram
    - Location revisit counts
    - Shiny object hit statistics (if applicable)

    Args:
        trace: DataTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with walk statistics panels.
    """
    fig, axes = plt.subplots(2, 2, figsize=ctx.figsize)
    axes = axes.flatten()

    # Panel 1: Walk length distribution (across all envs in batch)
    walk_lengths = [len(walk) for walk in trace.walks]
    axes[0].hist(walk_lengths, bins=20, edgecolor="black", alpha=0.7)
    axes[0].set_title("Walk Length Distribution")
    axes[0].set_xlabel("Walk Length (steps)")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Panel 2: Action frequency (flattened across all walks)
    all_actions = []
    for walk in trace.agent_info:
        all_actions.extend([step[2] for step in walk[:-1]])  # Exclude terminal step

    action_counts = Counter(all_actions)
    n_actions = max(action_counts.keys()) + 1 if action_counts else 4
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
    shiny_hits = []
    for env, walk in zip(trace.worlds, trace.walks):
        if env.shiny is not None:
            hits = sum(1 for step in walk if step[0].get("shiny", False))
            shiny_hits.append(hits)

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
