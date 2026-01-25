"""Split statistics figure module.

Generates per-split summary statistics for debugging dataset composition.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import DataTrace
from torch_tem.figures.registry import FigureContext


def plot(trace: DataTrace, ctx: FigureContext) -> Figure:
    """Generate dataset split statistics figure.

    Creates a summary figure showing:
    - Number of environments
    - Average walk length
    - Environment size distribution
    - Split metadata

    Args:
        trace: DataTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure with split statistics.
    """
    fig, axes = plt.subplots(2, 2, figsize=ctx.figsize)
    axes = axes.flatten()

    # Panel 1: Environment size distribution (n_locations)
    env_sizes = [env.n_locations for env in trace.worlds]
    axes[0].hist(env_sizes, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
    axes[0].set_title("Environment Size Distribution")
    axes[0].set_xlabel("Number of Locations")
    axes[0].set_ylabel("Frequency")
    axes[0].grid(alpha=0.3)

    # Panel 2: Walk length distribution
    walk_lengths = [len(walk) for walk in trace.walks]
    axes[1].hist(walk_lengths, bins=20, edgecolor="black", alpha=0.7, color="coral")
    axes[1].set_title("Walk Length Distribution")
    axes[1].set_xlabel("Walk Length (steps)")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    # Panel 3: Number of actions per environment
    n_actions_list = [env.n_actions for env in trace.worlds]
    axes[2].hist(n_actions_list, bins=20, edgecolor="black", alpha=0.7, color="mediumseagreen")
    axes[2].set_title("Action Space Size")
    axes[2].set_xlabel("Number of Actions")
    axes[2].set_ylabel("Frequency")
    axes[2].grid(alpha=0.3)

    # Panel 4: Summary statistics table
    axes[3].axis("off")
    summary_text = f"""
    Split: {trace.meta.get('split', 'N/A')}
    
    Batch Size: {trace.batch_size}
    Avg Walk Length: {np.mean(walk_lengths):.1f}
    Avg Env Size: {np.mean(env_sizes):.1f}
    
    Total Steps: {sum(walk_lengths)}
    """
    axes[3].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment="center", family="monospace")

    # Add overall title
    split_name = trace.meta.get("split", "Unknown")
    fig.suptitle(f"Split Statistics: {split_name}", fontsize=16, y=0.995)
    plt.tight_layout()
    return fig
