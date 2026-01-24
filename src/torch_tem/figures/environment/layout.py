"""Environment layout figure module.

Renders a static environment layout showing locations, connectivity, and optionally
per-location scalar values or policy information.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.figures.core.data_trace import DataTrace
from torch_tem.figures.core.registry import FigureContext
from torch_tem.figures.primitives import plot_map


def plot(trace: DataTrace, ctx: FigureContext) -> Figure:
    """Generate environment layout figure.

    Renders the spatial layout of an environment, showing:
    - Location positions and connectivity
    - Shiny object locations (if present)
    - Optionally: per-location values or policy arrows

    Args:
        trace: DataTrace with environment(s) and walk(s).
        ctx: Figure context (env_idx, figsize, style, etc.).

    Returns:
        matplotlib Figure showing the environment layout.
    """
    # Apply style context if provided
    style_ctx = (ctx.style or _get_default_style()).apply_context() if hasattr(ctx, "style") and ctx.style else _noop_context()

    with style_ctx:
        # Select single environment if batch trace
        if trace.batch_size > 1:
            trace = trace.select_env(ctx.env_idx)

        env = trace.worlds[0]
        n_locs = env.n_locations

        # Create figure
        fig, ax = plt.subplots(figsize=ctx.figsize)

        # Render environment with uniform values (just show layout)
        values = np.ones(n_locs)  # Uniform coloring
        plot_map(
            env,
            values,
            ax=ax,
            do_plot_actions=False,  # Don't clutter with action arrows by default
            shape="circle",
        )

        # Add title with environment metadata
        title = f"Environment Layout (n_locations={n_locs})"
        if ctx.split_name:
            title += f" - {ctx.split_name}"
        ax.set_title(title, fontsize=14, pad=10)

        plt.tight_layout()
        return fig


def _get_default_style():
    """Get default style config if torch_tem.figures.style is available."""
    try:
        from torch_tem.figures.style import StyleConfig

        return StyleConfig()
    except ImportError:
        return None


class _noop_context:
    """No-op context manager for when style is not available."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
