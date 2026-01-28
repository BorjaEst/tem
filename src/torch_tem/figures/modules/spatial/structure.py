"""Spatial structure summary figure."""

from __future__ import annotations

import numpy as np
import matplotlib.figure as mpl_figure

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures import compose
from torch_tem.figures.figures.styles import default_primitive_style, get_theme
from torch_tem.figures.plots import hist
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a spatial structure summary.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    env_idx = trace_access.validate_env_idx(trace, ctx.env_idx)
    location_ids = trace_access.get_location_ids_for_env(trace, env_idx)
    counts = np.asarray(location_ids, dtype=int)
    theme = get_theme("paper")
    hist_style = default_primitive_style("hist", theme)

    def _panel(ax, theme=theme):
        hist(ax, values=counts, bins=20, label="visits", style=hist_style)
        ax.set_title("Location visit counts")
        ax.set_xlabel("location id")
        ax.set_ylabel("count")

    fig = compose(
        panels=[_panel],
        layout=(1, 1),
        theme=theme,
        template="paper",
        legend="none",
        size=ctx.figsize,
    )
    return fig