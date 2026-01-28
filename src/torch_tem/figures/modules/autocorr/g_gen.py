"""Autocorrelation figure for inferred states."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures import compose
from torch_tem.figures.figures.styles import default_primitive_style, get_theme
from torch_tem.figures.plots import line
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.autocorr import autocorr_1d


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a simple autocorrelation curve.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    env_idx = trace_access.validate_env_idx(trace, ctx.env_idx)
    series = np.asarray(trace_access.get_location_ids_for_env(trace, env_idx), dtype=float)
    if series.size == 0:
        series = np.zeros(1)
    series = series - np.mean(series)
    max_lag = min(50, series.size - 1) if series.size > 1 else 0
    autocorr = autocorr_1d(series, max_lag)
    lags = np.arange(len(autocorr))

    theme = get_theme("paper")
    line_style = default_primitive_style("line", theme)

    def _panel(ax, theme=theme):
        line(ax, x=lags, y=autocorr, label="autocorr", style=line_style)
        ax.set_title("Autocorrelation (proxy)")
        ax.set_xlabel("lag")
        ax.set_ylabel("correlation")

    fig = compose(
        panels=[_panel],
        layout=(1, 1),
        theme=theme,
        template="paper",
        legend="none",
        size=ctx.figsize,
    )
    return fig
