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
    autocorr = _autocorr(series, max_lag)
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


def _autocorr(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute autocorrelation up to max_lag.

    Args:
        values: 1D array.
        max_lag: Maximum lag.

    Returns:
        Autocorrelation values.
    """
    if max_lag <= 0:
        return np.array([1.0])
    corr = np.correlate(values, values, mode="full")
    mid = corr.size // 2
    corr = corr[mid : mid + max_lag + 1]
    if corr[0] != 0:
        corr = corr / corr[0]
    return corr
