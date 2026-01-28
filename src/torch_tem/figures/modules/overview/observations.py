"""Overview figure showing key rollout identifiers."""

from __future__ import annotations

from typing import Sequence

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures import compose
from torch_tem.figures.figures.styles import default_primitive_style, get_theme
from torch_tem.figures.plots import line
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a rollout overview figure.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    env_idx = trace_access.validate_env_idx(trace, ctx.env_idx)
    location_ids = _safe_series(trace_access.get_location_ids_for_env(trace, env_idx))
    action_ids = _safe_series(trace_access.get_action_ids_for_env(trace, env_idx))
    observation_ids = _safe_series(trace_access.get_observation_ids_for_env(trace, env_idx))
    steps = np.arange(len(location_ids))

    theme = get_theme("paper")
    line_style = default_primitive_style("line", theme)

    def _panel(ax: Axes, series: Sequence[int], title: str, label: str) -> None:
        line(ax, x=steps, y=series, label=label, style=line_style)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(label)

    panels = [
        lambda ax, theme=theme: _panel(ax, location_ids, "Location IDs", "location"),
        lambda ax, theme=theme: _panel(ax, action_ids, "Action IDs", "action"),
        lambda ax, theme=theme: _panel(ax, observation_ids, "Observation IDs", "observation"),
    ]
    fig = compose(
        panels=panels,
        layout=(3, 1),
        theme=theme,
        template="paper",
        legend="none",
        sharex=True,
        size=ctx.figsize,
    )
    return fig


def _safe_series(values: Sequence[int]) -> np.ndarray:
    """Return a non-empty series for plotting.

    Args:
        values: Input sequence.

    Returns:
        NumPy array with at least one element.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.zeros(1)
    return array
