"""Trajectory plot primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from torch_tem.figures.plots.scatter import ScatterResult, scatter
from torch_tem.figures.utils.data import validate_xy


@dataclass(frozen=True)
class TrajectoryResult:
    """Result for trajectory plots with composition metadata."""

    scatter: ScatterResult
    line: Line2D
    mappable: Optional[Any]
    legend_handle: Optional[Any]
    label: Optional[str]
    colorbar_group: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None


def trajectory(
    ax: Axes,
    *,
    x: Sequence[float],
    y: Sequence[float],
    t: Optional[Sequence[float]] = None,
    c: Optional[Sequence[float]] = None,
    cmap: Optional[str] = None,
    label: Optional[str] = None,
    colorbar_group: Optional[str] = None,
    scatter_style: Optional[Mapping[str, Any]] = None,
    line_style: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> TrajectoryResult:
    """Draw a trajectory on the provided axes.

    Args:
        ax: Matplotlib Axes target.
        x: X values.
        y: Y values.
        t: Optional time values to color by if c is not provided.
        c: Optional color values.
        cmap: Optional colormap name for scatter.
        label: Optional legend label.
        colorbar_group: Optional group name for grouped colorbars.
        scatter_style: Optional scatter style mapping.
        line_style: Optional line style mapping.
        **kwargs: Additional Matplotlib scatter kwargs.

    Returns:
        TrajectoryResult with artist handles and optional mappable.
    """
    x_arr, y_arr = validate_xy(x, y)
    color_values = c if c is not None else t

    resolved_scatter_style: dict[str, Any] = {}
    if scatter_style:
        resolved_scatter_style.update(scatter_style)
    if color_values is not None:
        resolved_scatter_style.pop("color", None)

    resolved_line_style: dict[str, Any] = {
        "color": "0.5",
        "linewidth": 1.0,
        "alpha": 0.5,
    }
    if line_style:
        resolved_line_style.update(line_style)

    scatter_result = scatter(
        ax,
        x=x_arr,
        y=y_arr,
        c=color_values,
        label=label,
        colorbar_group=colorbar_group,
        style=resolved_scatter_style,
        cmap=cmap,
        **kwargs,
    )
    (line_artist,) = ax.plot(x_arr, y_arr, **resolved_line_style)
    return TrajectoryResult(
        scatter=scatter_result,
        line=line_artist,
        mappable=scatter_result.mappable,
        legend_handle=scatter_result.legend_handle,
        label=scatter_result.label,
        colorbar_group=scatter_result.colorbar_group,
    )
