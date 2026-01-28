"""Line plot primitive."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from torch_tem.figures.utils.data import validate_xy
from torch_tem.figures.utils.line import resolve_line_style


def line(
    ax: Axes,
    *,
    x: Sequence[float],
    y: Sequence[float],
    label: Optional[str] = None,
    style: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Line2D:
    """Draw a line on the provided axes.

    Args:
            ax: Matplotlib Axes target.
            x: X values.
            y: Y values.
            label: Optional legend label.
            style: Optional style mapping.
            **kwargs: Additional Matplotlib line kwargs.

    Returns:
            The created Line2D artist.
    """
    x_arr, y_arr = validate_xy(x, y)
    line_kwargs = resolve_line_style(style, **kwargs)
    if label is not None:
        line_kwargs["label"] = label
    (artist,) = ax.plot(x_arr, y_arr, **line_kwargs)
    return artist
