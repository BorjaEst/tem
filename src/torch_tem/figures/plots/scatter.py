"""Scatter plot primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection

from torch_tem.figures.utils.data import validate_xy


@dataclass(frozen=True)
class ScatterResult:
    """Result for scatter plots with composition metadata."""

    collection: PathCollection
    mappable: Optional[Any]
    legend_handle: Optional[Any]
    label: Optional[str]
    colorbar_group: Optional[str] = None


def scatter(
    ax: Axes,
    *,
    x: Sequence[float],
    y: Sequence[float],
    c: Optional[Sequence[float]] = None,
    label: Optional[str] = None,
    colorbar_group: Optional[str] = None,
    style: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> ScatterResult:
    """Draw a scatter plot on the provided axes.

    Args:
            ax: Matplotlib Axes target.
            x: X values.
            y: Y values.
            c: Optional color values.
            label: Optional legend label.
            style: Optional style mapping.
            **kwargs: Additional Matplotlib scatter kwargs.

        Returns:
            ScatterResult with artist handles and optional mappable.
    """
    x_arr, y_arr = validate_xy(x, y)
    scatter_kwargs: dict[str, Any] = {}
    if style:
        scatter_kwargs.update(style)
    scatter_kwargs.update(kwargs)
    if label is not None:
        scatter_kwargs["label"] = label
    if c is not None:
        scatter_kwargs["c"] = c
    collection = ax.scatter(x_arr, y_arr, **scatter_kwargs)
    mappable: Optional[Any] = None
    if c is not None:
        color_arr = np.asarray(c)
        if color_arr.ndim >= 1 and color_arr.size == x_arr.size:
            mappable = collection
    legend_handle = collection if label is not None else None
    return ScatterResult(
        collection=collection,
        mappable=mappable,
        legend_handle=legend_handle,
        label=label,
        colorbar_group=colorbar_group,
    )
