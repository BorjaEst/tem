"""Histogram plot primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch


@dataclass(frozen=True)
class HistResult:
    """Result for histogram plots with composition metadata."""

    patches: Sequence[Patch]
    bin_edges: np.ndarray
    legend_handle: Optional[Any]
    label: Optional[str]


def hist(
    ax: Axes,
    *,
    values: Sequence[float],
    bins: Optional[int] = None,
    label: Optional[str] = None,
    style: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> HistResult:
    """Draw a histogram on the provided axes.

    Args:
            ax: Matplotlib Axes target.
            values: Input values.
            bins: Optional number of bins.
            label: Optional legend label.
            style: Optional style mapping.
            **kwargs: Additional Matplotlib hist kwargs.

    Returns:
            HistResult with patches and bin edges.
    """
    hist_kwargs: dict[str, Any] = {}
    if style:
        hist_kwargs.update(style)
    hist_kwargs.update(kwargs)
    if label is not None:
        hist_kwargs["label"] = label
    if bins is not None:
        hist_kwargs["bins"] = bins
    _counts, bin_edges, patches = ax.hist(values, **hist_kwargs)
    legend_handle = patches[0] if label is not None and patches else None
    return HistResult(
        patches=tuple(patches),
        bin_edges=np.asarray(bin_edges),
        legend_handle=legend_handle,
        label=label,
    )
