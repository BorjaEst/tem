"""Heatmap plot primitive."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.image import AxesImage

from torch_tem.figures.palettes.sequential import get_colormap
from torch_tem.figures.utils.data import validate_heatmap_data


@dataclass(frozen=True)
class HeatmapResult:
    """Result for heatmap plots with composition metadata."""

    image: AxesImage
    mappable: Any
    norm: Normalize
    vmin: Optional[float]
    vmax: Optional[float]
    colorbar_group: Optional[str] = None


def heatmap(
    ax: Axes,
    *,
    values: Any,
    cmap: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colorbar_group: Optional[str] = None,
    style: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> HeatmapResult:
    """Draw a heatmap on the provided axes.

    Args:
            ax: Matplotlib Axes target.
            values: 2D data values.
            cmap: Optional colormap name.
            vmin: Optional minimum value for normalization.
            vmax: Optional maximum value for normalization.
            colorbar_group: Optional group name for grouped colorbars.
            style: Optional style mapping.
            **kwargs: Additional Matplotlib imshow kwargs.

    Returns:
            HeatmapResult with image and normalization metadata.
    """
    array = validate_heatmap_data(values)
    resolved_vmin = float(np.nanmin(array)) if vmin is None else float(vmin)
    resolved_vmax = float(np.nanmax(array)) if vmax is None else float(vmax)
    norm = Normalize(vmin=resolved_vmin, vmax=resolved_vmax)
    heatmap_kwargs: dict[str, Any] = {
        "cmap": get_colormap(cmap),
        "norm": norm,
        "aspect": "auto",
        "origin": "lower",
    }
    if style:
        heatmap_kwargs.update(style)
    heatmap_kwargs.update(kwargs)
    image = ax.imshow(array, **heatmap_kwargs)
    return HeatmapResult(
        image=image,
        mappable=image,
        norm=norm,
        vmin=resolved_vmin,
        vmax=resolved_vmax,
        colorbar_group=colorbar_group,
    )
