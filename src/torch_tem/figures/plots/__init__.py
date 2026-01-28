"""Axis-level plotting primitives."""

from torch_tem.figures.plots.heatmap import HeatmapResult, heatmap
from torch_tem.figures.plots.hist import HistResult, hist
from torch_tem.figures.plots.line import line
from torch_tem.figures.plots.scatter import ScatterResult, scatter

__all__ = [
    "HeatmapResult",
    "HistResult",
    "ScatterResult",
    "heatmap",
    "hist",
    "line",
    "scatter",
]
