"""Core tracing and registry infrastructure for the figures subsystem."""

from torch_tem.figures.core.collect import collect_trace
from torch_tem.figures.core.registry import REGISTRY, FigureContext, FigureRegistry, FigureSpec
from torch_tem.figures.core.types import PlotTrace, TraceExtractor

__all__ = [
    "PlotTrace",
    "TraceExtractor",
    "collect_trace",
    "FigureSpec",
    "FigureContext",
    "FigureRegistry",
    "REGISTRY",
]
