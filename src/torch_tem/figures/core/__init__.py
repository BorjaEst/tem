"""Core tracing and registry infrastructure for the figures subsystem."""

from torch_tem.figures.core.registry import REGISTRY, FigureContext, FigureRegistry, FigureSpec
from torch_tem.figures.core.types import PlotTrace

__all__ = ["PlotTrace", "FigureSpec", "FigureContext", "FigureRegistry", "REGISTRY"]
