"""TEM Figures Module.

Provides domain-namespaced figure modules for environment, walk, and split
visualization, along with core infrastructure (registry, sinks, style).
"""

from torch_tem.figures import style
from torch_tem.figures.modules import cells, environment, overview, split, walk
from torch_tem.figures.registry import REGISTRY, FigureContext, FigureSpec
from torch_tem.figures.sinks import log_tensorboard_figure, save_pdf, save_png

__all__ = [
    "overview",
    "cells",
    "environment",
    "walk",
    "split",
    "style",
    "REGISTRY",
    "FigureContext",
    "FigureSpec",
    "save_pdf",
    "save_png",
    "log_tensorboard_figure",
]
