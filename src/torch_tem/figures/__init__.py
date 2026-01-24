"""TEM Figures Module.

Provides domain-namespaced figure modules for environment, walk, and split
visualization, along with core infrastructure (registry, sinks, style).

Usage:
    from torch_tem import figures

    # Access domain modules
    figures.environment.layout.plot(trace, ctx)
    figures.walk.trajectories.plot(trace, ctx)

    # Access infrastructure
    from torch_tem.figures.core import REGISTRY, FigureContext
    from torch_tem.figures import style, sinks
"""

from torch_tem.figures import environment, split, style, walk
from torch_tem.figures.core import REGISTRY, FigureContext, FigureSpec
from torch_tem.figures.sinks import log_tensorboard_figure, save_pdf, save_png

__all__ = [
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
