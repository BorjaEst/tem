"""TEM Figures Module.

Provides domain-namespaced figure modules for environment, walk, and split
visualization, along with core infrastructure (registry, sinks, style).
"""

from torch_tem.figures import plots, style
from torch_tem.figures.modules import overview, spatial
from torch_tem.figures.registry import REGISTRY, FigureContext, FigureSpec
from torch_tem.figures.sinks import log_tensorboard_figure, save_pdf, save_png

__all__ = [
    "overview",
    "spatial",
    "plots",
    "style",
    "REGISTRY",
    "FigureContext",
    "FigureSpec",
    "save_pdf",
    "save_png",
    "log_tensorboard_figure",
]
