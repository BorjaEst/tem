"""Figure registry and plotting helpers."""

from torch_tem.figures import plots, register
from torch_tem.figures.modules import overview
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.sinks import log_tensorboard_figure, make_figure_path, save_pdf

__all__ = [
    "FigureContext",
    "log_tensorboard_figure",
    "make_figure_path",
    "overview",
    "plots",
    "register",
    "save_pdf",
]
