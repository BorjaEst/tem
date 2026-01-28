"""Figure registry and plotting helpers."""

from torch_tem.figures import plots, register
from torch_tem.figures.figures import compose, make_grid
from torch_tem.figures.modules import autocorr, overview, spatial
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.sinks import log_tensorboard_figure, make_figure_path, save_pdf

__all__ = [
    "FigureContext",
    "autocorr",
    "compose",
    "log_tensorboard_figure",
    "make_figure_path",
    "make_grid",
    "overview",
    "plots",
    "register",
    "save_pdf",
    "spatial",
]
