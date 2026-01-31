"""Figure registry and plotting helpers."""

from torch_tem.figures import plots, register
from torch_tem.figures.modules import grid_cells, place_cells, tem_overview
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.sinks import log_tensorboard_figure, make_figure_path, save_pdf, save_png

__all__ = [
    "FigureContext",
    "log_tensorboard_figure",
    "make_figure_path",
    "tem_overview",
    "grid_cells",
    "place_cells",
    "plots",
    "register",
    "save_pdf",
    "save_png",
]
