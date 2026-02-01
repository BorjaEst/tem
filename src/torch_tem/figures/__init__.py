"""Figure registry and plotting helpers."""

from torch_tem.figures import plots, register
from torch_tem.figures.modules.frequencies import feature_cells, grid_cells, place_cells
from torch_tem.figures.modules.overview import hpc_overview, lec_overview, mec_overview
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.sinks import log_tensorboard_figure, make_figure_path, save_pdf, save_png

__all__ = [
    "FigureContext",
    "log_tensorboard_figure",
    "make_figure_path",
    "lec_overview",
    "mec_overview",
    "hpc_overview",
    "grid_cells",
    "place_cells",
    "feature_cells",
    "plots",
    "register",
    "save_pdf",
    "save_png",
]
