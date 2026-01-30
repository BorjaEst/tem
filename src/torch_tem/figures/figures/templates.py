from abc import abstractmethod
from typing import Optional

import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from torch_tem.figures.figures.base import BaseFigureTemplate


class OverviewTemplate(BaseFigureTemplate):
    PANEL_NAMES = ["map_labels", "ratemap_a", "ratemap_b", "matrix", "temp_series"]
    BASE_FIGSIZE: float = 1.2
    HEIGHT_FRAC: float = 0.4

    def _create_layout(self, fig: Figure) -> list[Axes]:
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], figure=fig)

        ax_map = fig.add_subplot(gs[0, 0])
        ax_ratemap_a = fig.add_subplot(gs[0, 1])
        ax_ratemap_b = fig.add_subplot(gs[0, 2])
        ax_matrix = fig.add_subplot(gs[1, 0])
        ax_temp_series = fig.add_subplot(gs[1, 1:3])

        return [ax_map, ax_ratemap_a, ax_ratemap_b, ax_matrix, ax_temp_series]

    @abstractmethod
    def map_labels(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def ratemap_a(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def ratemap_b(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrix(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def temp_series(self, ax: Axes) -> None:
        raise NotImplementedError
