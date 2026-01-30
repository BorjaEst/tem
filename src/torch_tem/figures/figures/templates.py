from abc import abstractmethod
from typing import Optional

from matplotlib.axes import Axes

from torch_tem.figures.figures.base import BaseFigureTemplate, LayoutSpec


class OverviewTemplate(BaseFigureTemplate):
    LAYOUT = {
        "map_labels": LayoutSpec(position=(0, 0)),
        "ratemap_a": LayoutSpec(position=(0, 1)),
        "ratemap_b": LayoutSpec(position=(0, 2)),
        "matrix": LayoutSpec(position=(1, 0)),
        "temp_series": LayoutSpec(position=(1, 1), colspan=2),
    }
    COLORBAR_GROUPS = {
        "temp_series": {
            "panels": ["temp_series"],
            "source": "temp_series",
            "pad": 0.02,
            "fraction": 0.046,
            "shrink": 0.9,
            "aspect": 30,
        },
        "rate_maps": {
            "panels": ["ratemap_a", "ratemap_b"],
            "source": "ratemap_b",
            "pad": 0.02,
            "fraction": 0.046,
            "shrink": 0.9,
            "aspect": 30,
        },
    }
    SUBPLOTS_ADJUST = {
        "left": 0.07,
        "right": 0.95,
        "top": 0.93,
        "bottom": 0.07,
        "hspace": 0.3,
        "wspace": 0.3,
    }

    @abstractmethod
    def map_labels(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def ratemap_a(self, ax: Axes) -> Axes:
        raise NotImplementedError

    @abstractmethod
    def ratemap_b(self, ax: Axes) -> Axes:
        raise NotImplementedError

    @abstractmethod
    def matrix(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def temp_series(self, ax: Axes) -> Axes:
        raise NotImplementedError
