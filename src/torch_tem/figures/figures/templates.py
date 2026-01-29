from abc import abstractmethod

from matplotlib.axes import Axes

from torch_tem.figures.figures.base import BaseFigureTemplate, LayoutSpec


class OverviewTemplate(BaseFigureTemplate):
    LAYOUT = {
        "matrix_1": LayoutSpec(type="matrix", position=(0, 0)),
        "map_1": LayoutSpec(type="2D_map", position=(1, 0)),
        "temp_1a": LayoutSpec(type="time_series", position=(0, 1)),
        "temp_1b": LayoutSpec(type="time_series", position=(0, 2)),
        "map_2a": LayoutSpec(type="2D_map", position=(1, 1)),
        "map_2b": LayoutSpec(type="2D_map", position=(1, 2)),
    }
    COLORBAR_GROUPS = {
        "temp_1": {"panels": ["temp_1a", "temp_1b"]},
        "map_2": {"panels": ["map_2a", "map_2b"]},
    }

    @abstractmethod
    def matrix_1(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def temp_1a(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def temp_1b(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def map_1(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def map_2a(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def map_2b(self, ax: Axes) -> None:
        raise NotImplementedError
