from abc import abstractmethod
from typing import Dict, Optional

import matplotlib.gridspec as gridspec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from torch_tem.figures.figures.base import BaseFigureTemplate


class OverviewTemplate(BaseFigureTemplate):
    BASE_FIGSIZE: float = 1.00
    HEIGHT_FRAC: float = 0.45

    def _create_layout(self, fig: Figure) -> Dict[str, Axes]:
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], figure=fig)
        return {
            "map_labels": fig.add_subplot(gs[0, 0]),
            "ratemap_a": fig.add_subplot(gs[0, 1]),
            "ratemap_b": fig.add_subplot(gs[0, 2]),
            "matrix": fig.add_subplot(gs[1, 0]),
            "temp_series": fig.add_subplot(gs[1, 1:3]),
        }

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


class SpatialMatrix4Template(BaseFigureTemplate):
    BASE_FIGSIZE: float = 1.50
    HEIGHT_FRAC: float = 0.30

    def _create_layout(self, fig: Figure) -> Dict[str, Axes]:
        gs = gridspec.GridSpec(2, 5, height_ratios=[1] * 2, width_ratios=[1] * 5, figure=fig)
        return {
            # Map and matrix label axes
            "map_labels": fig.add_subplot(gs[0, 0]),  # map-labels
            "matrices_labels": fig.add_subplot(gs[1, 0]),  # matrix-labels
            # Spatial maps
            "spatial_map_a": fig.add_subplot(gs[0, 1]),  # Spatial map a
            "spatial_map_b": fig.add_subplot(gs[0, 2]),  # Spatial map b
            "spatial_map_c": fig.add_subplot(gs[0, 3]),  # Spatial map c
            "spatial_map_d": fig.add_subplot(gs[0, 4]),  # Spatial map d
            # Matrices
            "matrix_a": fig.add_subplot(gs[1, 1]),  # Matrix a
            "matrix_b": fig.add_subplot(gs[1, 2]),  # Matrix b
            "matrix_c": fig.add_subplot(gs[1, 3]),  # Matrix c
            "matrix_d": fig.add_subplot(gs[1, 4]),  # Matrix d
        }

    @abstractmethod
    def map_labels(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrices_labels(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def spatial_map_a(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrix_a(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def spatial_map_b(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrix_b(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def spatial_map_c(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrix_c(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def spatial_map_d(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def matrix_d(self, ax: Axes) -> None:
        raise NotImplementedError


class LECOverviewTemplate(BaseFigureTemplate):
    BASE_FIGSIZE: float = 1.00
    HEIGHT_FRAC: float = 0.45

    def _create_layout(self, fig: Figure) -> Dict[str, Axes]:
        gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], figure=fig)
        return {
            "trajectory": fig.add_subplot(gs[0, 0]),
            "features": fig.add_subplot(gs[0, 1]),
            "filtered": fig.add_subplot(gs[0, 2]),
            "cells": fig.add_subplot(gs[1, 0]),
            "raster": fig.add_subplot(gs[1, 1]),
            "params": fig.add_subplot(gs[1, 2]),
        }

    @abstractmethod
    def trajectory(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def features(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def filtered(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def cells(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def raster(self, ax: Axes) -> None:
        raise NotImplementedError

    @abstractmethod
    def params(self, ax: Axes) -> None:
        raise NotImplementedError
