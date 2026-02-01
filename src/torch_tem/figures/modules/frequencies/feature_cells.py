from __future__ import annotations

import matplotlib.figure as mpl_figure
import torch
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import BaseFigureTemplate
from torch_tem.figures.figures.panels import colorbar, panel
from torch_tem.figures.plots.rasterplot import plot_rasterplot
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a per-frequency LEC overview with observations and parameters."""
    return FeatCellsTimeseries(trace, ctx).plot()


class FeatCellsTimeseries(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the LEC overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("state/lec/cells", self.ctx.freq_idx)

        self.observations = self.trace.get("world_step/observation")[:, self.env_idx]
        self.feature_series = self.trace.get("output/features")[:, self.env_idx, :]
        self.filtered_series = self.trace.get(f"state/lec/filtered/{self.freq_idx}")[:, self.env_idx, :]
        self.cell_series = self.trace.get(f"state/lec/cells/{self.freq_idx}")[:, self.env_idx, :]

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_1(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        options = {"vmin": 0.0, "vmax": 1.0, "obs_height": 0.3, "activation_names": []}
        options["activation_names"].append("Input")
        options["activation_names"].append(f"Cells f{self.freq_idx}")
        activations = [self.feature_series, self.cell_series]
        plot_rasterplot(ax, self.observations, activations, **options)
        ax.set_title("LEC input features and activations (after ponderation)")

    @colorbar(group="lec_activity", label="Activation")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def raster_2(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        options = {"vmin": 0.0, "vmax": 1.0, "obs_height": 0.45, "activation_names": []}
        options["activation_names"].append(f"Filtered f{self.freq_idx}")
        activations = [self.filtered_series]
        plot_rasterplot(ax, self.observations, activations, **options)
        ax.set_title("Filtered activations (before ponderation)")
