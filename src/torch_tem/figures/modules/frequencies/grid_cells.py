"""Grid-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import BaseFigureTemplate
from torch_tem.figures.figures.panels import colorbar, panel
from torch_tem.figures.plots.autocorr import plot_radial_autocorr_cells, plot_spatial_autocorrelogram
from torch_tem.figures.plots.ratemap import plot_rate_map_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a 2x5 MEC grid-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return GridCellsAutocorr(trace, ctx).plot()


class GridCellsAutocorr(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the grid-cell overview."""

    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [2.0, 2.0, 2.0, 2.0, 2.0]}
    MOSAIC = [
        ["map_labels", "spatial_map_a", "spatial_map_b", "spatial_map_c", "spatial_map_d"],
        ["matrices_labels", "matrix_a", "matrix_b", "matrix_c", "matrix_d"],
    ]

    COLORBAR_VMAX = 1.00
    CELLS = {"a": 0, "b": 1, "c": 2, "d": 3}

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("state/mec/location/mean", self.ctx.freq_idx)
        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = self.trace.get(f"state/mec/location/mean/{self.freq_idx}")[:, self.env_idx, :]

    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    def _plot_rate_map(self, ax: Axes, cell_idx: int) -> None:
        options = {"vmin": 0.0, "vmax": self.COLORBAR_VMAX}
        plot_rate_map_cell(ax, self.world, self.cells, self.location_ids.tolist(), cell_idx, **options)
        ax.set_title(f"MEC f{self.freq_idx} cell {cell_idx} rate map")

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_map_a(self, ax: Axes) -> None:
        """Plot a MEC rate map for cell 0.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, self.CELLS["a"])

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_map_b(self, ax: Axes) -> None:
        """Plot a MEC rate map for cell 1.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, self.CELLS["b"])

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_map_c(self, ax: Axes) -> None:
        """Plot a MEC rate map for cell 2.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, self.CELLS["c"])

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_map_d(self, ax: Axes) -> None:
        """Plot a MEC rate map for cell 3.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, self.CELLS["d"])

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_radial_autocorr_cells(ax, self.world, self.cells, self.location_ids)
        ax.set_title("Mean radial autocorr (±1 std)")

    def _plot_autocorr(self, ax: Axes, cell_idx: int) -> None:
        options = {"vmin": -self.COLORBAR_VMAX, "vmax": self.COLORBAR_VMAX}
        plot_spatial_autocorrelogram(ax, self.world, self.cells, self.location_ids, cell_idx, **options)
        ax.set_title(f"MEC f{self.freq_idx} cell {cell_idx} autocorr")

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrix_a(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 0.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, self.CELLS["a"])

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrix_b(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 1.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, self.CELLS["b"])

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrix_c(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 2.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, self.CELLS["c"])

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrix_d(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 3.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, self.CELLS["d"])
