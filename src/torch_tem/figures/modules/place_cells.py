"""Place-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
from matplotlib.axes import Axes

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.colorbars import colorbar
from torch_tem.figures.figures.templates import SpatialMatrix4Template
from torch_tem.figures.plots.autocorr import plot_radial_autocorr_cells, plot_spatial_autocorrelogram
from torch_tem.figures.plots.ratemap import plot_rate_map_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a 2x5 HPC place-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return PlaceCellsAutocorr(trace, ctx).plot()


class PlaceCellsAutocorr(SpatialMatrix4Template):
    """Encapsulate state and rendering logic for the place-cell overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.env_idx = trace_access.validate_env_idx(self.trace, self.ctx.env_idx)
        self.freq_idx = trace_access.validate_freq_idx(self.trace, "output/inference/p_inf", self.ctx.freq_idx)
        self.world = trace_access.get_world(self.trace, self.env_idx)
        self.location_ids = trace_access.get_location_ids(self.trace)[:, self.env_idx]
        self.cells = trace_access.get_hpc_cells(self.trace, self.freq_idx)[:, self.env_idx, :]

    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist(), cmap="plasma")
        ax.set_title("Trajectory colored by time")

    def _plot_rate_map(self, ax: Axes, cell_idx: int) -> None:
        options = {"vmin": 0.0, "vmax": 1.0}
        plot_rate_map_cell(ax, self.world, self.cells, cell_idx, location_ids=self.location_ids.tolist(), **options)
        ax.set_title(f"HPC f{self.freq_idx} cell {cell_idx} rate map")

    @colorbar(group="ratemaps", label="Firing rate")
    def spatial_map_a(self, ax: Axes) -> None:
        """Plot a HPC rate map for cell 0.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, 0)

    @colorbar(group="ratemaps", label="Firing rate")
    def spatial_map_b(self, ax: Axes) -> None:
        """Plot a HPC rate map for cell 1.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, 1)

    @colorbar(group="ratemaps", label="Firing rate")
    def spatial_map_c(self, ax: Axes) -> None:
        """Plot a HPC rate map for cell 2.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, 2)

    @colorbar(group="ratemaps", label="Firing rate")
    def spatial_map_d(self, ax: Axes) -> None:
        """Plot a HPC rate map for cell 3.

        Args:
            ax: Axes to draw into.
        """
        self._plot_rate_map(ax, 3)

    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_radial_autocorr_cells(ax, self.world, self.cells, self.location_ids)
        ax.set_title("Mean radial autocorr (±1 std)")

    def _plot_autocorr(self, ax: Axes, cell_idx: int) -> None:
        options = {"vmin": -1.0, "vmax": 1.0}
        plot_spatial_autocorrelogram(ax, self.world, self.cells, self.location_ids, cell_idx, **options)
        ax.set_title(f"HPC f{self.freq_idx} cell {cell_idx} autocorr")

    @colorbar(group="autocorr", label="Autocorr")
    def matrix_a(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 0.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, 0)

    @colorbar(group="autocorr", label="Autocorr")
    def matrix_b(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 1.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, 1)

    @colorbar(group="autocorr", label="Autocorr")
    def matrix_c(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 2.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, 2)

    @colorbar(group="autocorr", label="Autocorr")
    def matrix_d(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for cell 3.

        Args:
            ax: Axes to draw into.
        """
        self._plot_autocorr(ax, 3)
