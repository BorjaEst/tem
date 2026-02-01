"""Grid-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import BaseFigureTemplate
from torch_tem.figures.figures.panels import colorbar, panel
from torch_tem.figures.plots.autocorr import plot_autocorr_mosaic, plot_radial_autocorr_cells
from torch_tem.figures.plots.ratemap import plot_ratemap_mosaic
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
    MOSAIC_KWARGS = {"width_ratios": [1.0, 4.0]}
    MOSAIC = [
        ["map_labels", "spatial_maps"],
        ["matrices_labels", "spatial_matrices"],
    ]

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

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    @colorbar(group="ratemaps", label="Firing rate")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_maps(self, ax: Axes) -> None:
        """Plot a HPC rate map for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_ratemap_mosaic(ax, self.world, self.cells, self.location_ids.tolist())
        # ax.set_title(f"HPC f{self.freq_idx} rate map")

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        plot_radial_autocorr_cells(ax, self.world, self.cells, self.location_ids)
        ax.set_title("Mean radial autocorr (±1 std)")

    @colorbar(group="autocorr", label="Autocorr")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_matrices(self, ax: Axes) -> None:
        """Plot a spatial autocorrelogram for all cell.

        Args:
            ax: Axes to draw into.
        """
        plot_autocorr_mosaic(ax, self.world, self.cells, self.location_ids)
        # ax.set_title(f"HPC f{self.freq_idx} spatial autocorr")
