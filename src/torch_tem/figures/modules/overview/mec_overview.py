"""Grid-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import math

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import BaseFigureTemplate
from torch_tem.figures.figures.panels import panel
from torch_tem.figures.plots.autocorr import plot_autocorr_mosaic, plot_radial_autocorr_cells
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.axes import subdivide_axes


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
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
    MOSAIC_KWARGS = {"width_ratios": [2.0, 6.0]}
    MOSAIC = [
        ["map_labels", "spatial_matrices"],
        ["matrices_labels", "spatial_matrices"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("state/mec/location/mean")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("state/mec/location/mean", f) for f in range(self.n_freq)]

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = [self.trace.get(f"state/mec/location/mean/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        for cells in self.cells:
            plot_radial_autocorr_cells(ax, self.world, cells, self.location_ids)
        ax.set_title("Radial autocorr (±1 std)")

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_matrices(self, ax: Axes) -> None:
        nrows = len(self.freq_idxs)
        for freq_idx, ax in enumerate(subdivide_axes(ax, nrows, 1, hspace=0.06)):
            cells = self.cells[freq_idx]
            plot_autocorr_mosaic(ax, self.world, cells, self.location_ids)
