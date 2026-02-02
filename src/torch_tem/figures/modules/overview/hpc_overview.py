"""Place-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

import math

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.base import BaseFigureTemplate
from torch_tem.figures.figures.panels import colorbar, panel
from torch_tem.figures.plots.autocorr import plot_radial_autocorr_cells
from torch_tem.figures.plots.ratemap import plot_ratemap_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot a 2x5 HPC place-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return PlaceCellsAutocorr(trace, ctx).plot()


class PlaceCellsAutocorr(BaseFigureTemplate):
    """Encapsulate state and rendering logic for the place-cell overview."""

    HEIGHT_FRAC: float = 0.40
    MOSAIC_KWARGS = {"width_ratios": [2.0, 4.0, 2.0]}
    MOSAIC = [
        ["map_labels", "spatial_matrices", "memory_panel_a"],
        ["matrices_labels", "spatial_matrices", "memory_panel_b"],
    ]

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("state/hpc/location/mean")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("state/hpc/location/mean", f) for f in range(self.n_freq)]

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = [self.trace.get(f"state/hpc/location/mean/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]
        self.memory_hier = self.trace.get("state/hpc/_memory/0")[-1, self.env_idx]
        self.memory_full = self.trace.get("state/hpc/_memory/1")[-1, self.env_idx]

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
        ax.set_title("Mean radial autocorr (±1 std)")

    @panel()  # Here some arguments to configure the pannel, position, etc.
    def spatial_matrices(self, ax: Axes) -> None:
        pass

    @colorbar(group="memory", label="Strength (0.0 to 0.1)")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def memory_panel_a(self, ax: Axes) -> None:
        """Plot HPC hierarchical memory matrices at the final timestep.

        Args:
            ax: Axes to draw into.
        """
        ax.matshow(self.memory_hier, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    @colorbar(group="memory", label="Strength (0.0 to 0.1)")
    @panel()  # Here some arguments to configure the pannel, position, etc.
    def memory_panel_b(self, ax: Axes) -> None:
        """Plot HPC full memory matrices at the final timestep.

        Args:
            ax: Axes to draw into.
        """
        ax.matshow(self.memory_full, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
