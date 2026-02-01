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
    MOSAIC_KWARGS = {"width_ratios": [1.0, 6.0, 1.0]}
    MOSAIC = [
        ["map_labels", "spatial_panels", "memory_panel_a"],
        ["matrices_labels", "spatial_panels", "memory_panel_b"],
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
    def spatial_panels(self, ax: Axes) -> None:
        pass

    # We comment this for now unitl the layer issue is solved
    def spatial_panels_(self, ax: Axes) -> None:
        fig, panel_spec = ax.figure, ax.get_subplotspec()
        ax.remove()

        # Create outer place for frequencies
        n_freq = len(self.cells)
        freq_layout: list[tuple[int, int, int]] = []
        height_ratios: list[float] = []
        title_weight = 0.75
        for f in range(n_freq):
            n_cells = int(self.cells[f].shape[1]) if getattr(self.cells[f], "ndim", 0) >= 2 else 0
            nrows, ncols = self._determine_grid_shape(n_cells)
            freq_layout.append((n_cells, nrows, ncols))
            height_ratios.append(title_weight + max(1, nrows))
        outer = panel_spec.subgridspec(n_freq, 1, height_ratios=height_ratios, hspace=0.25, wspace=0.00)

        # Plot each frequency block
        for f in range(n_freq):
            n_cells, nrows, ncols = freq_layout[f]
            self._plot_frequency_block(fig, outer[f, 0], f, n_cells, nrows, ncols)

    @colorbar(group="memory", label="Memory")
    @panel()
    def memory_panel_a(self, ax: Axes) -> None:
        """Plot HPC memory matrices at the final timestep.

        Args:
            ax: Axes to draw into.
        """
        ax.matshow(self.memory_hier, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    @colorbar(group="memory", label="Memory")
    @panel()
    def memory_panel_b(self, ax: Axes) -> None:
        """Plot HPC memory matrices at the final timestep.

        Args:
            ax: Axes to draw into.
        """
        ax.matshow(self.memory_full, cmap="coolwarm", vmin=-0.1, vmax=0.1)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def _plot_frequency_block(self, fig: Figure, slot: SubplotSpec, f: int, n_cells: int, nrows: int, ncols: int) -> None:
        freq_spec = slot.subgridspec(2, 1, height_ratios=[0.24, 0.83], hspace=0.02)

        # Title axis for frequency
        self._plot_frequency_title(fig, freq_spec[0, 0], f, n_cells)

        # Autocorr place axis
        self._plot_cell_ratemap_grid(fig, freq_spec[1, 0], self.cells[f], nrows, ncols)

    def _plot_frequency_title(self, fig: Figure, slot: SubplotSpec, f: int, n_cells: int) -> None:
        title_ax = fig.add_subplot(slot)
        title_ax.axis("off")
        title = f"Frequency {f} firing rate maps (n={n_cells})"
        title_ax.text(0.0, 0.5, title, ha="left", va="center", fontsize=9)

    def _plot_cell_ratemap_grid(self, fig: Figure, slot: SubplotSpec, cells_f, nrows: int, ncols: int) -> None:
        n_cells = int(cells_f.shape[1])
        inner = slot.subgridspec(nrows, ncols, wspace=0.0, hspace=0.0)
        capacity = max(1, nrows * ncols)

        for cell_idx in range(min(n_cells, capacity)):
            r = cell_idx // ncols
            c = cell_idx % ncols
            subax = fig.add_subplot(inner[r, c])
            plot_ratemap_cell(subax, self.world, cells_f, self.location_ids, cell_idx, vmin=0.0, vmax=0.1)

    def _determine_grid_shape(self, n_cells: int, min_cols: int = 2, max_cols: int = 36) -> tuple[int, int]:
        a = 3 * self.n_freq  # more freq => wider grid
        n_cells = max(1, n_cells)
        ncols = max(min_cols, min(max_cols, round(math.sqrt(n_cells * a))))
        nrows = max(1, math.ceil(n_cells / ncols))
        return nrows, ncols
