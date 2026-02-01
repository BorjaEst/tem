"""Grid-cell diagnostic figure with spatial maps and autocorrelograms."""

from __future__ import annotations

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import SubplotSpec

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.colorbars import colorbar
from torch_tem.figures.figures.templates import SpatialAutocorrTemplate
from torch_tem.figures.plots.autocorr import plot_radial_autocorr_cells, plot_spatial_autocorrelogram
from torch_tem.figures.plots.ratemap import plot_rate_map_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> Figure:
    """Plot a 2x5 MEC grid-cell overview for a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return GridCellsAutocorr(trace, ctx).plot()


class GridCellsAutocorr(SpatialAutocorrTemplate):
    """Encapsulate state and rendering logic for the grid-cell overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("state/lec/cells")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("state/lec/cells", f) for f in range(self.n_freq)]

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.cells = [self.trace.get(f"state/mec/location/mean/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]

    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    def matrices_labels(self, ax: Axes) -> None:
        """Plot radial autocorrelation matrix labels for all cells.

        Args:
            ax: Axes to draw into.
        """
        for cells in self.cells:
            plot_radial_autocorr_cells(ax, self.world, cells, self.location_ids)
        ax.set_title("Mean radial autocorr (±1 std)")

    def autocorrelations(self, ax: Axes) -> None:
        fig, panel_spec = ax.figure, ax.get_subplotspec()
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.01, wspace=0.1, hspace=0.06)
        ax.remove()

        # Create outer grid for frequencies
        n_freq = len(self.cells)
        outer = panel_spec.subgridspec(n_freq, 1, hspace=0.25, wspace=0.00)

        # Plot each frequency block
        for f in range(n_freq):
            self._plot_frequency_block(fig, outer[f, 0], f)

    def _plot_frequency_block(self, fig: Figure, slot: SubplotSpec, f: int) -> None:
        cells_f = self._get_cells_for_freq(f)
        n_cells = int(cells_f.shape[1]) if getattr(cells_f, "ndim", 0) >= 2 else 0
        freq_spec = slot.subgridspec(2, 1, height_ratios=[0.24, 0.83], hspace=0.02)

        # Title axis for frequency
        self._plot_frequency_title(fig, freq_spec[0, 0], f, n_cells)

        # Autocorr grid axis
        self._plot_cell_autocorr_grid(fig, freq_spec[1, 0], cells_f)

    def _plot_frequency_title(self, fig: Figure, slot: SubplotSpec, f: int, n_cells: int) -> None:
        title_ax = fig.add_subplot(slot)
        title_ax.axis("off")
        title_ax.text(0.0, 0.5, self._freq_title(f, n_cells), ha="left", va="center", fontsize=9)

    def _plot_cell_autocorr_grid(self, fig: Figure, slot: SubplotSpec, cells_f) -> None:
        n_cells = int(cells_f.shape[1])
        nrows, ncols = int(10 / self.n_freq), 16  # Rows based on freq
        inner = slot.subgridspec(nrows, ncols, wspace=0.0, hspace=0.0)

        # Optional shared scale per frequency:
        # vmin = vmax = None
        # if shared_scale:
        #     vmin, vmax = build_shared_autocorr_range(
        #         self.world, cells_f, self.location_ids, list(range(n_cells))
        #     )

        for cell_idx in range(n_cells):
            r = cell_idx // ncols
            c = cell_idx % ncols
            subax = fig.add_subplot(inner[r, c])
            # vmax=vmax, vmin=vmin,  # uncomment if using shared scaling
            plot_spatial_autocorrelogram(subax, self.world, cells_f, self.location_ids, cell_idx)

    def _freq_title(self, f: int, n_cells: int) -> str:
        return f"Frequency {f} (n={n_cells})"

    def _get_cells_for_freq(self, f: int):
        return self.cells[f]
