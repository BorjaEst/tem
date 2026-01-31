"""Model overview figure with memory, observations, and spatial summaries."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.colorbars import colorbar
from torch_tem.figures.figures.templates import OverviewTemplate
from torch_tem.figures.plots.rasterplot import plot_rasterplot
from torch_tem.figures.plots.ratemap import plot_rate_map_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a 2x3 model overview focused on a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    return RolloutOverview(trace, ctx).plot()


class RolloutOverview(OverviewTemplate):
    """Encapsulate state and rendering logic for the model overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        """Initialize the figure state from a trace and rendering context.

        Args:
            trace: TraceTree with rollout data.
            ctx: Figure context.
        """
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("state/lec/cells", self.ctx.freq_idx)
        self.map_cell_idx = 0

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.observations = self.trace.get("world_step/observation")[:, self.env_idx]
        self.memory_matrix = self.trace.get("state/hpc/_memory/0")[0, self.env_idx]

        self.lec_cells = self.trace.get(f"state/lec/cells/{self.freq_idx}")[:, self.env_idx, :]
        self.mec_cells = abs(self.trace.get(f"state/mec/location/mean/{self.freq_idx}")[:, self.env_idx, :])
        self.hpc_cells = abs(self.trace.get(f"state/hpc/location/mean/{self.freq_idx}")[:, self.env_idx, :])

    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    @colorbar(group="ratemaps", label="Firing rate")
    def ratemap_a(self, ax: Axes) -> None:
        """Plot a MEC grid-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        options = {"vmin": 0.0, "vmax": 1.0}
        plot_rate_map_cell(ax, self.world, self.mec_cells, cell_idx=self.map_cell_idx, location_ids=self.location_ids.tolist(), **options)
        ax.set_title(f"MEC cells f{self.freq_idx} (cell {self.map_cell_idx})")

    @colorbar(group="ratemaps", label="Firing rate")
    def ratemap_b(self, ax: Axes) -> None:
        """Plot a HPC place-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        options = {"vmin": 0.0, "vmax": 1.0}
        plot_rate_map_cell(ax, self.world, self.hpc_cells, cell_idx=self.map_cell_idx, location_ids=self.location_ids.tolist(), **options)
        ax.set_title(f"HPC cells f{self.freq_idx} (cell {self.map_cell_idx})")

    def matrix(self, ax: Axes) -> None:
        """Plot the hierarchical HPC memory matrix at the final step.

        Args:
            ax: Axes to draw into.
        """
        vmax = float(np.max(np.abs(self.memory_matrix))) if np.isfinite(self.memory_matrix).any() else 1.0
        vmax = max(vmax, 1e-6)
        ax.matshow(self.memory_matrix, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title("HPC memory (hierarchical)")
        ax.set_xticks([])
        ax.set_yticks([])

    @colorbar(group="ratetime", label="Firing rate")
    def temp_series(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time.

        Args:
            ax: Axes to draw into.
        """
        ax.set_axis_off()
        raster_ax = ax.inset_axes([0.0, 0.0, 1.0, 1.0])
        options = {"vmin": 0.0, "vmax": 1.0, "activation_names": [f"LEC cells f{self.freq_idx}"]}
        plot_rasterplot(raster_ax, observations=self.observations, activations=[self.lec_cells], **options)
        raster_ax.set_title(f"Observations and LEC cell f{self.freq_idx} activations over time")

    def params(self, ax: Axes) -> None:
        """Plot per-frequency LEC parameters if available."""
        n_freq = min(len(self.alpha), len(self.w_f))
        freq_ids = np.arange(n_freq)
        ax.plot(freq_ids, self.alpha[:n_freq], marker="o", label="sigmoid(alpha)")
        ax.plot(freq_ids, self.w_f[:n_freq], marker="s", label="sigmoid(w_f)")
        ax.axvline(self.freq_idx, linestyle="--", color="gray", linewidth=1.0)
        ax.set_title("LEC parameters by frequency")
        ax.set_xlabel("Frequency index")
        ax.set_ylabel("Value")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize="small")
