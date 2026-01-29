"""Model overview figure with memory, observations, and spatial summaries."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.templates import OverviewTemplate
from torch_tem.figures.plots.rasterplot import plot_rasterplot
from torch_tem.figures.plots.ratemap import plot_rate_map_cell
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.scales import build_shared_map_range, build_shared_norm


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
        self.env_idx = trace_access.validate_env_idx(self.trace, self.ctx.env_idx)
        self.freq_idx = trace_access.validate_freq_idx(self.trace, "state/lec/cells", self.ctx.freq_idx)
        self.world = trace_access.get_world(self.trace, self.env_idx)
        self.location_ids = trace_access.get_location_ids(self.trace)[:, self.env_idx]
        self.observations = trace_access.get_observations(self.trace)[:, self.env_idx]

        self.memory_matrix = trace_access.get_hpc_memory(self.trace, self.freq_idx)[0, self.env_idx]
        self.lec_cells = trace_access.get_lec_cells(self.trace, self.freq_idx)[:, self.env_idx, :]
        self.lec_norm = build_shared_norm([self.lec_cells])

        self.map_cell_idx = 0
        self.mec_cells = trace_access.get_mec_cells(self.trace, self.freq_idx)[:, self.env_idx, :]
        self.hpc_cells = trace_access.get_hpc_cells(self.trace, self.freq_idx)[:, self.env_idx, :]
        self.map_min_val, self.map_max_val = build_shared_map_range(self.mec_cells, self.hpc_cells, self.location_ids, len(self.world.locations), self.map_cell_idx)

    def map_labels(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(
            ax,
            self.world,
            self.location_ids.tolist(),
            cmap="plasma",
        )
        ax.set_title("Trajectory colored by time")

    def ratemap_a(self, ax: Axes) -> Axes:
        """Plot a MEC grid-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        plot_rate_map_cell(
            ax,
            self.world,
            self.mec_cells,
            cell_idx=self.map_cell_idx,
            location_ids=self.location_ids.tolist(),
            min_val=self.map_min_val,
            max_val=self.map_max_val,
            location_cm="viridis",
        )
        ax.set_title(f"MEC cells f{self.freq_idx} (cell {self.map_cell_idx})")
        return ax

    def ratemap_b(self, ax: Axes) -> Axes:
        """Plot a HPC place-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        plot_rate_map_cell(
            ax,
            self.world,
            self.hpc_cells,
            cell_idx=self.map_cell_idx,
            location_ids=self.location_ids.tolist(),
            min_val=self.map_min_val,
            max_val=self.map_max_val,
            location_cm="viridis",
        )
        ax.set_title(f"HPC cells f{self.freq_idx} (cell {self.map_cell_idx})")
        return ax

    def matrix(self, ax: Axes) -> None:
        """Plot the hierarchical HPC memory matrix at the final step.

        Args:
            ax: Axes to draw into.
        """
        max_val = float(np.max(np.abs(self.memory_matrix))) if np.isfinite(self.memory_matrix).any() else 1.0
        max_val = max(max_val, 1e-6)
        ax.imshow(self.memory_matrix, cmap="bwr", vmin=-max_val, vmax=max_val)
        ax.set_title("HPC memory (hierarchical)")
        ax.set_xticks([])
        ax.set_yticks([])

    def temp_series(self, ax: Axes) -> Axes:
        """Plot observations and LEC activations over time.

        Args:
            ax: Axes to draw into.
        """
        plot_rasterplot(
            ax,
            observations=self.observations,
            activations=[self.lec_cells],
            activation_names=[f"LEC cells f{self.freq_idx}"],
            act_norm=self.lec_norm,
        )
        ax.set_title(f"Observations and LEC cell f{self.freq_idx} activations over time")
        return ax
