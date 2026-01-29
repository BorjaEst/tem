"""Model overview figure with memory, observations, and spatial summaries."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.templates import OverviewTemplate
from torch_tem.figures.plots.rate_map import plot_rate_map_cell
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
        self.env_idx = trace_access.validate_env_idx(self.trace, self.ctx.env_idx)
        self.freq_idx = trace_access.validate_freq_idx(self.trace, "state/lec/cells", self.ctx.freq_idx)
        self.world = trace_access.get_world(self.trace, self.env_idx)
        self.location_ids = trace_access.get_location_ids(self.trace)[:, self.env_idx]
        self.observations = trace_access.get_observations(self.trace)[:, self.env_idx]

        self.memory_matrix = trace_access.get_hpc_memory(self.trace, self.freq_idx)[0, self.env_idx]
        self.lec_cells = trace_access.get_lec_cells(self.trace, self.freq_idx)[:, self.env_idx, :]
        self.mec_cells = trace_access.get_mec_cells(self.trace, self.freq_idx)[:, self.env_idx, :]
        self.hpc_cells = trace_access.get_hpc_cells(self.trace, self.freq_idx)[:, self.env_idx, :]

    def matrix_1(self, ax: Axes) -> None:
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

    def temp_1a(self, ax: Axes) -> None:
        """Plot observations over time.

        Args:
            ax: Axes to draw into.
        """
        ax.imshow(self.observations.T, aspect="auto", cmap="viridis")
        ax.set_title("Observations over time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Observation dimension")
        ax.set_yticks([])

    def temp_1b(self, ax: Axes) -> None:
        """Plot LEC cell activations for a selected frequency.

        Args:
            ax: Axes to draw into.
        """
        ax.imshow(self.lec_cells.T, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xlabel("Time")
        ax.set_ylabel("Cells")
        ax.set_yticks([])
        ax.set_title(f"LEC cells f{self.freq_idx}")

    def map_1(self, ax: Axes) -> None:
        """Plot the trajectory colored by time.

        Args:
            ax: Axes to draw into.
        """
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist(), cmap="plasma")
        ax.set_title("Trajectory colored by time")

    def map_2a(self, ax: Axes) -> None:
        """Plot a MEC grid-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        plot_rate_map_cell(ax, self.world, self.mec_cells, cell_idx=0, location_ids=self.location_ids.tolist(), location_cm="viridis")
        ax.set_title(f"MEC cells f{self.freq_idx} (cell {0})")

    def map_2b(self, ax: Axes) -> None:
        """Plot a HPC place-cell rate map for the selected frequency.

        Args:
            ax: Axes to draw into.
        """
        plot_rate_map_cell(ax, self.world, self.hpc_cells, cell_idx=0, location_ids=self.location_ids.tolist(), location_cm="viridis")
        ax.set_title(f"HPC cells f{self.freq_idx} (cell {0})")
