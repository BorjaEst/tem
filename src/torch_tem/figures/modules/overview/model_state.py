"""Model overview figure with memory, observations, and spatial summaries."""

from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures import compose_gridspec
from torch_tem.figures.figures.styles import default_primitive_style, get_theme
from torch_tem.figures.plots import heatmap, scatter, trajectory
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.axes import format_map_axes
from torch_tem.figures.utils.spatial import aggregate_by_location, path_xy_from_location_ids, world_xy_from_locations


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a 2x3 model overview focused on a single frequency module.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context.

    Returns:
        Matplotlib Figure instance.
    """
    env_idx = trace_access.validate_env_idx(trace, ctx.env_idx)
    freq_idx = _validate_freq(trace, ctx.freq_idx)
    world = trace_access.get_world(trace, env_idx)

    obs = _safe_2d(_select_env(trace_access.get_observations(trace), env_idx))
    lec = _safe_2d(_select_env(trace_access.get_lec_cells(trace, freq_idx), env_idx))
    mec = _select_env(trace_access.get_mec_cells(trace, freq_idx), env_idx)
    hpc = _select_env(trace_access.get_hpc_cells(trace, freq_idx), env_idx)
    memory = _extract_hier_memory(trace, env_idx)
    loc_ids = np.asarray(trace_access.get_location_ids_for_env(trace, env_idx))

    mec_last = aggregate_by_location(
        loc_ids,
        _safe_1d(mec, default=0.0),
        n_locations=world.n_locations,
        mode="last",
    )
    hpc_last = aggregate_by_location(
        loc_ids,
        _safe_1d(hpc, default=0.0),
        n_locations=world.n_locations,
        mode="last",
    )

    theme = get_theme("paper")
    scatter_style = {key: value for key, value in default_primitive_style("scatter", theme).items() if key != "color"} | {"s": 45}

    def _memory_panel(ax, theme=theme):
        result = heatmap(ax, values=memory, cmap="viridis", colorbar_group="memory")
        ax.set_title("HPC hierarchical memory (t=last)")
        ax.set_xlabel("cell")
        ax.set_ylabel("cell")
        return result

    def _obs_panel(ax, theme=theme):
        heatmap(ax, values=obs.T, cmap="Greys", vmin=0, vmax=1)
        ax.set_title("Observations (one-hot)")
        ax.set_xlabel("step")
        ax.set_ylabel("observation")

    def _lec_panel(ax, theme=theme):
        result = heatmap(ax, values=lec.T, cmap="viridis", colorbar_group="lec")
        ax.set_title(f"LEC cells (freq={freq_idx})")
        ax.set_xlabel("step")
        ax.set_ylabel("cell")
        return result

    def _trajectory_panel(ax, theme=theme):
        xs, ys = path_xy_from_location_ids(world.locations, loc_ids)
        if xs.size:
            trajectory(
                ax,
                x=xs,
                y=ys,
                scatter_style={
                    "color": theme.tokens["secondary"],
                    "s": 6,
                    "alpha": 0.6,
                },
                line_style={
                    "color": theme.tokens["primary"],
                    "linewidth": 1.5,
                },
            )
        ax.set_title("Trajectory")
        format_map_axes(ax)

    def _mec_panel(ax, theme=theme):
        x, y = world_xy_from_locations(world.locations)
        values = np.ma.masked_invalid(mec_last)
        result = scatter(
            ax,
            x=x,
            y=y,
            c=values,
            cmap="viridis",
            colorbar_group="mec",
            style=scatter_style,
        )
        ax.set_title(f"MEC cell 0 (freq={freq_idx})")
        format_map_axes(ax)
        return result

    def _hpc_panel(ax, theme=theme):
        x, y = world_xy_from_locations(world.locations)
        values = np.ma.masked_invalid(hpc_last)
        result = scatter(
            ax,
            x=x,
            y=y,
            c=values,
            cmap="viridis",
            colorbar_group="hpc",
            style=scatter_style,
        )
        ax.set_title(f"HPC cell 0 (freq={freq_idx})")
        format_map_axes(ax)
        return result

    fig = compose_gridspec(
        panels=[
            _memory_panel,
            _obs_panel,
            _lec_panel,
            _trajectory_panel,
            _mec_panel,
            _hpc_panel,
        ],
        layout=(2, 3),
        theme=theme,
        template="paper",
        legend="none",
        colorbar="grouped",
        size=ctx.figsize,
        gridspec_layout={
            "nrows": 2,
            "ncols": 6,
            "width_ratios": [1, 0.06, 1, 0.06, 1, 0.06],
        },
        content_positions=[(0, 0), (0, 2), (0, 4), (1, 0), (1, 2), (1, 4)],
        colorbar_groups={
            "memory": (0, 1),
            "lec": (0, 5),
            "mec": (1, 3),
            "hpc": (1, 5),
        },
    )
    return fig


def _validate_freq(trace: TraceTree, freq_idx: int) -> int:
    """Validate frequency index against LEC and MEC paths."""
    trace_access.validate_freq_idx(trace, "state/lec/cells", freq_idx)
    trace_access.validate_freq_idx(trace, "output/inference/g_inf", freq_idx)
    trace_access.validate_freq_idx(trace, "output/inference/p_inf", freq_idx)
    return freq_idx


def _select_env(values: np.ndarray, env_idx: int) -> np.ndarray:
    """Select a single environment slice from a (T, B, ...) array."""
    if values.size == 0:
        return np.array([])
    if values.ndim < 2:
        return values
    return values[:, env_idx, ...]


def _safe_2d(values: np.ndarray) -> np.ndarray:
    """Ensure a 2D array for heatmap rendering."""
    if values.size == 0:
        return np.zeros((1, 1))
    if values.ndim == 1:
        return values[:, None]
    if values.ndim >= 2:
        return values
    return np.zeros((1, 1))


def _safe_1d(values: np.ndarray, *, default: float) -> np.ndarray:
    """Ensure a 1D array for per-location mapping."""
    if values.size == 0:
        return np.array([default])
    if values.ndim == 1:
        return values
    return values.reshape(values.shape[0], -1)[:, 0]


def _extract_hier_memory(trace: TraceTree, env_idx: int) -> np.ndarray:
    """Extract the last hierarchical memory matrix for an environment."""
    try:
        memory = trace_access.get_hpc_memory(trace, 0)
    except (IndexError, ValueError):
        return np.zeros((1, 1))
    if memory.size == 0:
        return np.zeros((1, 1))
    if memory.ndim == 4:
        return memory[-1, env_idx]
    if memory.ndim == 3:
        return memory[-1]
    return np.zeros((1, 1))
