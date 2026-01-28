"""Grid cell diagnostics for generative MEC states (g_gen)."""

from __future__ import annotations

from typing import List, Sequence

import matplotlib.figure as mpl_figure
import numpy as np
from matplotlib.axes import Axes

from torch_tem.diagnostics import trace_access
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures import compose_gridspec
from torch_tem.figures.figures.styles import default_primitive_style, get_theme
from torch_tem.figures.plots import heatmap, line, trajectory
from torch_tem.figures.registry import FigureContext
from torch_tem.figures.utils.autocorr import RadialStats, autocorr2d, radial_stats_from_location_values
from torch_tem.figures.utils.spatial import aggregate_by_location, build_grid_index, grid_from_location_values, infer_location_count, path_xy_from_location_ids


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot grid cell diagnostics for g_gen at a selected frequency.

    Args:
        trace: TraceTree with rollout data.
        ctx: Figure context with env/frequency selection.

    Returns:
        Matplotlib Figure instance.
    """
    env_idx = trace_access.validate_env_idx(trace, ctx.env_idx)
    freq_idx = trace_access.validate_freq_idx(
        trace,
        "output/generative/g_gen",
        ctx.freq_idx,
    )
    theme = get_theme("paper")
    line_style = default_primitive_style("line", theme)
    scatter_style = default_primitive_style("scatter", theme)

    location_ids = trace_access.get_location_ids_for_env(trace, env_idx)
    world = trace_access.get_world(trace, env_idx)
    grid_index = build_grid_index(world.locations)

    g_gen = trace_access.get_mec_g_gen(trace, freq_idx)
    if g_gen.size == 0:
        g_env = np.zeros((1, 1))
    elif g_gen.ndim == 3:
        g_env = g_gen[:, env_idx, :]
    else:
        g_env = g_gen

    exemplar_cells = _select_exemplar_cells(g_env, n_cells=3)
    n_locations = infer_location_count(grid_index)
    rate_by_location = _aggregate_by_location(location_ids, g_env, n_locations)

    rate_maps = [grid_from_location_values(rate_by_location[:, cell], grid_index) for cell in exemplar_cells]
    autocorr_maps = [autocorr2d(rate_map) for rate_map in rate_maps]

    radial_stats = radial_stats_from_location_values(rate_by_location, grid_index)

    def _panel_traj(ax: Axes, theme=theme) -> None:
        xs, ys = path_xy_from_location_ids(world.locations, location_ids)
        if xs.size == 0:
            xs = np.array([0.0])
            ys = np.array([0.0])
        times = np.arange(xs.size)
        trajectory(
            ax,
            x=xs,
            y=ys,
            c=times,
            cmap="viridis",
            scatter_style={"s": 12, "alpha": 0.9, **scatter_style},
            line_style={"color": "0.5", "linewidth": 0.8, "alpha": 0.5},
        )
        ax.set_title("Trajectory (time-colored)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    def _panel_rate(ax: Axes, values: np.ndarray, cell_idx: int) -> None:
        heatmap(ax, values=values, cmap="viridis", colorbar_group="rate_maps")
        ax.set_title(f"g_gen rate map (cell {cell_idx})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    def _panel_radial(ax: Axes, stats: RadialStats) -> None:
        line(ax, x=stats.radii, y=stats.mean, label="mean", style=line_style)
        ax.fill_between(
            stats.radii,
            stats.mean - stats.std,
            stats.mean + stats.std,
            color=line_style.get("color"),
            alpha=0.2,
        )
        ax.set_title("Radial autocorr (mean ± std)")
        ax.set_xlabel("radius")
        ax.set_ylabel("correlation")

    def _panel_autocorr(ax: Axes, values: np.ndarray, cell_idx: int) -> None:
        heatmap(ax, values=values, cmap="viridis", colorbar_group="autocorr_maps")
        ax.set_title(f"2D autocorr (cell {cell_idx})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    panels = [
        _panel_traj,
        lambda ax, theme=theme: _panel_rate(ax, rate_maps[0], exemplar_cells[0]),
        lambda ax, theme=theme: _panel_rate(ax, rate_maps[1], exemplar_cells[1]),
        lambda ax, theme=theme: _panel_rate(ax, rate_maps[2], exemplar_cells[2]),
        lambda ax, theme=theme: _panel_radial(ax, radial_stats),
        lambda ax, theme=theme: _panel_autocorr(ax, autocorr_maps[0], exemplar_cells[0]),
        lambda ax, theme=theme: _panel_autocorr(ax, autocorr_maps[1], exemplar_cells[1]),
        lambda ax, theme=theme: _panel_autocorr(ax, autocorr_maps[2], exemplar_cells[2]),
    ]

    fig = compose_gridspec(
        panels=panels,
        layout=(2, 4),
        theme=theme,
        template="paper",
        legend="none",
        colorbar="grouped",
        share_color_norm=True,
        size=ctx.figsize,
        gridspec_layout={"nrows": 2, "ncols": 5, "width_ratios": [1, 1, 1, 1, 0.05], "wspace": 0.25, "hspace": 0.35},
        content_positions=[(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        colorbar_groups={"rate_maps": (0, 4), "autocorr_maps": (1, 4)},
    )
    return fig


def _select_exemplar_cells(values: np.ndarray, n_cells: int) -> List[int]:
    """Select exemplar cell indices from a (T, C) array."""
    if values.ndim != 2 or values.shape[1] == 0:
        return list(range(n_cells))
    max_cells = values.shape[1]
    return list(range(min(n_cells, max_cells)))


def _aggregate_by_location(location_ids: Sequence[int], values: np.ndarray, n_locations: int) -> np.ndarray:
    """Compute mean activation per location for all cells."""
    if values.ndim != 2:
        raise ValueError("Expected values with shape (T, C)")
    return aggregate_by_location(location_ids, values, n_locations=n_locations, mode="mean")
