"""Reusable axis-level plotting primitives for TEM figures."""

from torch_tem.figures.plots.autocorr import plot_autocorr2d
from torch_tem.figures.plots.insets import coverage_over_time
from torch_tem.figures.plots.map import action_patch, configure_environment_axes, plot_actions, plot_map, plot_walk
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory

__all__ = [
    "action_patch",
    "configure_environment_axes",
    "coverage_over_time",
    "plot_actions",
    "plot_autocorr2d",
    "plot_map",
    "plot_time_colored_trajectory",
    "plot_walk",
]
