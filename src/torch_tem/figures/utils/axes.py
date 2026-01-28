"""Axes formatting helpers for figures."""

from __future__ import annotations

from matplotlib.axes import Axes


def format_map_axes(
    ax: Axes,
    *,
    equal: bool = True,
    hide_ticks: bool = True,
) -> None:
    """Apply consistent map formatting for spatial panels.

    Args:
        ax: Matplotlib Axes target.
        equal: Whether to enforce equal aspect ratio.
        hide_ticks: Whether to hide axis ticks.
    """
    if equal:
        ax.set_aspect("equal", adjustable="box")
    if hide_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
