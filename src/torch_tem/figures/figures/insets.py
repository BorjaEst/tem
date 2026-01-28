"""Figure-level inset helpers for composition and layout."""

from __future__ import annotations

from typing import List

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def add_coverage_inset(
    ax: Axes,
    location_ids: List[int],
    n_locations: int,
    *,
    width: str = "55%",
    height: str = "35%",
    loc: str = "lower left",
    borderpad: float = 1.0,
    line_color: str = "#333333",
    title: str = "Coverage",
) -> Axes | None:
    """Add a coverage-over-time inset to an axes.

    Args:
        ax: Parent axes to attach inset to.
        location_ids: Ordered list of visited location indices.
        n_locations: Total number of locations in the environment.
        width: Inset width (matplotlib inset_axes format).
        height: Inset height (matplotlib inset_axes format).
        loc: Inset location anchor.
        borderpad: Padding around the inset.
        line_color: Line color for the coverage curve.
        title: Title for the inset plot.

    Returns:
        The inset axes if created, otherwise None.
    """
    if not location_ids or n_locations == 0:
        return None
    coverage = coverage_over_time(location_ids, n_locations)
    inset = inset_axes(ax, width=width, height=height, loc=loc, borderpad=borderpad)
    inset.plot(coverage, color=line_color, linewidth=1.2)
    inset.set_ylim(0.0, 1.0)
    inset.set_xlim(0, max(len(coverage) - 1, 1))
    inset.set_title(title, fontsize=8)
    inset.set_xticks([])
    inset.set_yticks([0.0, 1.0])
    inset.set_yticklabels(["0", "1"], fontsize=7)
    return inset


def coverage_over_time(location_ids: List[int], n_locations: int) -> np.ndarray:
    """Compute coverage ratio over time for a sequence of locations.

    Args:
        location_ids: Ordered list of visited location indices.
        n_locations: Total number of locations.

    Returns:
        Array of coverage ratios over time.
    """
    visited = np.zeros(n_locations, dtype=bool)
    coverage = np.zeros(len(location_ids), dtype=float)
    for idx, loc_id in enumerate(location_ids):
        if 0 <= loc_id < n_locations:
            visited[loc_id] = True
        coverage[idx] = visited.sum() / max(n_locations, 1)
    return coverage
