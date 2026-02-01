from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def aggregate_rate_map(
    cells: np.ndarray,
    location_ids: Iterable[int],
    n_locations: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate cell activity into a per-location rate map.

    Args:
            cells: Cell activity array with shape (T, C) or (T, B, C).
            location_ids: Sequence of visited location indices per timestep.
            n_locations: Total number of locations in the environment.

    Returns:
            Tuple of (rate_map, counts) where rate_map is (C, n_locations).
    """
    cell_array = np.asarray(cells, dtype=float)
    if cell_array.ndim == 3:
        cell_array = np.nanmean(cell_array, axis=1)
    if cell_array.ndim == 1:
        cell_array = cell_array[:, None]

    loc_ids = np.asarray(list(location_ids), dtype=int)
    if cell_array.shape[0] == 0 or loc_ids.size == 0:
        rate_map = np.full((cell_array.shape[-1], n_locations), np.nan)
        counts = np.zeros(n_locations, dtype=int)
        return rate_map, counts

    n_steps = min(cell_array.shape[0], loc_ids.shape[0])
    cell_array = cell_array[:n_steps]
    loc_ids = loc_ids[:n_steps]

    n_cells = cell_array.shape[1]
    rate_map = np.full((n_cells, n_locations), np.nan, dtype=float)
    counts = np.zeros(n_locations, dtype=int)

    for loc in range(n_locations):
        mask = loc_ids == loc
        counts[loc] = int(mask.sum())
        if counts[loc] == 0:
            continue
        rate_map[:, loc] = np.nanmean(cell_array[mask], axis=0)

    return rate_map, counts


def select_mosaic_grid(
    n_cells: int,
    slot_width: float,
    slot_height: float,
    *,
    min_cols: int = 2,
    max_cols: int = 36,
    wspace: float = 0.0,
    hspace: float = 0.0,
) -> tuple[int, int]:
    """Select a grid shape that minimizes unused space in the slot.

    Args:
        n_cells: Number of cells to plot.
        slot_width: Slot width in inches.
        slot_height: Slot height in inches.
        min_cols: Minimum number of columns.
        max_cols: Maximum number of columns.

    Returns:
        Tuple of (nrows, ncols) for the mosaic grid.
    """
    if n_cells <= 0:
        return 1, 1

    max_cols = max(1, min(int(max_cols), n_cells))
    min_cols = max(1, min(int(min_cols), max_cols))

    if slot_width <= 0.0 or slot_height <= 0.0:
        ncols = max(
            min_cols,
            min(max_cols, int(math.ceil(math.sqrt(n_cells)))),
        )
        nrows = int(math.ceil(n_cells / ncols))
        return nrows, ncols

    aspect = slot_width / slot_height
    best_score: float | None = None
    best_shape = (1, 1)
    for ncols in range(min_cols, max_cols + 1):
        nrows = int(math.ceil(n_cells / ncols))
        denom_cols = ncols + max(ncols - 1, 0) * wspace
        denom_rows = nrows + max(nrows - 1, 0) * hspace
        if denom_cols <= 0 or denom_rows <= 0:
            continue
        cell_aspect = aspect * (denom_rows / denom_cols)
        if cell_aspect > 0:
            square_penalty = abs(math.log(cell_aspect))
        else:
            square_penalty = 0.0
        waste_penalty = (nrows * ncols - n_cells) / n_cells
        score = square_penalty + 0.25 * waste_penalty
        if best_score is None or score < best_score:
            best_score = score
            best_shape = (nrows, ncols)

    return best_shape
