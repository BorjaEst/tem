from __future__ import annotations

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
