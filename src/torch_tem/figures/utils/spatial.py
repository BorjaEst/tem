"""Spatial utilities for figure preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class GridIndex:
    """Grid index mapping for environment coordinates."""

    shape: Tuple[int, int]
    index: dict[int, Tuple[int, int]]
    xs: np.ndarray
    ys: np.ndarray


def get_location_coord(location: Mapping[str, float], keys: Sequence[str]) -> float:
    """Return a coordinate from a location dict using fallback keys.

    Args:
        location: Location mapping.
        keys: Coordinate keys to check in order.

    Returns:
        Coordinate value.

    Raises:
        KeyError: If no matching keys are found.
    """
    for key in keys:
        if key in location:
            return float(location[key])
    raise KeyError(f"Missing coordinate keys {tuple(keys)} in location {location}")


def world_xy_from_locations(
    locations: Sequence[Mapping[str, float]],
    *,
    x_keys: Sequence[str] = ("o", "x"),
    y_keys: Sequence[str] = ("y",),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates for all locations.

    Args:
        locations: Sequence of location mappings.
        x_keys: Keys to search for the x coordinate.
        y_keys: Keys to search for the y coordinate.

    Returns:
        Tuple of x and y coordinate arrays.
    """
    xs = np.array([get_location_coord(loc, x_keys) for loc in locations], dtype=float)
    ys = np.array([get_location_coord(loc, y_keys) for loc in locations], dtype=float)
    return xs, ys


def path_xy_from_location_ids(
    locations: Sequence[Mapping[str, float]],
    location_ids: Iterable[int],
    *,
    x_keys: Sequence[str] = ("o", "x"),
    y_keys: Sequence[str] = ("y",),
) -> Tuple[np.ndarray, np.ndarray]:
    """Return path coordinates for a sequence of location ids.

    Args:
        locations: Sequence of location mappings.
        location_ids: Location id sequence.
        x_keys: Keys to search for the x coordinate.
        y_keys: Keys to search for the y coordinate.

    Returns:
        Tuple of x and y coordinate arrays along the path.
    """
    coord_map = {
        int(loc["id"]): (
            get_location_coord(loc, x_keys),
            get_location_coord(loc, y_keys),
        )
        for loc in locations
        if "id" in loc
    }
    xs = []
    ys = []
    for loc_id in location_ids:
        loc_int = int(loc_id)
        if loc_int not in coord_map:
            continue
        x, y = coord_map[loc_int]
        xs.append(x)
        ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def build_grid_index(
    locations: Sequence[Mapping[str, float]],
    *,
    x_keys: Sequence[str] = ("o", "x"),
    y_keys: Sequence[str] = ("y",),
) -> GridIndex:
    """Build a grid index mapping from environment locations.

    Args:
        locations: Sequence of location mappings.
        x_keys: Keys to search for the x coordinate.
        y_keys: Keys to search for the y coordinate.

    Returns:
        GridIndex mapping ids to grid coordinates.
    """
    xs, ys = world_xy_from_locations(locations, x_keys=x_keys, y_keys=y_keys)
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    index: dict[int, Tuple[int, int]] = {}
    for loc, x, y in zip(locations, xs, ys):
        if "id" not in loc:
            continue
        row = int(np.where(unique_y == y)[0][0])
        col = int(np.where(unique_x == x)[0][0])
        index[int(loc["id"])] = (row, col)
    return GridIndex(
        shape=(unique_y.size, unique_x.size),
        index=index,
        xs=unique_x,
        ys=unique_y,
    )


def infer_location_count(grid_index: GridIndex) -> int:
    """Infer the number of locations from a grid index.

    Args:
        grid_index: GridIndex mapping location ids.

    Returns:
        Inferred location count.
    """
    return max(grid_index.index.keys(), default=-1) + 1


def grid_from_location_values(
    values: np.ndarray,
    grid_index: GridIndex,
) -> np.ndarray:
    """Map per-location values into a 2D grid.

    Args:
        values: 1D array of values indexed by location id.
        grid_index: GridIndex mapping ids to grid coordinates.

    Returns:
        2D grid with values placed at their coordinates.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected 1D values for grid_from_location_values")
    grid = np.full(grid_index.shape, np.nan, dtype=float)
    for loc_id, (row, col) in grid_index.index.items():
        if loc_id < 0 or loc_id >= array.size:
            continue
        grid[row, col] = array[loc_id]
    return grid


def aggregate_by_location(
    location_ids: Sequence[int],
    values: np.ndarray,
    *,
    n_locations: int,
    mode: str = "mean",
) -> np.ndarray:
    """Aggregate values by location id.

    Args:
        location_ids: Location ids over time.
        values: Per-step values with leading time dimension.
        n_locations: Total number of locations.
        mode: Aggregation mode ("mean" or "last").

    Returns:
        Array aggregated per location id.
    """
    if n_locations <= 0:
        return np.zeros((0,), dtype=float)
    array = np.asarray(values, dtype=float)
    if array.ndim < 1:
        raise ValueError("Expected values with leading time dimension")
    max_steps = min(len(location_ids), array.shape[0])
    trailing_shape = array.shape[1:]
    if mode == "mean":
        sums = np.zeros((n_locations, *trailing_shape), dtype=float)
        counts = np.zeros(n_locations, dtype=float)
        for step, loc_id in enumerate(location_ids[:max_steps]):
            loc_int = int(loc_id)
            if loc_int < 0 or loc_int >= n_locations:
                continue
            sums[loc_int] += array[step]
            counts[loc_int] += 1
        counts = np.where(counts == 0, np.nan, counts)
        expand_shape = (n_locations,) + (1,) * len(trailing_shape)
        return sums / counts.reshape(expand_shape)
    if mode == "last":
        output = np.full((n_locations, *trailing_shape), np.nan, dtype=float)
        for step, loc_id in enumerate(location_ids[:max_steps]):
            loc_int = int(loc_id)
            if loc_int < 0 or loc_int >= n_locations:
                continue
            output[loc_int] = array[step]
        return output
    raise ValueError(f"Unsupported aggregation mode: {mode}")
