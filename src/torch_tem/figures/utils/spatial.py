"""Spatial utilities for figure modules."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def aggregate_rate_map(
    activity_env: np.ndarray,
    location_ids: list[int],
    n_locations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate per-step activity into per-location means and occupancy.

    Args:
        activity_env: Array of shape (T,) or (T, C) for one environment.
        location_ids: Per-step visited location IDs (T,).
        n_locations: Total number of locations.

    Returns:
        Tuple of (rate_map, occupancy), where rate_map has shape (n_locations, C)
        and occupancy is visit counts per location.
    """
    if activity_env.ndim == 1:
        activity_env = activity_env[:, None]

    rate_map = np.full((n_locations, activity_env.shape[1]), np.nan, dtype=np.float32)
    occupancy = np.zeros(n_locations, dtype=int)
    loc_ids = np.asarray(location_ids, dtype=int)

    for loc_id in range(n_locations):
        mask = loc_ids == loc_id
        if mask.any():
            occupancy[loc_id] = int(mask.sum())
            rate_map[loc_id] = activity_env[mask].mean(axis=0)

    return rate_map, occupancy


def robust_min_max(values: np.ndarray, lower: float = 5.0, upper: float = 95.0) -> tuple[float, float]:
    """Compute robust min/max using percentile bounds on finite values."""
    flat = values[np.isfinite(values)]
    if flat.size == 0:
        return 0.0, 1.0
    min_val = float(np.percentile(flat, lower))
    max_val = float(np.percentile(flat, upper))
    if min_val == max_val:
        min_val = float(np.min(flat))
        max_val = float(np.max(flat)) if np.max(flat) != np.min(flat) else min_val + 1.0
    return min_val, max_val


def select_feature_by_spatial_variance(rate_map: np.ndarray) -> int:
    """Select the most spatially varying feature index.

    Args:
        rate_map: Array of shape (n_locations, n_features).

    Returns:
        Index of the feature with highest spatial variance.
    """
    if rate_map.size == 0:
        return 0
    variances = np.nanvar(rate_map, axis=0)
    if not np.isfinite(variances).any():
        return 0
    return int(np.nanargmax(variances))


def select_top_k_by_spatial_variance(
    rate_map: np.ndarray,
    occupancy: np.ndarray | None,
    k: int,
    min_coverage: float = 0.1,
) -> np.ndarray:
    """Select top-k feature indices by spatial variance with coverage filtering.

    Args:
        rate_map: Array of shape (n_locations, n_features).
        occupancy: Optional per-location visit counts.
        k: Number of features to select.
        min_coverage: Minimum fraction of locations required per feature.

    Returns:
        Array of selected feature indices (length <= k).
    """
    if rate_map.size == 0 or k <= 0:
        return np.array([], dtype=int)

    n_locations, n_features = rate_map.shape
    min_locations = max(1, int(np.ceil(min_coverage * n_locations)))

    if occupancy is not None and occupancy.size:
        visited_mask = occupancy > 0
    else:
        visited_mask = np.isfinite(rate_map).any(axis=1)

    scores = np.full(n_features, -np.inf, dtype=float)
    for idx in range(n_features):
        values = rate_map[:, idx]
        valid = np.isfinite(values) & visited_mask
        if valid.sum() < min_locations:
            continue
        scores[idx] = float(np.nanvar(values[valid]))

    if not np.isfinite(scores).any():
        return np.array([], dtype=int)

    order = np.argsort(-scores)
    k = min(k, n_features)
    return order[:k]


def clip_unit_interval(values: np.ndarray) -> np.ndarray:
    """Clip values to the [0, 1] interval for display."""
    return np.clip(values, 0.0, 1.0)


def radial_autocorr(values: np.ndarray, world, n_bins: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Compute radial spatial autocorrelogram curve for a single feature.

    Args:
        values: Per-location values with NaN for unvisited locations.
        world: Environment world with locations containing "o" and "y".
        n_bins: Number of distance bins.

    Returns:
        Tuple of (bin_centers, curve). Empty arrays if unavailable.
    """
    coords = _location_coords(world)
    if coords.size == 0:
        return np.array([]), np.array([])
    distances = _pairwise_distances(coords)
    bins, centers = _distance_bins(distances, n_bins=n_bins)
    curve = _autocorr_curve(values, distances, bins)
    return centers, curve


def summarize_radial_autocorr(
    rate_map: np.ndarray,
    world: object,
    *,
    n_bins: int = 12,
    min_coverage: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Summarize radial autocorrelograms across features.

    Args:
        rate_map: Array of shape (n_locations, n_features).
        world: Environment world with location coordinates.
        n_bins: Number of distance bins.
        min_coverage: Minimum fraction of locations required per feature.

    Returns:
        Tuple of (centers, median, q25, q75). Empty arrays if unavailable.
    """
    if rate_map.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    n_locations, n_features = rate_map.shape
    min_locations = max(1, int(np.ceil(min_coverage * n_locations)))

    curves: list[np.ndarray] = []
    centers: np.ndarray | None = None

    for idx in range(n_features):
        values = rate_map[:, idx]
        if np.isfinite(values).sum() < min_locations:
            continue
        centers, curve = radial_autocorr(values, world, n_bins=n_bins)
        if curve.size:
            curves.append(curve)

    if not curves or centers is None or centers.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    stacked = np.vstack(curves)
    median = np.nanmedian(stacked, axis=0)
    q25 = np.nanpercentile(stacked, 25, axis=0)
    q75 = np.nanpercentile(stacked, 75, axis=0)
    return centers, median, q25, q75


def autocorr_2d(values: np.ndarray, world) -> np.ndarray:
    """Compute a 2D spatial autocorrelogram for a single feature.

    Args:
        values: Per-location values with NaN for unvisited locations.
        world: Environment world with locations containing "o" and "y".

    Returns:
        2D autocorrelogram array with shape (2 * ny - 1, 2 * nx - 1).
        Returns an empty array if coordinates are unavailable.
    """
    grid = _values_to_grid(values, world)
    if grid.size == 0:
        return np.array([])

    finite = np.isfinite(grid)
    if finite.sum() < 2:
        ny, nx = grid.shape
        return np.zeros((2 * ny - 1, 2 * nx - 1), dtype=float)

    mean = np.mean(grid[finite])
    std = np.std(grid[finite])
    if not np.isfinite(std) or std == 0.0:
        ny, nx = grid.shape
        return np.zeros((2 * ny - 1, 2 * nx - 1), dtype=float)

    z_grid = (grid - mean) / std
    return _autocorr_2d_from_zgrid(z_grid)


def infer_grid_spacing(world) -> tuple[float | None, float | None]:
    """Infer grid spacing from world location coordinates.

    Returns:
        Tuple of (dx, dy) in world units. Returns (None, None) if spacing
        cannot be inferred.
    """
    coords = _location_coords(world)
    if coords.size == 0:
        return None, None

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    dx = _median_positive_diff(xs)
    dy = _median_positive_diff(ys)
    return dx, dy


def infer_distance_scale(world) -> float | None:
    """Infer a representative distance scale for converting to cell units."""
    dx, dy = infer_grid_spacing(world)
    spacings = [val for val in (dx, dy) if val is not None and val > 0]
    if not spacings:
        return None
    return float(np.mean(spacings))


def autocorr_extent(world, *, units: str = "world") -> tuple[float, float, float, float] | None:
    """Compute a 2D autocorr extent for imshow.

    Args:
        world: Environment world with location coordinates.
        units: "world" for world units or "cells" for grid-cell steps.

    Returns:
        Extent tuple (xmin, xmax, ymin, ymax) or None if unavailable.
    """
    coords = _location_coords(world)
    if coords.size == 0:
        return None

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    if xs.size < 2 or ys.size < 2:
        return None

    nx = xs.size
    ny = ys.size

    if units == "cells":
        x_max = float(nx - 1)
        y_max = float(ny - 1)
    else:
        dx = _median_positive_diff(xs)
        dy = _median_positive_diff(ys)
        if dx is None or dy is None:
            return None
        x_max = float((nx - 1) * dx)
        y_max = float((ny - 1) * dy)

    return (-x_max, x_max, -y_max, y_max)


def _location_coords(world) -> np.ndarray:
    return np.asarray([[float(loc["o"]), float(loc["y"])] for loc in world.locations], dtype=float)


def _values_to_grid(values: np.ndarray, world) -> np.ndarray:
    coords = _location_coords(world)
    if coords.size == 0:
        return np.array([])

    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    x_index = {float(x): idx for idx, x in enumerate(xs)}
    y_index = {float(y): idx for idx, y in enumerate(ys)}

    grid = np.full((len(ys), len(xs)), np.nan, dtype=float)
    for loc_idx, (x_val, y_val) in enumerate(coords):
        col = x_index.get(float(x_val))
        row = y_index.get(float(y_val))
        if row is None or col is None:
            continue
        grid[row, col] = values[loc_idx]

    return grid


def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _median_positive_diff(values: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    diffs = np.diff(np.sort(values))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _autocorr_2d_from_zgrid(z_grid: np.ndarray) -> np.ndarray:
    ny, nx = z_grid.shape
    out = np.zeros((2 * ny - 1, 2 * nx - 1), dtype=float)

    for dy in range(-(ny - 1), ny):
        y_slice_a, y_slice_b = _overlap_slices(ny, dy)
        for dx in range(-(nx - 1), nx):
            x_slice_a, x_slice_b = _overlap_slices(nx, dx)
            a = z_grid[y_slice_a, x_slice_a]
            b = z_grid[y_slice_b, x_slice_b]
            valid = np.isfinite(a) & np.isfinite(b)
            if valid.sum() < 2:
                out[dy + ny - 1, dx + nx - 1] = 0.0
            else:
                out[dy + ny - 1, dx + nx - 1] = float(np.mean(a[valid] * b[valid]))

    return out


def _overlap_slices(length: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)


def _distance_bins(distances: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    max_dist = float(np.nanmax(distances)) if distances.size else 0.0
    bins = np.linspace(0.0, max_dist, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    return bins, centers


def _autocorr_curve(values: np.ndarray, distances: np.ndarray, bins: np.ndarray) -> np.ndarray:
    n_bins = len(bins) - 1
    out = np.full(n_bins, np.nan, dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros(n_bins, dtype=float)

    std = np.std(values[finite])
    if not np.isfinite(std) or std == 0.0:
        return np.zeros(n_bins, dtype=float)
    z_vals = (values[finite] - np.mean(values[finite])) / std
    z_full = np.full(values.shape[0], np.nan)
    z_full[finite] = z_vals

    upper = np.triu_indices(values.shape[0], k=1)
    dists = distances[upper]
    vals_i = z_full[upper[0]]
    vals_j = z_full[upper[1]]
    valid = np.isfinite(vals_i) & np.isfinite(vals_j)
    if not valid.any():
        return np.zeros(n_bins, dtype=float)

    dists = dists[valid]
    products = vals_i[valid] * vals_j[valid]
    for bin_idx in range(n_bins):
        if bin_idx == n_bins - 1:
            mask = (dists >= bins[bin_idx]) & (dists <= bins[bin_idx + 1])
        else:
            mask = (dists >= bins[bin_idx]) & (dists < bins[bin_idx + 1])
        if mask.any():
            out[bin_idx] = float(np.mean(products[mask]))
        else:
            out[bin_idx] = 0.0
    return out
