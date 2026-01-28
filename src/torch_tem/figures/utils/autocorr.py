"""Autocorrelation utilities for figure preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from torch_tem.figures.utils.spatial import GridIndex, grid_from_location_values


@dataclass(frozen=True)
class RadialStats:
    """Radial autocorrelation statistics."""

    radii: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def autocorr_1d(values: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute 1D autocorrelation up to max_lag.

    Args:
        values: 1D array.
        max_lag: Maximum lag.

    Returns:
        Autocorrelation values.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected 1D values for autocorr_1d")
    if max_lag <= 0:
        return np.array([1.0], dtype=float)
    corr = np.correlate(array, array, mode="full")
    mid = corr.size // 2
    corr = corr[mid : mid + max_lag + 1]
    if corr[0] != 0:
        corr = corr / corr[0]
    return corr


def autocorr2d(
    values: np.ndarray,
    *,
    normalize: bool = True,
    nan_fill: float = 0.0,
) -> np.ndarray:
    """Compute a 2D autocorrelation map.

    Args:
        values: 2D input array.
        normalize: Whether to normalize by the peak magnitude.
        nan_fill: Replacement value for NaNs.

    Returns:
        Autocorrelation map.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected 2D values for autocorr2d")
    array = np.nan_to_num(array, nan=nan_fill)
    array = array - np.mean(array)
    fft = np.fft.fft2(array)
    corr = np.fft.ifft2(fft * np.conj(fft)).real
    corr = np.fft.fftshift(corr)
    if normalize:
        peak = float(np.max(np.abs(corr)))
        if peak > 0:
            corr = corr / peak
    return corr


def radial_profile(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radial mean profile from a 2D array.

    Args:
        values: 2D input array.

    Returns:
        Tuple of radii and radial mean values.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected 2D values for radial_profile")
    rows, cols = array.shape
    center = (rows - 1) / 2.0, (cols - 1) / 2.0
    y, x = np.indices(array.shape)
    radii = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    max_radius = int(np.floor(min(rows, cols) / 2))
    radial = []
    for r in range(max_radius + 1):
        mask = (radii >= r - 0.5) & (radii < r + 0.5)
        if not np.any(mask):
            radial.append(np.nan)
        else:
            radial.append(float(np.nanmean(array[mask])))
    return np.arange(max_radius + 1), np.array(radial, dtype=float)


def radial_stats_from_location_values(
    rate_by_location: np.ndarray,
    grid_index: GridIndex,
) -> RadialStats:
    """Compute radial autocorr mean and std for all cells.

    Args:
        rate_by_location: Array with shape (L, C).
        grid_index: Grid index mapping location ids to grid coordinates.

    Returns:
        RadialStats with radii, mean, and std across cells.
    """
    if rate_by_location.ndim != 2:
        raise ValueError("Expected rate_by_location with shape (L, C)")
    n_cells = rate_by_location.shape[1]
    profiles = []
    radii = None
    for cell in range(n_cells):
        grid = grid_from_location_values(rate_by_location[:, cell], grid_index)
        corr = autocorr2d(grid)
        r, profile = radial_profile(corr)
        profiles.append(profile)
        if radii is None:
            radii = r
    if radii is None:
        radii = np.arange(1, dtype=float)
    stacked = np.stack(profiles, axis=0) if profiles else np.zeros((1, len(radii)), dtype=float)
    valid = np.isfinite(stacked)
    counts = valid.sum(axis=0)
    safe = np.where(valid, stacked, 0.0)
    mean = np.divide(
        safe.sum(axis=0),
        counts,
        out=np.zeros(stacked.shape[1], dtype=float),
        where=counts > 0,
    )
    diffs = np.where(valid, stacked - mean, 0.0)
    var = np.divide(
        (diffs**2).sum(axis=0),
        counts,
        out=np.zeros(stacked.shape[1], dtype=float),
        where=counts > 0,
    )
    std = np.sqrt(var)
    return RadialStats(radii=radii, mean=mean, std=std)
