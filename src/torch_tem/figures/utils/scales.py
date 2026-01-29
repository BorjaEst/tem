"""Shared scaling utilities for figure normalization."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from matplotlib.colors import Normalize

from torch_tem.figures.utils import aggregate_rate_map


def build_shared_norm(arrays: Iterable[np.ndarray]) -> Normalize:
    """Create a shared normalization for multiple arrays.

    Args:
        arrays: Iterable of arrays to combine when computing min/max.

    Returns:
        A Normalize instance covering the finite range of the arrays.
    """
    array_list = list(arrays)
    values = np.concatenate([arr.ravel() for arr in array_list if arr.size]) if array_list else np.array([])
    finite_mask = np.isfinite(values)
    if values.size == 0 or not finite_mask.any():
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return Normalize(vmin=vmin, vmax=vmax)


def build_shared_minmax(arrays: Iterable[np.ndarray]) -> tuple[float, float]:
    """Compute a shared min/max range for multiple arrays.

    Args:
        arrays: Iterable of arrays to combine when computing min/max.

    Returns:
        Tuple of (vmin, vmax) for the finite values.
    """
    array_list = list(arrays)
    values = np.concatenate([arr.ravel() for arr in array_list if arr.size]) if array_list else np.array([])
    finite_mask = np.isfinite(values)
    if values.size == 0 or not finite_mask.any():
        return 0.0, 1.0
    vmin = float(values[finite_mask].min())
    vmax = float(values[finite_mask].max())
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def build_shared_map_range(
    mec_cells: np.ndarray,
    hpc_cells: np.ndarray,
    location_ids: np.ndarray,
    n_locations: int,
    cell_idx: int,
) -> tuple[float, float]:
    """Compute a shared min/max range for MEC and HPC rate maps.

    Args:
        mec_cells: MEC activation array.
        hpc_cells: HPC activation array.
        location_ids: Location ids aligned with the time dimension.
        n_locations: Number of locations in the environment.
        cell_idx: Index of the cell to compare.

    Returns:
        Tuple of (vmin, vmax) for the selected cell across both maps.
    """
    mec_rate, _ = aggregate_rate_map(mec_cells, location_ids, n_locations)
    hpc_rate, _ = aggregate_rate_map(hpc_cells, location_ids, n_locations)

    mec_values = mec_rate[cell_idx] if mec_rate.size else np.array([])
    hpc_values = hpc_rate[cell_idx] if hpc_rate.size else np.array([])
    return build_shared_minmax([mec_values, hpc_values])
