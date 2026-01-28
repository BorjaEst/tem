"""Data validation helpers for figure primitives."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np


def as_1d_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return a 1D NumPy array.

    Args:
            values: Input values.

    Returns:
            1D NumPy array.

    Raises:
            ValueError: If the input cannot be coerced into 1D.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("Expected a 1D array")
    return array


def as_2d_array(values: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Return a 2D NumPy array.

    Args:
            values: Input values.

    Returns:
            2D NumPy array.

    Raises:
            ValueError: If the input cannot be coerced into 2D.
    """
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Expected a 2D array")
    return array


def validate_xy(x: Sequence[float], y: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Validate paired x/y arrays.

    Args:
            x: X values.
            y: Y values.

    Returns:
            Tuple of 1D NumPy arrays.

    Raises:
            ValueError: If lengths or dimensions do not match.
    """
    x_arr = as_1d_array(x)
    y_arr = as_1d_array(y)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length")
    return x_arr, y_arr


def validate_heatmap_data(values: Iterable[Iterable[float]] | np.ndarray) -> np.ndarray:
    """Validate heatmap input data.

    Args:
            values: 2D array-like heatmap values.

    Returns:
            2D NumPy array.
    """
    return as_2d_array(values)
