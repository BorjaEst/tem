"""Matrix generation utilities for torch_tem package."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from scipy.special import comb
from torch import Tensor


def create_W_repeat(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Tensor]:
    """Create repeat matrices for outer product computation.

    Matrix for repeating abstract location g to do outer product with sensory
    information x using elementwise product after matrix multiplication.

    Args:
        n_g_subsampled: Subsampled abstract location dimensions per frequency
        n_x_f: Sensory dimensions per frequency

    Returns:
        List of repeat matrices, one per frequency module
    """
    return [torch.tensor(np.kron(np.eye(g), np.ones((1, x))), dtype=torch.float) for g, x in zip(n_g_subsampled, n_x_f)]


def create_W_tile(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Tensor]:
    """Create tile matrices for outer product computation.

    Matrix for tiling sensory observation x to do outer product with abstract
    location using elementwise product after matrix multiplication.

    Args:
        n_g_subsampled: Subsampled abstract location dimensions per frequency
        n_x_f: Sensory dimensions per frequency

    Returns:
        List of tile matrices, one per frequency module
    """
    return [torch.tensor(np.kron(np.ones((1, g)), np.eye(x)), dtype=torch.float) for g, x in zip(n_g_subsampled, n_x_f)]


def create_g_downsample(n_g: List[int], n_g_subsampled: List[int]) -> List[Tensor]:
    """Create downsampling matrices for abstract location.

    Downsampling matrix to go from grid cells to compressed grid cells for
    indexing memories by simply taking only the first n_g_subsampled grid cells.

    Args:
        n_g: Full abstract location dimensions per frequency
        n_g_subsampled: Subsampled abstract location dimensions per frequency

    Returns:
        List of downsampling matrices, one per frequency module
    """
    return [torch.cat([torch.eye(dim_out, dtype=torch.float), torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)]) for dim_in, dim_out in zip(n_g, n_g_subsampled)]


def create_two_hot_table(n_x: int, n_x_c: int) -> List[Tensor]:
    """Create two-hot encoding lookup table.

    Table for converting one-hot to two-hot compressed representation.
    Generates all possible 2-hot codes up to the number of observations.

    Args:
        n_x: Number of possible observations
        n_x_c: Compressed sensory dimension

    Returns:
        List of two-hot code tensors, one per possible observation
    """
    # Start with first code: [0, 0, ..., 0, 1, 1]
    two_hot_table = [[0] * (n_x_c - 2) + [1] * 2]

    # Generate remaining codes up to min(C(n_x_c, 2), n_x)
    max_codes = min(int(comb(n_x_c, 2)), n_x)

    for i in range(1, max_codes):
        # Copy previous code
        code = two_hot_table[-1].copy()

        # Find latest occurrence of [0, 1] in that code
        swap = [index for index in range(len(code) - 1, -1, -1) if code[index : index + 2] == [0, 1]][0]

        # Swap those to get new code
        code[swap : swap + 2] = [1, 0]

        # If the first one was swapped: value after swapped pair is 1
        if swap + 2 < len(code) and code[swap + 2] == 1:
            # Move the second 1 all the way back - reverse everything after the swapped pair
            code[swap + 2 :] = code[: swap + 1 : -1]

        # Append new code to array
        two_hot_table.append(code)

    # Convert each code to column vector pytorch tensor
    return [torch.tensor(code, dtype=torch.float) for code in two_hot_table]


def detect_grid_structure(adj: np.ndarray, n_locs: int) -> Optional[Tuple[int, int]]:
    """Detect if adjacency matrix represents a grid and return (width, height).

    Analyzes the graph structure to determine if it matches a rectangular grid
    topology where each node connects to its 4-neighbors (up, down, left, right).

    Args:
        adj: Adjacency matrix as numpy array
        n_locs: Number of locations (nodes) in the graph

    Returns:
        Tuple of (width, height) if grid detected, None otherwise

    Example:
        >>> adj = np.array([[0, 1, 1, 0],
        ...                 [1, 0, 0, 1],
        ...                 [1, 0, 0, 1],
        ...                 [0, 1, 1, 0]])
        >>> detect_grid_structure(adj, 4)
        (2, 2)
    """
    # Try common grid dimensions
    for width in range(2, int(np.sqrt(n_locs)) + 2):
        if n_locs % width == 0:
            height = n_locs // width

            # Check if adjacency matches grid pattern
            is_grid = True
            for loc_id in range(n_locs):
                i, j = loc_id // width, loc_id % width

                # Count expected neighbors
                expected_neighbors = []
                if i > 0:
                    expected_neighbors.append((i - 1) * width + j)  # up
                if i < height - 1:
                    expected_neighbors.append((i + 1) * width + j)  # down
                if j > 0:
                    expected_neighbors.append(i * width + (j - 1))  # left
                if j < width - 1:
                    expected_neighbors.append(i * width + (j + 1))  # right

                # Check actual neighbors match
                actual_neighbors = [k for k in range(n_locs) if adj[loc_id, k] > 0]
                if set(actual_neighbors) != set(expected_neighbors):
                    is_grid = False
                    break

            if is_grid:
                return (width, height)

    return None
