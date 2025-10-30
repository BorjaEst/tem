"""Utility functions for TEM parameter calculations and matrix operations."""

from typing import List

import numpy as np
from scipy.special import comb
from torch import Tensor, cat, eye, kron, ones, tensor, zeros


def create_hierarchical_mask(
    n_p_list: List[int],
    n_f: int,
    n_f_g: int,
    f_initial: List[float],
) -> Tensor:
    """Create hierarchical Hebbian memory connection mask from low to high frequency.

    Args:
        n_p_list: Number of neurons for hippocampal grounded location per frequency
        n_f: Total number of frequency modules
        n_f_g: Number of grid cell frequency modules
        f_initial: Initial frequencies per module

    Returns:
        Connection mask tensor where M_ij represents connection FROM i TO j
    """
    mask = zeros((sum(n_p_list), sum(n_p_list)))
    n_p_cumsum = np.cumsum(np.concatenate(([0], n_p_list)))

    for f_from in range(n_f):
        for f_to in range(n_f):
            if f_from > n_f_g or f_to > n_f_g:
                # OVC module connections
                if f_from > n_f_g and f_to > n_f_g:
                    # Both OVC: hierarchical (low to high frequency)
                    if f_initial[f_from] <= f_initial[f_to]:
                        mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
                else:
                    # Mixed OVC and grid: allow all connections
                    mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
            else:
                # Grid cell connections: hierarchical (low to high frequency)
                if f_initial[f_from] <= f_initial[f_to]:
                    mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0

    return mask


def create_retrieval_masks(
    n_p_list: List[int],
    n_attractor: int,
    max_iters_per_freq: List[int],
) -> List[Tensor]:
    """Create hierarchical memory retrieval masks for early-stopping low-frequency modules.

    Args:
        n_p_list: Number of neurons for hippocampal grounded location per frequency
        n_attractor: Number of attractor dynamics iterations
        max_iters_per_freq: Maximum iterations per frequency module

    Returns:
        List of masks for each retrieval iteration
    """
    n_p_cumsum = np.cumsum(np.concatenate(([0], n_p_list)))
    masks = [zeros(sum(n_p_list)) for _ in range(n_attractor)]

    for f, max_i in enumerate(max_iters_per_freq):
        for i in range(max_i):
            masks[i][n_p_cumsum[f] : n_p_cumsum[f + 1]] = 1.0

    return masks


def create_hierarchical_connections(
    n_f: int,
    n_f_g: int,
    n_f_ovc: int,
    f_initial: List[float],
) -> List[List[bool]]:
    """Create hierarchical connection matrix for abstract location module transitions.

    Args:
        n_f: Total number of frequency modules
        n_f_g: Number of grid cell frequency modules
        n_f_ovc: Number of OVC frequency modules
        f_initial: Initial frequencies per module

    Returns:
        Connection matrix where connections[i][j] indicates if module j connects to i
    """
    connections = []

    # Grid cell connections (hierarchical: low to high frequency)
    for f_to in range(n_f_g):
        row = [f_initial[f_from] <= f_initial[f_to] for f_from in range(n_f_g)] + [False for _ in range(n_f_ovc)]
        connections.append(row)

    # OVC connections (hierarchical between OVC modules only)
    for f_to in range(n_f_g, n_f):
        row = [False for _ in range(n_f_g)] + [f_initial[f_from] <= f_initial[f_to] for f_from in range(n_f_g, n_f)]
        connections.append(row)

    return connections


def create_outer_product_repeat_matrix(n_rows: int, n_cols: int) -> Tensor:
    """Create matrix for repeating rows in outer product calculation.

    Args:
        n_rows: Number of rows (e.g., abstract location dimension)
        n_cols: Number of columns (e.g., sensory observation dimension)

    Returns:
        Repeat matrix for outer product flattening via matrix multiplication
    """
    return kron(eye(n_rows), ones((1, n_cols))).float()


def create_outer_product_tile_matrix(n_rows: int, n_cols: int) -> Tensor:
    """Create matrix for tiling columns in outer product calculation.

    Args:
        n_rows: Number of rows (e.g., abstract location dimension)
        n_cols: Number of columns (e.g., sensory observation dimension)

    Returns:
        Tile matrix for outer product flattening via matrix multiplication
    """
    return kron(ones((1, n_rows)), eye(n_cols)).float()


def generate_two_hot_codes(n_bits: int, n_codes: int) -> List[List[int]]:
    """Generate two-hot encoding codes for compressed representation.

    Args:
        n_bits: Number of bits in each code
        n_codes: Number of codes to generate

    Returns:
        List of two-hot codes, each with exactly two 1s
    """
    max_possible = int(comb(n_bits, 2))
    n_codes = min(n_codes, max_possible)

    table = [[0] * (n_bits - 2) + [1] * 2]

    for _ in range(1, n_codes):
        code = table[-1].copy()
        # Find rightmost [0, 1] pattern
        swap = [idx for idx in range(len(code) - 1, -1, -1) if code[idx : idx + 2] == [0, 1]][0]
        # Swap to [1, 0]
        code[swap : swap + 2] = [1, 0]
        # If next element is 1, move all 1s after swap to the right
        if swap + 2 < len(code) and code[swap + 2] == 1:
            code[swap + 2 :] = code[: swap + 1 : -1]
        table.append(code)

    return table


def create_downsampling_matrix(dim_in: int, dim_out: int) -> Tensor:
    """Create downsampling matrix to select first dim_out elements from dim_in.

    Args:
        dim_in: Input dimension
        dim_out: Output dimension (must be <= dim_in)

    Returns:
        Downsampling matrix [dim_in x dim_out]
    """
    return cat([eye(dim_out, dtype=float), zeros((dim_in - dim_out, dim_out))])
