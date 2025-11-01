"""Mask generation utilities for torch_tem package."""

from typing import List

import numpy as np
import torch
from torch import Tensor


def create_p_update_mask(n_p: List[int], n_f: int, n_f_g: int, n_f_ovc: int, f_initial: List[float]) -> Tensor:
    """Create hierarchical mask for memory updates.

    Set connections when forming Hebbian memory of grounded locations:
    from low frequency modules to high. Entry M_ij (row i, col j) is the
    connection FROM cell i TO cell j.

    Args:
        n_p: Grounded location dimensions per frequency
        n_f: Total number of frequency modules
        n_f_g: Number of grid cell frequency modules
        n_f_ovc: Number of OVC frequency modules
        f_initial: Initial frequencies for each module

    Returns:
        Binary mask tensor [sum(n_p), sum(n_p)]
    """
    p_update_mask = torch.zeros((sum(n_p), sum(n_p)), dtype=torch.float)
    n_p_cumsum = np.cumsum(np.concatenate(([0], n_p)))

    for f_from in range(n_f):
        for f_to in range(n_f):
            # For connections involving separate object vector modules
            if f_from >= n_f_g or f_to >= n_f_g:
                # Connection between object vector modules: hierarchical
                if f_from >= n_f_g and f_to >= n_f_g:
                    if f_initial[f_from] <= f_initial[f_to]:
                        p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
                # Connection between OVC and normal modules: allow any
                else:
                    p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
            # Connection between grid modules: hierarchical only
            else:
                if f_initial[f_from] <= f_initial[f_to]:
                    p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0

    return p_update_mask


def create_p_retrieve_masks(n_p: List[int], i_attractor: int, i_attractor_max_freq_inf: List[int], i_attractor_max_freq_gen: List[int]) -> tuple[List[Tensor], List[Tensor]]:
    """Create hierarchical masks for memory retrieval with early-stopping.

    Hierarchical memory retrieval is implemented by early-stopping low-frequency
    memory updates, using a mask for updates at every retrieval iteration.

    Args:
        n_p: Grounded location dimensions per frequency
        i_attractor: Number of attractor iterations
        i_attractor_max_freq_inf: Max iterations per frequency (inference)
        i_attractor_max_freq_gen: Max iterations per frequency (generation)

    Returns:
        Tuple of (inference_masks, generation_masks), each a list of [i_attractor]
        masks of shape [sum(n_p)]
    """
    n_p_cumsum = np.cumsum(np.concatenate(([0], n_p)))

    # Initialize masks
    p_retrieve_mask_inf = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]
    p_retrieve_mask_gen = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]

    # Build masks for each retrieval iteration
    for mask, max_iters in zip([p_retrieve_mask_inf, p_retrieve_mask_gen], [i_attractor_max_freq_inf, i_attractor_max_freq_gen]):
        for f, max_i in enumerate(max_iters):
            # Update masks up to maximum iteration for this frequency
            for i in range(max_i):
                mask[i][n_p_cumsum[f] : n_p_cumsum[f + 1]] = 1.0

    return p_retrieve_mask_inf, p_retrieve_mask_gen


def create_g_connections(n_f: int, n_f_g: int, n_f_ovc: int, f_initial: List[float]) -> List[List[bool]]:
    """Create hierarchical connections between frequency modules for transitions.

    Abstract location frequency modules can influence the transition of other
    modules hierarchically (low to high frequency).

    Args:
        n_f: Total number of frequency modules
        n_f_g: Number of grid cell frequency modules
        n_f_ovc: Number of OVC frequency modules
        f_initial: Initial frequencies for each module

    Returns:
        List[n_f][n_f] of boolean connections. connections[f_to][f_from]
        indicates if f_from connects to f_to.
    """
    # Grid cell connections: hierarchical
    g_connections = [[f_initial[f_from] <= f_initial[f_to] for f_from in range(n_f_g)] + [False for _ in range(n_f_ovc)] for f_to in range(n_f_g)]

    # OVC connections: only between OVC modules, also hierarchical
    g_connections += [[False for _ in range(n_f_g)] + [f_initial[f_from] <= f_initial[f_to] for f_from in range(n_f_g, n_f)] for f_to in range(n_f_g, n_f)]

    return g_connections
