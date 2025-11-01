"""Precision weighting utilities for torch_tem package.

This module implements precision-weighted averaging of multiple estimates,
commonly used in hierarchical predictive coding models to combine predictions
from different sources (e.g., sensory, memory, transition).

The precision-weighted mean is computed as:
    precision_i = 1 / sigma_i^2
    mean = sum(precision_i * mu_i) / sum(precision_i)

where mu_i and sigma_i are the mean and standard deviation of the i-th source.
"""

from typing import List

import torch
from torch import Tensor


def precision_weighted_mean(means: List[Tensor], sigmas: List[Tensor], epsilon: float = 1e-8) -> Tensor:
    """Combine multiple estimates via precision weighting.

    Args:
        means: List of mean estimates [n_sources] of [B, ...]
        sigmas: List of uncertainty estimates [n_sources] of [B, ...]
        epsilon: Small constant to avoid division by zero

    Returns:
        Precision-weighted mean [B, ...]
    """
    precisions = [1.0 / (sigma**2 + epsilon) for sigma in sigmas]
    weighted_sum = sum(p * mu for p, mu in zip(precisions, means))
    precision_sum = sum(precisions)
    return weighted_sum / precision_sum


def precision_weighted_mean_list(means_list: List[List[Tensor]], sigmas_list: List[List[Tensor]], epsilon: float = 1e-8) -> List[Tensor]:
    """Combine multiple estimates via precision weighting for frequency modules.

    This function applies precision-weighted averaging per frequency module.

    Args:
        means_list: List of mean lists [n_sources] of [n_f] of [B, n_dim[f]]
        sigmas_list: List of sigma lists [n_sources] of [n_f] of [B, n_dim[f]]
        epsilon: Small constant to avoid division by zero

    Returns:
        Precision-weighted means [n_f] of [B, n_dim[f]]
    """
    if not means_list or not sigmas_list:
        raise ValueError("means_list and sigmas_list must not be empty")

    if len(means_list) != len(sigmas_list):
        raise ValueError("means_list and sigmas_list must have the same length")

    n_f = len(means_list[0])
    result = []

    for f in range(n_f):
        means_f = [means[f] for means in means_list]
        sigmas_f = [sigmas[f] for sigmas in sigmas_list]
        result.append(precision_weighted_mean(means_f, sigmas_f, epsilon))

    return result
