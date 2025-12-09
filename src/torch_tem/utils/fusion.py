"""Bayesian fusion utilities for probabilistic estimates.

This module provides functions for fusing multiple Gaussian estimates via
precision-weighted averaging. It implements the core algorithm for combining
probabilistic location estimates in the Tolman-Eichenbaum Machine.

Theory:
    Given multiple independent Gaussian estimates N(μ_i, σ²_i), the optimal
    linear fusion under Gaussian assumptions is:

        precision_i = 1 / σ²_i
        μ_fused = Σ(precision_i * μ_i) / Σ(precision_i)
        σ_fused = 1 / √Σ(precision_i)

    This is equivalent to maximum likelihood estimation when observations
    are independent and Gaussian-distributed.

Key Functions:
    - fuse_transitions: Fuse multiple Transition estimates
    - precision_weighted_mean: Core fusion algorithm for tensors
    - sample_transition: Sample from Gaussian transition estimate
"""

from typing import List, Tuple

import torch
from torch import Tensor

from torch_tem.types import AbstractLocation, Transition


def fuse_transitions(estimates: List[Transition]) -> Transition:
    """Fuse multiple transition estimates via precision weighting.

    Combines multiple Gaussian estimates of abstract location by weighting
    each according to its precision (inverse variance). More certain estimates
    (lower σ) receive higher weight in the fusion.

    Args:
        estimates: List of transition estimates to fuse. Must contain at least one.

    Returns:
        Fused transition with combined mean and reduced uncertainty

    Raises:
        ValueError: If estimates list is empty

    Theory:
        Precision-weighted fusion is the optimal way to combine independent
        Gaussian estimates under the assumption of independence. The fused
        uncertainty is always smaller than the minimum input uncertainty,
        reflecting increased certainty from multiple observations.

    Example:
        >>> path = Transition(mean=mu_path, uncertainty=sigma_path)
        >>> memory = Transition(mean=mu_mem, uncertainty=sigma_mem)
        >>> fused = fuse_transitions([path, memory])
        >>> # fused.uncertainty < min(sigma_path, sigma_mem)
    """
    if not estimates:
        raise ValueError("Cannot fuse empty list of estimates")

    if len(estimates) == 1:
        return estimates[0]

    n_f = len(estimates[0].mean)
    fused_mean = []
    fused_uncertainty = []

    # Fuse per frequency module
    for f in range(n_f):
        means_f = [est.mean[f] for est in estimates]
        uncertainties_f = [est.uncertainty[f] for est in estimates]
        mu, sigma = precision_weighted_mean(means_f, uncertainties_f)
        fused_mean.append(mu)
        fused_uncertainty.append(sigma)

    return Transition(mean=fused_mean, uncertainty=fused_uncertainty)


def precision_weighted_mean(means: List[Tensor], uncertainties: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Compute precision-weighted mean of Gaussian estimates.

    Implements the core Bayesian fusion algorithm for combining multiple
    Gaussian-distributed estimates. Each estimate is weighted by its precision
    (inverse variance), giving more influence to more certain estimates.

    Args:
        means: List of mean estimates [μ_1, μ_2, ..., μ_n]
        uncertainties: List of uncertainty estimates [σ_1, σ_2, ..., σ_n]

    Returns:
        (μ_fused, σ_fused): Combined mean and uncertainty

    Theory:
        Given n independent Gaussian estimates N(μ_i, σ²_i):

        1. Compute precisions: w_i = 1/σ²_i
        2. Weighted mean: μ = Σ(w_i·μ_i) / Σ(w_i)
        3. Fused uncertainty: σ = 1/√Σ(w_i)

        This minimizes the variance of the fused estimate and is equivalent
        to maximum likelihood estimation under independence.

    Example:
        >>> mu1 = torch.tensor([1.0, 2.0])
        >>> mu2 = torch.tensor([1.5, 2.5])
        >>> sigma1 = torch.tensor([0.1, 0.1])  # High certainty
        >>> sigma2 = torch.tensor([1.0, 1.0])  # Low certainty
        >>> mu_fused, sigma_fused = precision_weighted_mean([mu1, mu2], [sigma1, sigma2])
        >>> # mu_fused will be close to mu1 (higher precision)
        >>> # sigma_fused < 0.1 (fusion reduces uncertainty)
    """
    # Stack estimates along first dimension
    mus = torch.stack(means, dim=0)  # [n_sources, ...]
    sigmas = torch.stack(uncertainties, dim=0)  # [n_sources, ...]

    # Compute precisions (inverse variance)
    precisions = 1.0 / (sigmas**2)  # [n_sources, ...]

    # Precision-weighted mean
    total_precision = torch.sum(precisions, dim=0)  # [...]
    weighted_mean = torch.sum(precisions * mus, dim=0) / total_precision  # [...]

    # Combined uncertainty (inverse of square root of total precision)
    combined_sigma = 1.0 / torch.sqrt(total_precision)  # [...]

    return weighted_mean, combined_sigma


def sample_transition(transition: Transition) -> AbstractLocation:
    """Sample from Gaussian transition estimate.

    Generates a random sample from the Gaussian distribution represented
    by the transition's mean and uncertainty. This is used during training
    to add exploration noise to location estimates.

    Args:
        transition: Transition with mean and uncertainty

    Returns:
        Sampled abstract location (mean + σ·ε where ε ~ N(0,1))

    Theory:
        Sampling from N(μ, σ²) is implemented via the reparameterization trick:
            x = μ + σ·ε where ε ~ N(0,1)

        This allows gradients to flow through the sampling operation during
        backpropagation, enabling end-to-end learning.

    Example:
        >>> estimate = Transition(mean=mu_g, uncertainty=sigma_g)
        >>> sample1 = sample_transition(estimate)  # Random sample
        >>> sample2 = sample_transition(estimate)  # Different random sample
        >>> # sample1 ≠ sample2, both distributed around estimate.mean
    """
    return [transition.mean[f] + transition.uncertainty[f] * torch.randn_like(transition.mean[f]) for f in range(len(transition.mean))]
