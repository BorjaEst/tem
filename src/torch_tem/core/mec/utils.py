"""MEC utility functions and helpers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
from torch import Tensor

from torch_tem import utils as tem_utils


def fuse_inv_var(
    mu_base: List[Tensor],
    sigma_base: List[Tensor],
    mu_corr: List[Tensor],
    sigma_corr: List[Tensor],
    *,
    mask: Optional[Tensor] = None,
    freqs: Optional[range] = None,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Fuse correction into base using inverse-variance weighting.

    Supports selective fusion by frequency range and batch masking.
    Uses precision weighting to combine two multi-scale estimates.

    Args:
        mu_base: Base means per frequency (length n_freq).
        sigma_base: Base uncertainties per frequency (length n_freq).
        mu_corr: Correction means.
        sigma_corr: Correction uncertainties.
        mask: Optional boolean mask (batch,) to fuse only subset of batch.
            If None, fuses entire batch.
        freqs: Frequencies to fuse.
            - None: fuse all frequencies; mu_corr/sigma_corr must have length n_freq.
            - range: fuse only those indices; mu_corr/sigma_corr must have length len(freqs).

    Returns:
        mu_fused: Fused means (same shape as mu_base)
        sigma_fused: Fused uncertainties (same shape as sigma_base)

    Examples:
        >>> # Fuse all frequencies, entire batch
        >>> mu, sigma = fuse_inv_var(mu_path, sig_path, mu_mem, sig_mem)

        >>> # Fuse last 2 frequencies, only shiny environments
        >>> mu, sigma = fuse_inv_var(
        ...     mu_base, sig_base, mu_shiny, sig_shiny,
        ...     mask=shiny_mask, freqs=range(3, 5)
        ... )
    """
    if freqs is None:  # Fuse all frequencies by default
        freqs = range(len(mu_base))

    mu_out, sigma_out = list(mu_base), list(sigma_base)

    if mask is None:
        for i, f in enumerate(freqs):
            mu_out[f], sigma_out[f] = tem_utils.inv_var_weight([mu_base[f], mu_corr[i]], [sigma_base[f], sigma_corr[i]])

    else:
        for i, f in enumerate(freqs):
            mu_f, sigma_f = mu_base[f].clone(), sigma_base[f].clone()
            mu_u, sig_u = tem_utils.inv_var_weight([mu_base[f][mask], mu_corr[i]], [sigma_base[f][mask], sigma_corr[i]])
            mu_f[mask], sigma_f[mask] = mu_u, sig_u
            mu_out[f], sigma_out[f] = mu_f, sigma_f

    return mu_out, sigma_out


def connections(f_grid: list[float]) -> list[list[bool]]:
    """Compute hierarchical connection matrix from frequency list.

    Entry [f_to][f_from] is True if f_from connects to f_to.
    Connection rule: lower-frequency modules connect to higher-frequency.

    Args:
        f_grid: Frequency values per module (higher value = higher frequency)

    Returns:
        Connection matrix where [f_to][f_from] indicates connectivity
    """
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]


def resolve_ovc_slice(n_freq_total: int, n_freq_ovc: Optional[int]) -> Tuple[int, int]:
    """Determine which MEC modules are OVC (receive shiny landmark correction).

    Args:
        n_freq_total: Total number of MEC modules
        n_freq_ovc: User setting for OVC module count:
            - None: all modules are OVC (legacy separate_ovc=False)
            - 0: no OVC modules (disable shiny correction)
            - k>0: last k modules are OVC (legacy separate_ovc=True, n_f_ovc=k)

    Returns:
        (ovc_start, ovc_count): slice range [ovc_start:ovc_start+ovc_count] for OVC modules

    Examples:
        >>> resolve_ovc_slice(5, None)  # All modules are OVC
        (0, 5)
        >>> resolve_ovc_slice(5, 0)     # No OVC modules
        (5, 0)
        >>> resolve_ovc_slice(5, 2)     # Last 2 modules are OVC
        (3, 2)
    """
    if n_freq_ovc is None:
        # Apply to all modules (legacy separate_ovc=False)
        return 0, n_freq_total

    n_freq_ovc = int(n_freq_ovc)
    if n_freq_ovc < 0 or n_freq_ovc > n_freq_total:
        raise ValueError(f"MECSettings.n_freq_ovc must be in [0, {n_freq_total}] or None; got {n_freq_ovc}")

    if n_freq_ovc == 0:
        # No OVC correction
        return n_freq_total, 0

    # Apply to last k modules (legacy separate_ovc=True)
    return n_freq_total - n_freq_ovc, n_freq_ovc
