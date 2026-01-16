"""MEC utility functions and helpers."""

from __future__ import annotations

from typing import Optional, Tuple


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
