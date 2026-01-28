"""Color utilities for figure primitives and style resolution."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.colors as mcolors

from torch_tem.figures.palettes.categorical import get_palette


def to_rgba(color: str | Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a color specification to RGBA.

    Args:
            color: Color name, hex string, or RGB/RGBA tuple.

    Returns:
            RGBA tuple.
    """
    return mcolors.to_rgba(color)


def resolve_categorical_colors(name: str | None, n: int) -> List[str]:
    """Resolve a categorical palette to a list of colors.

    Args:
            name: Palette name or None for the default palette.
            n: Number of colors to return.

    Returns:
            List of color strings.
    """
    return get_palette(name, n)


def ensure_color_sequence(colors: Iterable[str] | None, n: int) -> List[str]:
    """Ensure a list of colors with a requested length.

    Args:
            colors: Optional iterable of colors.
            n: Requested number of colors.

    Returns:
            List of colors with length n.
    """
    if colors is None:
        return resolve_categorical_colors(None, n)
    resolved = list(colors)
    if not resolved:
        return resolve_categorical_colors(None, n)
    if len(resolved) >= n:
        return resolved[:n]
    padded = []
    while len(padded) < n:
        padded.extend(resolved)
    return padded[:n]
