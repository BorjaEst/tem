"""Categorical color palettes."""

from __future__ import annotations

from typing import Dict, List

_PALETTES: Dict[str, List[str]] = {
    "default": [
        "#4C78A8",
        "#F58518",
        "#E45756",
        "#72B7B2",
        "#54A24B",
        "#EECA3B",
        "#B279A2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ],
    "muted": [
        "#1F77B4",
        "#FF7F0E",
        "#2CA02C",
        "#D62728",
        "#9467BD",
        "#8C564B",
        "#E377C2",
        "#7F7F7F",
        "#BCBD22",
        "#17BECF",
    ],
}


def get_palette(name: str | None, n: int) -> List[str]:
    """Return a categorical palette with n colors.

    Args:
            name: Palette name or None for the default.
            n: Number of colors to return.

    Returns:
            List of color hex strings.
    """
    key = name or "default"
    colors = _PALETTES.get(key, _PALETTES["default"])
    if len(colors) >= n:
        return colors[:n]
    resolved: List[str] = []
    while len(resolved) < n:
        resolved.extend(colors)
    return resolved[:n]
