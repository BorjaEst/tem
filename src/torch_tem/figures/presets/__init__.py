"""Figure presets for publication-oriented layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class FigurePreset:
    name: str
    figsize: tuple[float, float]
    dpi: Optional[int] = None
    png_dpi: Optional[int] = None
    style: Optional[dict[str, Any]] = None
    tick_fontsize: Optional[int] = None
    color_cycle: Optional[list[str]] = None


_PAPER_DOUBLE_STYLE: dict[str, Any] = {
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "lines.linewidth": 0.9,
}


_PRESETS: dict[str, FigurePreset] = {
    "paper_double": FigurePreset(
        name="paper_double",
        figsize=(7.2, 4.8),
        dpi=100,
        png_dpi=300,
        style=_PAPER_DOUBLE_STYLE,
        tick_fontsize=6,
    ),
}


def get_preset(name: Optional[str]) -> Optional[FigurePreset]:
    if name is None:
        return None
    return _PRESETS.get(name)


def list_presets() -> list[str]:
    return sorted(_PRESETS.keys())
