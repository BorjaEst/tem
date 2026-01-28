"""Registry for figure specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Set

import matplotlib.figure as mpl_figure

from torch_tem.diagnostics.traces import TraceTree


@dataclass(frozen=True)
class FigureContext:
    """Context information passed to figure plot functions."""

    env_idx: int = 0
    freq_idx: int = 0
    figsize: Optional[tuple[float, float]] = None
    dpi: Optional[int] = None
    global_step: Optional[int] = None
    split_name: Optional[str] = None


@dataclass(frozen=True)
class FigureSpec:
    """Specification for a named figure."""

    name: str
    plot: Callable[[TraceTree, FigureContext], mpl_figure.Figure]
    default_filename: str
    tags: Set[str] = field(default_factory=set)
    description: str = ""


class Registry:
    """Registry of figure specifications."""

    def __init__(self) -> None:
        self._specs: dict[str, FigureSpec] = {}

    def register(self, spec: FigureSpec) -> None:
        """Register a figure specification.

        Args:
            spec: Figure specification.

        Raises:
            ValueError: If the name is already registered.
        """
        if spec.name in self._specs:
            raise ValueError(f"Figure '{spec.name}' already registered")
        self._specs[spec.name] = spec

    def get(self, name: str) -> FigureSpec:
        """Return a registered figure spec."""
        if name not in self._specs:
            raise KeyError(f"Unknown figure '{name}'")
        return self._specs[name]

    def validate(self, names: Iterable[str]) -> None:
        """Validate that all names are registered."""
        missing = [name for name in names if name not in self._specs]
        if missing:
            raise ValueError(f"Unknown figures: {', '.join(missing)}")

    def list(self) -> list[str]:
        """Return sorted list of registered figure names."""
        return sorted(self._specs.keys())


REGISTRY = Registry()
