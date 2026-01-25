"""Figure registry for discoverable, name-based figure generation.

Provides a central registry mapping stable figure names to plotting functions.
This enables configuration-driven figure selection and validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from matplotlib.figure import Figure


@dataclass
class FigureContext:
    """Metadata and styling context for figure generation.

    Provides figures with runtime information and styling preferences.

    Attributes:
        env_idx: Environment index to visualize (for batch traces).
        freq_idx: Frequency module index to visualize (for multi-scale models).
        figsize: Figure size in inches (width, height).
        style: Optional styling configuration (e.g., StyleConfig from figures.style).
        global_step: Optional training step for context.
        split_name: Optional split name (e.g., "train", "val").
    """

    env_idx: int = 0
    freq_idx: int = 0
    figsize: tuple[float, float] = (12, 8)
    style: Optional[Any] = None
    global_step: Optional[int] = None
    split_name: Optional[str] = None


@dataclass
class FigureSpec:
    """Specification for a registered figure module.

    Attributes:
        name: Unique stable identifier (e.g., "overview").
        description: Human-readable description of the figure.
        plot: Plotting function (trace, ctx) -> Figure.
        accepts: Type hint for accepted trace type.
        default_filename: Default filename stem for saved figures (without extension).
        tags: Optional set of tags for grouping/filtering figures.
    """

    name: str
    description: str
    plot: Callable[[Any, FigureContext], Figure]
    accepts: type[Any]
    default_filename: Optional[str] = None
    tags: frozenset[str] = frozenset()

    def __post_init__(self):
        """Set default filename if not provided."""
        if self.default_filename is None:
            self.default_filename = self.name.replace(".", "_")

        # Normalize tags to an immutable set for deterministic behavior.
        if not isinstance(self.tags, frozenset):
            self.tags = frozenset(self.tags)


class FigureRegistry:
    """Registry for figure modules.

    Maintains a mapping from stable figure names to FigureSpec objects.
    Supports registration, lookup, listing, and validation.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._specs: dict[str, FigureSpec] = {}

    def register(self, spec: FigureSpec) -> None:
        """Register a figure specification.

        Idempotent: Re-registering the same name is allowed and is a no-op.

        Args:
            spec: FigureSpec to register.
        """
        if spec.name in self._specs:
            # Idempotent: Allow re-registration (no-op)
            return
        self._specs[spec.name] = spec

    def get(self, name: str) -> FigureSpec:
        """Retrieve a figure specification by name.

        Args:
            name: Figure name to lookup.

        Returns:
            FigureSpec for the requested figure.

        Raises:
            KeyError: If figure name is not registered.
        """
        if name not in self._specs:
            available = ", ".join(sorted(self._specs.keys()))
            raise KeyError(f"Figure '{name}' not found. Available: {available}")
        return self._specs[name]

    def list(self) -> list[FigureSpec]:
        """List all registered figures.

        Returns:
            List of FigureSpec objects in alphabetical order by name.
        """
        return [self._specs[name] for name in sorted(self._specs.keys())]

    def validate(self, names: list[str]) -> None:
        """Validate that all names are registered.

        Args:
            names: List of figure names to validate.

        Raises:
            ValueError: If any name is not registered (includes available names).
        """
        missing = [name for name in names if name not in self._specs]
        if missing:
            available = ", ".join(sorted(self._specs.keys()))
            raise ValueError(f"Unknown figure names: {missing}. Available: {available}")


# Global registry instance
REGISTRY = FigureRegistry()
