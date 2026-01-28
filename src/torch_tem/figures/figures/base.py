"""Base types for figure orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Protocol


class PanelCallable(Protocol):
    """Protocol for panel render functions."""

    def __call__(self, ax: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        ...


@dataclass(frozen=True)
class StyleContext:
    """Style context for resolving tokenized styles."""

    theme: "Theme"

    def resolve(self, style: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Resolve style tokens to concrete values.

        Args:
                style: Optional style mapping.

        Returns:
                Resolved style dictionary.
        """
        from torch_tem.figures.figures.styles import resolve_style_tokens

        return resolve_style_tokens(style or {}, self.theme)


@dataclass(frozen=True)
class GuidePolicy:
    """Guide policy configuration for composition."""

    legend: str = "shared"
    colorbar: str = "shared"


@dataclass(frozen=True)
class LayoutConfig:
    """Layout configuration for figure grids."""

    nrows: int
    ncols: int
    sharex: bool = False
    sharey: bool = False
