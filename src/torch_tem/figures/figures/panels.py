"""Decorators for figure panel methods."""

from __future__ import annotations

from typing import Any, Callable


def colorbar(*, group: str, label: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Attach colorbar grouping metadata to a panel method.

    Args:
        group: Shared colorbar group name.
        label: Optional colorbar label.

    Returns:
        A decorator that attaches metadata to the function.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        setattr(fn, "_tem_colorbar", {"group": group, "label": label})
        return fn

    return decorator
