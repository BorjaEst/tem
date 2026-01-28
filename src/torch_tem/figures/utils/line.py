"""Line styling helpers for plot primitives."""

from __future__ import annotations

from typing import Any, Mapping


def resolve_line_style(style: Mapping[str, Any] | None, **overrides: Any) -> dict[str, Any]:
    """Resolve line style configuration.

    Args:
            style: Optional style mapping.
            overrides: Explicit keyword overrides.

    Returns:
            Resolved style dictionary.
    """
    resolved: dict[str, Any] = {
        "linewidth": 2.0,
        "alpha": 0.9,
    }
    if style:
        resolved.update(style)
    resolved.update(overrides)
    return resolved
