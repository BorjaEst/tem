"""Theme definitions and style resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class Theme:
    """Theme tokens for style resolution."""

    name: str
    tokens: Mapping[str, Any]


_THEMES: Dict[str, Theme] = {
    "default": Theme(
        name="default",
        tokens={
            "primary": "#4C78A8",
            "secondary": "#F58518",
            "grid": "#D9D9D9",
            "background": "#FFFFFF",
            "font_size": 10,
        },
    ),
    "paper": Theme(
        name="paper",
        tokens={
            "primary": "#1F77B4",
            "secondary": "#FF7F0E",
            "grid": "#E0E0E0",
            "background": "#FFFFFF",
            "font_size": 9,
        },
    ),
}


def get_theme(name: str | None) -> Theme:
    """Return a theme instance.

    Args:
            name: Theme name or None for default.

    Returns:
            Theme instance.
    """
    key = name or "default"
    return _THEMES.get(key, _THEMES["default"])


def resolve_style_tokens(style: Mapping[str, Any], theme: Theme) -> Dict[str, Any]:
    """Resolve tokenized style values to concrete values.

    Args:
            style: Style mapping with optional token values.
            theme: Theme used for token resolution.

    Returns:
            Resolved style dictionary.
    """
    resolved: Dict[str, Any] = {}
    for key, value in style.items():
        if isinstance(value, str) and value in theme.tokens:
            resolved[key] = theme.tokens[value]
        else:
            resolved[key] = value
    return resolved


def default_primitive_style(primitive: str, theme: Theme) -> Dict[str, Any]:
    """Return default styles for a plot primitive.

    Args:
            primitive: Primitive name.
            theme: Theme instance.

    Returns:
            Default style dictionary.
    """
    if primitive == "line":
        return {"color": theme.tokens["primary"], "linewidth": 2.0}
    if primitive == "scatter":
        return {"color": theme.tokens["primary"], "alpha": 0.9}
    if primitive == "hist":
        return {"color": theme.tokens["primary"], "alpha": 0.8}
    return {}
