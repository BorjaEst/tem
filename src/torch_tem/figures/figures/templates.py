"""Figure templates for common layout and export defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Template:
    """Template bundle for themes and export settings."""

    name: str
    theme: str
    size: Tuple[float, float]
    dpi: int
    constrained_layout: bool = True


_TEMPLATES: Dict[str, Template] = {
    "default": Template(
        name="default",
        theme="default",
        size=(10.0, 6.0),
        dpi=120,
        constrained_layout=True,
    ),
    "paper": Template(
        name="paper",
        theme="paper",
        size=(8.0, 5.0),
        dpi=150,
        constrained_layout=True,
    ),
    "presentation": Template(
        name="presentation",
        theme="default",
        size=(12.0, 7.0),
        dpi=120,
        constrained_layout=False,
    ),
}


def get_template(name: Optional[str]) -> Template:
    """Return a template instance.

    Args:
            name: Template name or None for default.

    Returns:
            Template instance.
    """
    key = name or "default"
    return _TEMPLATES.get(key, _TEMPLATES["default"])
