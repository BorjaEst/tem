"""Registration helpers for built-in figures."""

from __future__ import annotations

from torch_tem.figures.modules import overview
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications."""
    if "overview" not in REGISTRY.list():
        REGISTRY.register(
            FigureSpec(
                name="overview",
                plot=overview.plot,
                default_filename="overview",
                tags={"episode"},
                description="Overview of rollout observations",
            )
        )
