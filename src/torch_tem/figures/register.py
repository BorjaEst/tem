"""Registration helpers for built-in figures."""

from __future__ import annotations

from torch_tem.figures.modules import grid_cells, place_cells, tem_overview
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications."""
    if not REGISTRY.has("tem-overview"):
        REGISTRY.register(
            FigureSpec(
                name="tem-overview",
                plot=tem_overview.plot,
                default_filename="tem-overview",
                tags={"episode"},
                description="Overview of rollout observations",
            )
        )
    if not REGISTRY.has("grid-cells"):
        REGISTRY.register(
            FigureSpec(
                name="grid-cells",
                plot=grid_cells.plot,
                default_filename="grid-cells",
                tags={"episode"},
                description="MEC grid-cell overview",
            )
        )
    if not REGISTRY.has("place-cells"):
        REGISTRY.register(
            FigureSpec(
                name="place-cells",
                plot=place_cells.plot,
                default_filename="place-cells",
                tags={"episode"},
                description="HPC place-cell overview",
            )
        )
