"""Registration helpers for built-in figures."""

from __future__ import annotations

from torch_tem.figures.modules import grid_cells, overview, place_cells
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register built-in figure specifications."""
    if not REGISTRY.has("overview"):
        REGISTRY.register(
            FigureSpec(
                name="overview",
                plot=overview.plot,
                default_filename="overview",
                tags={"episode"},
                description="Overview of rollout observations",
            )
        )
    if not REGISTRY.has("grid_cells"):
        REGISTRY.register(
            FigureSpec(
                name="grid_cells",
                plot=grid_cells.plot,
                default_filename="grid_cells",
                tags={"episode"},
                description="MEC grid-cell overview",
            )
        )
    if not REGISTRY.has("place_cells"):
        REGISTRY.register(
            FigureSpec(
                name="place_cells",
                plot=place_cells.plot,
                default_filename="place_cells",
                tags={"episode"},
                description="HPC place-cell overview",
            )
        )
