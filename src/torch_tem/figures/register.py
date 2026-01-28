"""Registration helpers for built-in figures."""

from __future__ import annotations

from torch_tem.figures.modules import autocorr, grid_cells, overview, spatial
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
    if "overview.model_state" not in REGISTRY.list():
        REGISTRY.register(
            FigureSpec(
                name="overview.model_state",
                plot=overview.model_state.plot,
                default_filename="overview_model_state",
                tags={"episode"},
                description="Model overview with memory, observations, and spatial maps",
            )
        )
    if "spatial.structure" not in REGISTRY.list():
        REGISTRY.register(
            FigureSpec(
                name="spatial.structure",
                plot=spatial.structure.plot,
                default_filename="spatial_structure",
                tags={"coverage"},
                description="Spatial structure summary",
            )
        )
    if "autocorr.g_gen" not in REGISTRY.list():
        REGISTRY.register(
            FigureSpec(
                name="autocorr.g_gen",
                plot=autocorr.g_gen.plot,
                default_filename="autocorr_g_gen",
                tags={"episode"},
                description="Autocorrelation of inferred state",
            )
        )
    if "grid_cells.g_gen" not in REGISTRY.list():
        REGISTRY.register(
            FigureSpec(
                name="grid_cells.g_gen",
                plot=grid_cells.g_gen.plot,
                default_filename="grid_cells_g_gen",
                tags={"episode", "grid"},
                description="Grid-cell diagnostics for g_gen",
            )
        )
