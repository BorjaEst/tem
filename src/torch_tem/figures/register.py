"""Registration helpers for built-in figures."""

from __future__ import annotations

from torch_tem.figures.modules import autocorr, overview, spatial
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
