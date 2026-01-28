"""Explicit figure registration module.

Import this module to register all built-in figures. This avoids circular
import issues by separating registration from core infrastructure.
"""

from torch_tem.figures.modules import overview, spatial
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register all built-in TEM figures.

    This function is idempotent - it can be called multiple times safely.
    Re-registering a figure with the same name is a no-op.
    """
    # Model overview figures
    REGISTRY.register(
        FigureSpec(
            name="overview",
            description="Multi-panel TEM circuit overview (LEC/MEC/HPC)",
            plot=overview.plot,
            tags={"model", "rollout", "overview"},
        )
    )

    # Spatial structure figures
    REGISTRY.register(
        FigureSpec(
            name="spatial.structure",
            description="Multi-panel overview with occupancy, abstract rate maps, autocorrelograms, and trajectory",
            plot=spatial.structure.plot,
            tags={"model", "rollout", "spatial"},
        )
    )
    REGISTRY.register(
        FigureSpec(
            name="spatial.autocorrelogram",
            description="Multi-panel spatial summary for HPC place-like activity",
            plot=spatial.autocorrelogram.plot,
            tags={"model", "rollout", "spatial"},
        )
    )
