"""Explicit figure registration module.

Import this module to register all built-in figures. This avoids circular
import issues by separating registration from core infrastructure.
"""

from torch_tem.diagnostics.traces import DataTrace, SimulationTrace, TEMTrace
from torch_tem.figures.modules import environment, overview, split, walk
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register all built-in TEM figures.

    This function is idempotent - it can be called multiple times safely.
    Re-registering a figure with the same name is a no-op.
    """
    # Model figures
    REGISTRY.register(
        FigureSpec(
            name="overview",
            description="Multi-panel TEM model overview (g_inf, g_gen, actions)",
            plot=overview.observations.plot,
            accepts=TEMTrace,
            tags={"model", "rollout", "overview"},
        )
    )

    # Combined model + spatial figures
    REGISTRY.register(
        FigureSpec(
            name="overview.rate_maps",
            description="Multi-panel overview with spatial rate maps (g_inf, g_gen)",
            plot=overview.rate_maps.plot,
            accepts=SimulationTrace,
            tags={"model", "rollout", "spatial", "overview"},
        )
    )

    # Environment figures
    REGISTRY.register(
        FigureSpec(
            name="environment.layout",
            description="Static environment layout showing locations and connectivity",
            plot=environment.layout.plot,
            accepts=DataTrace,
            tags={"data", "environment", "debug"},
        )
    )

    # Walk figures
    REGISTRY.register(
        FigureSpec(
            name="walk.trajectories",
            description="Walk trajectories overlaid on environment map (deterministic)",
            plot=walk.trajectories.plot,
            accepts=DataTrace,
            tags={"data", "walk", "debug"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="walk.statistics",
            description="Walk summary statistics (length, actions, revisits, shiny hits)",
            plot=walk.statistics.plot,
            accepts=DataTrace,
            tags={"data", "walk", "statistics"},
        )
    )

    # Split figures
    REGISTRY.register(
        FigureSpec(
            name="split.statistics",
            description="Dataset split summary statistics (env sizes, walk lengths)",
            plot=split.statistics.plot,
            accepts=DataTrace,
            tags={"data", "split", "statistics"},
        )
    )
