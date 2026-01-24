"""Explicit figure registration module.

Import this module to register all built-in figures. This avoids circular
import issues by separating registration from core infrastructure.
"""

from torch_tem.diagnostics.traces import ModelTrace
from torch_tem.figures.core.data_trace import DataTrace
from torch_tem.figures.core.registry import REGISTRY, FigureSpec
from torch_tem.figures.environment import layout
from torch_tem.figures.modules import overview
from torch_tem.figures.split import statistics as split_statistics
from torch_tem.figures.walk import statistics as walk_statistics
from torch_tem.figures.walk import trajectories


def register_builtin_figures() -> None:
    """Register all built-in TEM figures.

    This function is idempotent - it can be called multiple times safely.
    Re-registering a figure with the same name is a no-op.
    """
    # Model figures
    REGISTRY.register(
        FigureSpec(
            name="tem.overview",
            description="Multi-panel TEM model overview (g_inf, g_gen, actions)",
            plot=overview.plot,
            accepts=ModelTrace,
            tags={"model", "rollout", "overview"},
        )
    )

    # Environment figures
    REGISTRY.register(
        FigureSpec(
            name="environment.layout",
            description="Static environment layout showing locations and connectivity",
            plot=layout.plot,
            accepts=DataTrace,
            tags={"data", "environment", "debug"},
        )
    )

    # Walk figures
    REGISTRY.register(
        FigureSpec(
            name="walk.trajectories",
            description="Walk trajectories overlaid on environment map (deterministic)",
            plot=trajectories.plot,
            accepts=DataTrace,
            tags={"data", "walk", "debug"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="walk.statistics",
            description="Walk summary statistics (length, actions, revisits, shiny hits)",
            plot=walk_statistics.plot,
            accepts=DataTrace,
            tags={"data", "walk", "statistics"},
        )
    )

    # Split figures
    REGISTRY.register(
        FigureSpec(
            name="split.statistics",
            description="Dataset split summary statistics (env sizes, walk lengths)",
            plot=split_statistics.plot,
            accepts=DataTrace,
            tags={"data", "split", "statistics"},
        )
    )
