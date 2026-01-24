"""Explicit figure registration module.

Import this module to register all built-in figures. This avoids circular
import issues by separating registration from core infrastructure.
"""

from torch_tem.diagnostics.traces import ModelTrace
from torch_tem.figures import tem_overview
from torch_tem.figures.core.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register all built-in TEM figures.

    This function is idempotent - it can be called multiple times safely.
    Re-registering a figure with the same name is a no-op.
    """
    REGISTRY.register(
        FigureSpec(
            name="tem.overview",
            description="Multi-panel TEM model overview (g_inf, g_gen, actions)",
            plot=tem_overview.plot,
            accepts=ModelTrace,
            tags={"model", "rollout", "overview"},
        )
    )
