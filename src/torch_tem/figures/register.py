"""Explicit figure registration module.

Import this module to register all built-in figures. This avoids circular
import issues by separating registration from core infrastructure.
"""

from torch_tem.figures.modules import cells, coverage, decoding, dynamics, environment, memory, overview, representation, split, uncertainty, walk
from torch_tem.figures.registry import REGISTRY, FigureSpec


def register_builtin_figures() -> None:
    """Register all built-in TEM figures.

    This function is idempotent - it can be called multiple times safely.
    Re-registering a figure with the same name is a no-op.
    """
    # Model overview figures
    REGISTRY.register(
        FigureSpec(
            name="overview.circuit_overview",
            description="Multi-panel TEM circuit overview (LEC/MEC/HPC)",
            plot=overview.circuit_overview.plot,
            tags={"model", "rollout", "overview"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="overview.spatial_structure",
            description="Multi-panel overview with occupancy, abstract rate maps, autocorrelograms, and trajectory",
            plot=overview.spatial_structure.plot,
            tags={"model", "rollout", "spatial", "overview"},
        )
    )

    # Environment figures
    REGISTRY.register(
        FigureSpec(
            name="environment.layout",
            description="Static environment layout showing locations and connectivity",
            plot=environment.layout.plot,
            tags={"data", "environment", "debug"},
        )
    )

    # Walk figures
    REGISTRY.register(
        FigureSpec(
            name="walk.trajectories",
            description="Walk trajectories overlaid on environment map (deterministic)",
            plot=walk.trajectories.plot,
            tags={"data", "walk", "debug"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="walk.statistics",
            description="Walk summary statistics (length, actions, revisits, shiny hits)",
            plot=walk.statistics.plot,
            tags={"data", "walk", "statistics"},
        )
    )

    # Split figures
    REGISTRY.register(
        FigureSpec(
            name="split.statistics",
            description="Dataset split summary statistics (env sizes, walk lengths)",
            plot=split.statistics.plot,
            tags={"data", "split", "statistics"},
        )
    )

    # Cell figures
    REGISTRY.register(
        FigureSpec(
            name="cells.place_rate_maps",
            description="Place-cell rate maps with occupancy overview",
            plot=cells.place_rate_maps.plot,
            tags={"model", "rollout", "spatial", "cells"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="cells.place_rate_maps.frequencies",
            description="Place-cell rate maps across frequency modules",
            plot=cells.place_rate_maps.plot_frequencies,
            tags={"model", "rollout", "spatial", "cells"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="cells.place_rate_maps.pathways",
            description="Place-cell rate maps across inference/generative pathways",
            plot=cells.place_rate_maps.plot_pathways,
            tags={"model", "rollout", "spatial", "cells"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="cells.place_field_summary",
            description="Summary statistics of place fields (sparsity, peaks, size)",
            plot=cells.place_field_summary.plot,
            tags={"model", "rollout", "cells", "statistics"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="cells.remapping_correlation",
            description="Correlation of rate maps across environments",
            plot=cells.remapping_correlation.plot,
            tags={"model", "rollout", "cells", "spatial"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="cells.spatial_autocorrelogram",
            description="Radial spatial autocorrelogram for place-like activity",
            plot=cells.spatial_autocorrelogram.plot,
            tags={"model", "rollout", "cells", "spatial"},
        )
    )

    # Coverage figures
    REGISTRY.register(
        FigureSpec(
            name="coverage.occupancy_map",
            description="Occupancy map showing visited locations",
            plot=coverage.occupancy_map.plot,
            tags={"data", "coverage", "spatial"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="coverage.action_bias_map",
            description="Action bias map based on per-location action entropy",
            plot=coverage.action_bias_map.plot,
            tags={"data", "coverage", "spatial"},
        )
    )

    # Decoding figures
    REGISTRY.register(
        FigureSpec(
            name="decoding.location_error_map",
            description="Decoding error aggregated by location",
            plot=decoding.location_error_map.plot,
            tags={"model", "rollout", "spatial", "decoding"},
        )
    )

    # Dynamics figures
    REGISTRY.register(
        FigureSpec(
            name="dynamics.path_integration_drift",
            description="Path integration drift over time",
            plot=dynamics.path_integration_drift.plot,
            tags={"model", "rollout", "dynamics"},
        )
    )

    REGISTRY.register(
        FigureSpec(
            name="dynamics.sequence_consistency",
            description="Consistency between generated and inferred sequences",
            plot=dynamics.sequence_consistency.plot,
            tags={"model", "rollout", "dynamics"},
        )
    )

    # Representation figures
    REGISTRY.register(
        FigureSpec(
            name="representation.freq_similarity",
            description="Representational similarity across frequencies",
            plot=representation.freq_similarity.plot,
            tags={"model", "rollout", "representation"},
        )
    )

    # Memory figures
    REGISTRY.register(
        FigureSpec(
            name="memory.retrieval_error_by_location",
            description="Retrieval error aggregated by location",
            plot=memory.retrieval_error_by_location.plot,
            tags={"model", "rollout", "memory", "spatial"},
        )
    )

    # Uncertainty figures
    REGISTRY.register(
        FigureSpec(
            name="uncertainty.calibration",
            description="Uncertainty calibration curve",
            plot=uncertainty.calibration.plot,
            tags={"model", "rollout", "uncertainty"},
        )
    )
