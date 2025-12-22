"""Visualization utilities for TEM data and model outputs.

This module provides standardized plotting functions for:
- Data generation visualization (environments, policies, walks)
- Sensory processing visualization (frequency banks, temporal filtering)
- Model state visualization (activations, memories, predictions)
- Training diagnostics (losses, metrics, convergence)

All plotting functions follow consistent conventions:
- Return matplotlib Figure objects for flexible saving/display
- Accept Protocol-based interfaces for loose coupling
- Use Pydantic for configuration validation
- Provide both quick-plot and customizable interfaces
"""

from torch_tem.figures.abstract import plot_abstract_location_snapshot, plot_g_inf_evolution, plot_schedule_effect, plot_source_contributions, plot_uncertainty_evolution
from torch_tem.figures.data import (
    plot_batch_tensors,
    plot_batch_tensors_time_major,
    plot_environment_layout,
    plot_location_visit_map,
    plot_policy_comparison,
    plot_split_location_visit_statistics,
    plot_walk_statistics,
    plot_walks,
)
from torch_tem.figures.grounded import plot_grounded_location_activity, plot_outer_product_structure, plot_place_cell_dynamics
from torch_tem.figures.memory import plot_attractor_convergence, plot_hierarchical_masks, plot_learning_curve, plot_memory_difference, plot_memory_matrices, plot_retrieval_quality
from torch_tem.figures.patterns import (
    plot_grid_cell_patterns,
    plot_grid_temporal_evolution,
    plot_oscillatory_patterns,
    plot_pattern_comparison,
    plot_pattern_correlations,
    plot_place_cell_patterns,
    plot_temporal_patterns,
    plot_transition_uncertainty,
)
from torch_tem.figures.sensory import (
    plot_frequency_bank,
    plot_frequency_comparison,
    plot_multi_frequency_representation,
    plot_normalization_effects,
    plot_sensory_projection,
    plot_temporal_filtering,
)
from torch_tem.figures.simulation import plot_abstract_location_heatmap, plot_memory_formation_timeline, plot_prediction_accuracy

__all__ = [
    # Data generation
    "plot_environment_layout",
    "plot_policy_comparison",
    "plot_walks",
    "plot_walk_statistics",
    "plot_batch_tensors",
    "plot_batch_tensors_time_major",
    "plot_location_visit_map",
    "plot_split_location_visit_statistics",
    # Pattern generation
    "plot_place_cell_patterns",
    "plot_grid_cell_patterns",
    "plot_temporal_patterns",
    "plot_oscillatory_patterns",
    "plot_pattern_correlations",
    "plot_pattern_comparison",
    "plot_transition_uncertainty",
    "plot_grid_temporal_evolution",
    # Grounded location inference
    "plot_grounded_location_activity",
    "plot_outer_product_structure",
    "plot_place_cell_dynamics",
    # Memory system
    "plot_memory_matrices",
    "plot_attractor_convergence",
    "plot_learning_curve",
    "plot_hierarchical_masks",
    "plot_retrieval_quality",
    "plot_memory_difference",
    # Sensory processing
    "plot_frequency_bank",
    "plot_temporal_filtering",
    "plot_frequency_comparison",
    "plot_normalization_effects",
    "plot_multi_frequency_representation",
    "plot_sensory_projection",
    # Abstract location inference
    "plot_source_contributions",
    "plot_uncertainty_evolution",
    "plot_g_inf_evolution",
    "plot_schedule_effect",
    "plot_abstract_location_snapshot",
    # Simulation state evolution
    "plot_abstract_location_heatmap",
    "plot_memory_formation_timeline",
    "plot_prediction_accuracy",
]
