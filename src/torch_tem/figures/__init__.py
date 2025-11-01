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

from torch_tem.figures.data import (
    plot_batch_tensors,
    plot_environment_layout,
    plot_policy_comparison,
    plot_walk_statistics,
    plot_walks,
)
from torch_tem.figures.sensory import (
    plot_frequency_bank,
    plot_frequency_comparison,
    plot_multi_frequency_representation,
    plot_normalization_effects,
    plot_temporal_filtering,
)

__all__ = [
    # Data generation
    "plot_environment_layout",
    "plot_policy_comparison",
    "plot_walks",
    "plot_walk_statistics",
    "plot_batch_tensors",
    # Sensory processing
    "plot_frequency_bank",
    "plot_temporal_filtering",
    "plot_frequency_comparison",
    "plot_normalization_effects",
    "plot_multi_frequency_representation",
]
