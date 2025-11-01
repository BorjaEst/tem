"""Visualization utilities for TEM data and model outputs.

This module provides standardized plotting functions for:
- Data generation visualization (environments, policies, walks)
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

__all__ = [
    "plot_environment_layout",
    "plot_policy_comparison",
    "plot_walks",
    "plot_walk_statistics",
    "plot_batch_tensors",
]
