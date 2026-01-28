"""Shared utilities for figures and plots."""

from torch_tem.figures.utils.color import ensure_color_sequence, resolve_categorical_colors, to_rgba
from torch_tem.figures.utils.data import as_1d_array, as_2d_array, validate_heatmap_data, validate_xy
from torch_tem.figures.utils.line import resolve_line_style

__all__ = [
    "as_1d_array",
    "as_2d_array",
    "ensure_color_sequence",
    "resolve_categorical_colors",
    "resolve_line_style",
    "to_rgba",
    "validate_heatmap_data",
    "validate_xy",
]
