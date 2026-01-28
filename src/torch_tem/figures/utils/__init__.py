"""Shared utilities for figures and plots."""

from torch_tem.figures.utils.autocorr import RadialStats, autocorr2d, autocorr_1d, radial_profile, radial_stats_from_location_values
from torch_tem.figures.utils.axes import format_map_axes
from torch_tem.figures.utils.color import ensure_color_sequence, resolve_categorical_colors, to_rgba
from torch_tem.figures.utils.data import as_1d_array, as_2d_array, validate_heatmap_data, validate_xy
from torch_tem.figures.utils.line import resolve_line_style
from torch_tem.figures.utils.spatial import (
    GridIndex,
    aggregate_by_location,
    build_grid_index,
    grid_from_location_values,
    infer_location_count,
    path_xy_from_location_ids,
    world_xy_from_locations,
)

__all__ = [
    "as_1d_array",
    "as_2d_array",
    "GridIndex",
    "RadialStats",
    "aggregate_by_location",
    "autocorr2d",
    "autocorr_1d",
    "build_grid_index",
    "ensure_color_sequence",
    "format_map_axes",
    "grid_from_location_values",
    "infer_location_count",
    "path_xy_from_location_ids",
    "radial_profile",
    "radial_stats_from_location_values",
    "resolve_categorical_colors",
    "resolve_line_style",
    "to_rgba",
    "validate_heatmap_data",
    "validate_xy",
    "world_xy_from_locations",
]
