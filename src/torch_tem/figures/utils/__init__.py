"""Utility helpers for figure modules."""

from torch_tem.figures.utils.spatial import (
    aggregate_rate_map,
    autocorr_2d,
    autocorr_extent,
    clip_unit_interval,
    infer_distance_scale,
    infer_grid_spacing,
    radial_autocorr,
    robust_min_max,
    select_feature_by_spatial_variance,
    select_top_k_by_spatial_variance,
    summarize_radial_autocorr,
)

__all__ = [
    "aggregate_rate_map",
    "autocorr_2d",
    "autocorr_extent",
    "clip_unit_interval",
    "infer_distance_scale",
    "infer_grid_spacing",
    "radial_autocorr",
    "robust_min_max",
    "select_feature_by_spatial_variance",
    "select_top_k_by_spatial_variance",
    "summarize_radial_autocorr",
]
