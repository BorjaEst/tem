"""Diagnostics extraction for TEM visualization.

This package provides utilities for extracting plot-ready data from TEM
rollouts and model outputs. All extraction functions return CPU-based NumPy
arrays or dataclasses, isolating figure generation from GPU memory and live
model execution.
"""

from torch_tem.diagnostics.extract_tem import compute_location_uncertainty, compute_sensory_accuracy, extract_rollout_trace
from torch_tem.diagnostics.rollout_trace import EnvMapData, TEMRolloutTrace

__all__ = ["TEMRolloutTrace", "EnvMapData", "extract_rollout_trace", "compute_sensory_accuracy", "compute_location_uncertainty"]
