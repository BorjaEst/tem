"""Rasterplot utilities for observations and activations."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable


def plot_rasterplot(
    ax: Axes,
    observations: np.ndarray,
    activations: Sequence[np.ndarray] | None = None,
    *,
    activation_names: Sequence[str] | None = None,
    n_observations: Optional[int] = None,
    cmap: str = "GnBu",
    vmin: float = 0.0,
    vmax: float = 1.0,
    panel_pad: float = 0.02,
    obs_height: float = 0.42,
) -> Axes:
    """Plot observations (raster) and activations (heatmaps) in a stacked panel.

    The container axes is used to place inset axes that share the time axis.
    Colorbar discovery relies on Matplotlib artists in the inset axes.

    Args:
            ax: Container axes to draw into.
            observations: Observations as (T, O) one-hot/multi-hot or (T,) IDs.
            activations: Sequence of activation arrays shaped (T, D).
            activation_names: Optional labels for activation panels.
            n_observations: Optional number of observation dimensions when
                    observations are provided as IDs.
            cmap: Colormap for observation raster and activation heatmaps.
            vmin: Minimum value for activations normalization.
            vmax: Maximum value for activations normalization.
            panel_pad: Vertical padding between panels (axes fraction).
            obs_height: Height fraction of the observation panel.

        Returns:
            The container axes with the rasterplot panels rendered.
    """
    activations_list = list(activations or [])
    observation_matrix = _coerce_observations(observations, n_observations)
    n_steps = observation_matrix.shape[0]
    _validate_time_lengths(n_steps, activations_list)

    ax.set_axis_off()
    axes = _create_panel_axes(ax=ax, n_activation=len(activations_list), obs_height=obs_height, panel_pad=panel_pad)
    obs_ax = axes[0]
    act_axes = axes[1:]

    obs_mappable = plot_observations(obs_ax, observation_matrix, cmap=cmap)
    if obs_mappable is not None:
        obs_ax.set_ylabel("Observations")

    act_mappable = None
    for idx, (act_ax, activation) in enumerate(zip(act_axes, activations_list)):
        label = activation_names[idx] if activation_names and idx < len(activation_names) else None
        act_mappable = plot_activation(act_ax, activation, cmap=cmap, label=label, vmin=vmin, vmax=vmax)
        if idx < len(act_axes) - 1:
            act_ax.set_xticklabels([])
    if act_axes:
        act_axes[-1].set_xlabel("Time step")
    if not act_axes and obs_ax is not None:
        obs_ax.set_xlabel("Time step")

    return ax


def _coerce_observations(observations: np.ndarray, n_observations: Optional[int]) -> np.ndarray:
    """Ensure observations are a (T, O) float matrix."""
    obs = np.asarray(observations)
    if obs.ndim == 1:
        n_obs = n_observations if n_observations is not None else int(np.max(obs)) + 1 if obs.size else 0
        if n_obs <= 0:
            return np.zeros((obs.shape[0], 0), dtype=float)
        matrix = np.zeros((obs.shape[0], n_obs), dtype=float)
        valid = (obs >= 0) & (obs < n_obs)
        if valid.any():
            rows = np.arange(obs.shape[0])[valid]
            cols = obs[valid].astype(int)
            matrix[rows, cols] = 1.0
        return matrix
    if obs.ndim != 2:
        raise ValueError("observations must be 1D IDs or a 2D (T, O) array")
    return obs.astype(float, copy=False)


def _validate_time_lengths(n_steps: int, activations: Sequence[np.ndarray]) -> None:
    """Validate that all activations share the same time dimension."""
    for activation in activations:
        if activation.ndim != 2:
            raise ValueError("activations must be 2D arrays shaped (T, D)")
        if activation.shape[0] != n_steps:
            raise ValueError("activations must share the same time length as observations")


def _create_panel_axes(*, ax: Axes, n_activation: int, obs_height: float, panel_pad: float) -> list[Axes]:
    """Create inset axes for observation and activation panels."""
    n_panels = 1 + n_activation
    pad_total = panel_pad * max(n_panels - 1, 0)
    available = max(1.0 - pad_total, 0.01)
    obs_height = min(max(obs_height, 0.05), available)
    act_height = (available - obs_height) / max(n_activation, 1)

    axes: list[Axes] = []
    current_top = 1.0
    obs_bottom = current_top - obs_height
    axes.append(ax.inset_axes([0.0, obs_bottom, 1.0, obs_height]))
    current_top = obs_bottom

    for _ in range(n_activation):
        act_bottom = current_top - panel_pad - act_height
        axes.append(ax.inset_axes([0.0, act_bottom, 1.0, act_height]))
        current_top = act_bottom

    for idx in range(1, len(axes)):
        axes[idx].sharex(axes[0])
    return axes


def plot_observations(
    ax: Axes,
    observations: np.ndarray,
    *,
    cmap: str,
) -> ScalarMappable | None:
    """Plot the observation raster panel."""
    if observations.size == 0:
        ax.text(0.5, 0.5, "No observations", ha="center", va="center", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    image = ax.imshow(observations.T, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([])
    return image


def plot_activation(
    ax: Axes,
    activation: np.ndarray,
    *,
    cmap: str,
    label: str | None,
    vmax: Optional[float] = None,
    vmin: Optional[float] = None,
) -> ScalarMappable | None:
    """Plot a single activation heatmap panel."""
    if activation.size == 0:
        ax.text(0.5, 0.5, "No activations", ha="center", va="center", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return None

    image = ax.imshow(activation.T, aspect="auto", cmap=cmap, interpolation="nearest", vmax=vmax, vmin=vmin)
    if label:
        # Titles keep labels legible without shrinking panel height.
        ax.set_ylabel(label)
    ax.set_yticks([])
    return image
