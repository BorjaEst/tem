"""Deprecated compatibility wrappers for map primitives."""

from __future__ import annotations

from typing import List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from torch_tem.figures.plots.map import (
    action_patch,
    configure_environment_axes,
    plot_actions as _plot_actions,
    plot_map as _plot_map,
    plot_walk as _plot_walk,
    _default_radius,
)


def initialise_axes(
    ax: Optional[Axes] = None,
    *,
    environment: Optional[object] = None,
    radius: Optional[float] = None,
    padding_scale: float = 2.0,
) -> Axes:
    """Configure axes for environment map plotting (deprecated).

    Args:
        ax: Existing axes to configure. If None, creates new figure and axes.
        environment: Optional environment with a ``locations`` list.
        radius: Optional marker radius used to pad axis limits.
        padding_scale: Multiplier applied to radius for axis padding.

    Returns:
        Configured matplotlib Axes object.
    """
    warnings.warn(
        "initialise_axes is deprecated. Use configure_environment_axes.",
        DeprecationWarning,
        stacklevel=2,
    )
    if ax is None:
        _, ax = plt.subplots()
    return configure_environment_axes(
        ax,
        environment=environment,
        radius=radius,
        padding_scale=padding_scale,
    )


def plot_map(
    environment: object,
    values: np.ndarray,
    ax: Optional[Axes] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    num_cols: int = 100,
    location_cm: str = "viridis",
    action_cm: str = "Pastel1",
    do_plot_actions: bool = False,
    shape: str = "circle",
    radius: Optional[float] = None,
) -> Axes:
    """Render an environment map with per-location scalar values (deprecated)."""
    warnings.warn(
        "plot_map has moved to torch_tem.figures.plots.map.",
        DeprecationWarning,
        stacklevel=2,
    )
    if ax is None:
        _, ax = plt.subplots()
    return _plot_map(
        ax,
        environment,
        values,
        min_val=min_val,
        max_val=max_val,
        num_cols=num_cols,
        location_cm=location_cm,
        action_cm=action_cm,
        do_plot_actions=do_plot_actions,
        shape=shape,
        radius=radius,
    )


def plot_walk(
    environment: object,
    walk: List,
    max_steps: Optional[int] = None,
    n_steps: int = 1,
    ax: Optional[Axes] = None,
) -> Axes:
    """Overlay a walk trajectory onto an environment map (deprecated)."""
    warnings.warn(
        "plot_walk has moved to torch_tem.figures.plots.map.",
        DeprecationWarning,
        stacklevel=2,
    )
    if ax is None:
        _, ax = plt.subplots()
        configure_environment_axes(ax, environment=environment)
    return _plot_walk(
        ax,
        environment,
        walk,
        max_steps=max_steps,
        n_steps=n_steps,
    )


def plot_actions(
    environment: object,
    field: str = "probability",
    ax: Optional[Axes] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    num_cols: int = 100,
    action_cm: str = "viridis",
) -> Axes:
    """Visualize action properties across the environment (deprecated)."""
    warnings.warn(
        "plot_actions has moved to torch_tem.figures.plots.map.",
        DeprecationWarning,
        stacklevel=2,
    )
    if ax is None:
        _, ax = plt.subplots()
    return _plot_actions(
        ax,
        environment,
        field=field,
        min_val=min_val,
        max_val=max_val,
        num_cols=num_cols,
        action_cm=action_cm,
    )


__all__ = [
    "action_patch",
    "configure_environment_axes",
    "initialise_axes",
    "plot_actions",
    "plot_map",
    "plot_walk",
    "_default_radius",
]
