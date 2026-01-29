import matplotlib.pyplot as plt
import numpy as np

from torch_tem.figures.plots.map import plot_map
from torch_tem.figures.utils import aggregate_rate_map


def plot_rate_map_cell(
    ax: plt.Axes,
    world: object,
    cells: np.ndarray,
    cell_idx: int,
    location_ids: list[int],
    *,
    min_val: float | None = None,
    max_val: float | None = None,
    shape: str = "square",
    location_cm: str = "viridis",
) -> plt.Axes:
    """Plot a single cell's rate map on an existing axes.

    Args:
        ax: Axes to draw into.
        world: Environment world with location coordinates.
        cells: Array of shape (n_steps, n_cells, n_locations) with cell activations.
        cell_idx: Index of the cell to plot.
        location_ids: Ordered list of visited location indices.
        min_val: Minimum value for color scaling.
        max_val: Maximum value for color scaling.
        shape: Shape for the background map markers.
        location_cm: Colormap name for the rate map.

    Returns:
        The axes with the rate map rendered.
    """
    rate_map, _ = aggregate_rate_map(cells, location_ids, len(world.locations))
    values = rate_map[cell_idx]
    max_val = float(np.max(values[~np.isnan(values)])) if np.isfinite(values).any() else 1.0
    max_val = max(max_val, 0.1)

    plot_map(world, values, ax=ax, min_val=min_val, max_val=max_val, shape=shape, location_cm=location_cm)
    return ax
