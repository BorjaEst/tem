"""Graph layout utilities for visualization.

Provides functions to compute optimal node positions for graph visualization.
"""

from typing import Optional, Tuple

import numpy as np

from .matrices import detect_grid_structure


def compute_graph_layout(adj: np.ndarray, n_locs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal node positions based on graph structure.

    Uses automatic layout detection:
    1. Detects grid structure → use grid layout
    2. Otherwise → use NetworkX spring/kamada_kawai layout (if available)
    3. Fallback → circular layout

    Args:
        adj: Adjacency matrix as numpy array
        n_locs: Number of locations (nodes) in the graph

    Returns:
        Tuple of (x_positions, y_positions) arrays normalized to [-1, 1] range

    Example:
        >>> adj = np.array([[0, 1, 1, 0],
        ...                 [1, 0, 0, 1],
        ...                 [1, 0, 0, 1],
        ...                 [0, 1, 1, 0]])
        >>> x, y = compute_graph_layout(adj, 4)
        >>> x.shape
        (4,)
    """
    # Try to detect grid structure
    grid_dims = detect_grid_structure(adj, n_locs)

    if grid_dims is not None:
        # Use grid layout
        width, height = grid_dims
        x = np.array([loc_id % width for loc_id in range(n_locs)])
        y = np.array([loc_id // width for loc_id in range(n_locs)])

        # Normalize to [-1, 1] range
        if width > 1:
            x = 2 * (x / (width - 1)) - 1
        else:
            x = np.zeros_like(x)

        if height > 1:
            y = 2 * (y / (height - 1)) - 1
        else:
            y = np.zeros_like(y)

        # Flip y to match standard orientation (top = higher index)
        y = -y

        return x, y

    # Try NetworkX layouts if available
    try:
        import networkx as nx

        # Build NetworkX graph
        G = nx.Graph()
        G.add_nodes_from(range(n_locs))
        for i in range(n_locs):
            for j in range(i + 1, n_locs):
                if adj[i, j] > 0 or adj[j, i] > 0:
                    G.add_edge(i, j)

        # Choose layout based on graph properties
        if n_locs <= 20:
            # Kamada-Kawai works well for small graphs
            try:
                pos = nx.kamada_kawai_layout(G)
            except:
                # Fallback to spring layout
                pos = nx.spring_layout(G, k=1 / np.sqrt(n_locs), iterations=50)
        else:
            # Spring layout scales better for larger graphs
            pos = nx.spring_layout(G, k=1 / np.sqrt(n_locs), iterations=50)

        # Extract coordinates
        x = np.array([pos[i][0] for i in range(n_locs)])
        y = np.array([pos[i][1] for i in range(n_locs)])

    except ImportError:
        # NetworkX not available, use circular layout as fallback
        angles = np.linspace(0, 2 * np.pi, n_locs, endpoint=False)
        x = np.cos(angles)
        y = np.sin(angles)

    # Normalize to [-1, 1] range
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range > 0:
        x = 2 * (x - x.min()) / x_range - 1
    if y_range > 0:
        y = 2 * (y - y.min()) / y_range - 1

    return x, y
