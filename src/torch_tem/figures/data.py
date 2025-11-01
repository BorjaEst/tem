"""Data generation visualization functions.

Provides plotting utilities for environments, policies, walks, and batches.
All functions use Protocol-based typing for flexibility and testability.
"""

from typing import List, Optional, Protocol, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch import Tensor

from torch_tem import utils


# ==============================================================================
# Protocols
# ==============================================================================
class EnvironmentProtocol(Protocol):
    """Minimal environment interface for plotting."""

    n_locations: int
    n_observations: int
    n_actions: int
    adjacency: Tensor


class LocationProtocol(Protocol):
    """Minimal location interface for plotting."""

    id: int
    observation: int
    actions: List


class WalkProtocol(Protocol):
    """Minimal walk interface for plotting."""

    observations: Tensor
    actions: Tensor
    locations: Tensor


# ==============================================================================
# Environment Visualization
# ==============================================================================
def _compute_layout(adj: np.ndarray, n_locs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal node positions based on graph structure.

    Uses automatic layout detection:
    1. Detects grid structure → use grid layout
    2. Otherwise → use NetworkX spring/kamada_kawai layout (if available)
    3. Fallback → circular layout

    Returns:
        Tuple of (x_positions, y_positions) arrays
    """
    # Try to detect grid structure
    grid_dims = utils.detect_grid_structure(adj, n_locs)

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

    # Normalize to [-1, 1] range
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range > 0:
        x = 2 * (x - x.min()) / x_range - 1
    if y_range > 0:
        y = 2 * (y - y.min()) / y_range - 1

    return x, y


def plot_environment_layout(env: EnvironmentProtocol, title: str = "Environment Layout", figsize: Tuple[float, float] = (10, 10)) -> plt.Figure:
    """Plot environment structure with automatic optimal layout.

    Automatically detects graph structure and chooses best visualization:
    - Grid graphs: Use grid layout preserving spatial structure
    - General graphs: Use force-directed layout (NetworkX if available)
    - Fallback: Circular layout

    Args:
        env: Environment with adjacency matrix and location count
        title: Plot title
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object

    Note:
        Install networkx for better graph layouts: `pip install networkx`
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Get adjacency matrix (handle both Tensor and list)
    adj = env.adjacency
    if isinstance(adj, list):
        adj = torch.tensor(adj).numpy()
    else:
        adj = adj.numpy()
    n_locs = env.n_locations

    # Compute optimal layout
    x, y = _compute_layout(adj, n_locs)

    # Plot connections
    for i in range(n_locs):
        for j in range(n_locs):
            if adj[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], "k-", alpha=0.2, linewidth=1)

    # Plot locations
    ax.scatter(x, y, s=500, c="lightblue", edgecolors="black", linewidths=2, zorder=10)

    # Label locations
    for i in range(n_locs):
        ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=10, fontweight="bold")

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    return fig


# ==============================================================================
# Policy Visualization
# ==============================================================================
def plot_policy_comparison(
    env: EnvironmentProtocol, policies: List[Tuple[str, List[LocationProtocol]]], goal_location: Optional[int] = None, figsize: Tuple[float, float] = (6, 6)
) -> plt.Figure:
    """Compare multiple policy types side by side.

    Args:
        env: Environment providing dimensions
        policies: List of (name, policy) tuples to compare
        goal_location: Optional goal location to highlight
        figsize: Figure size per subplot

    Returns:
        matplotlib Figure object
    """
    n_policies = len(policies)
    fig, axes = plt.subplots(1, n_policies, figsize=(figsize[0] * n_policies, figsize[1]))

    # Handle single policy case
    if n_policies == 1:
        axes = [axes]

    for ax, (name, policy) in zip(axes, policies):
        # Extract action probabilities for each location
        n_locs = env.n_locations
        n_actions = env.n_actions
        prob_matrix = np.zeros((n_locs, n_actions))

        for loc_id, location in enumerate(policy):
            for action in location.actions:
                prob_matrix[loc_id, action.id] = action.probability

        im = ax.imshow(prob_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xlabel("Location", fontsize=12)
        ax.set_ylabel("Action", fontsize=12)

        if goal_location is not None:
            title = f"{name} Policy (Goal: {goal_location})"
        else:
            title = f"{name} Policy"
        ax.set_title(title, fontsize=14, fontweight="bold")

        ax.set_xticks(range(0, n_locs, max(1, n_locs // 10)))
        ax.set_yticks(range(n_actions))
        plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    return fig


# ==============================================================================
# Walk Visualization
# ==============================================================================
def plot_walks(
    env: EnvironmentProtocol, walks: List[WalkProtocol], title: str = "Generated Walks", figsize: Tuple[float, float] = (12, 8), max_walks: Optional[int] = None
) -> plt.Figure:
    """Visualize multiple walks through the environment.

    Args:
        env: Environment providing structure
        walks: List of Walk objects to visualize
        title: Plot title
        figsize: Figure size
        max_walks: Maximum number of walks to plot (None = all)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_locs = env.n_locations

    # Create circular layout
    angles = np.linspace(0, 2 * np.pi, n_locs, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Plot connections (light gray)
    adj = env.adjacency
    if isinstance(adj, list):
        adj = torch.tensor(adj).numpy()
    else:
        adj = adj.numpy()

    for i in range(n_locs):
        for j in range(n_locs):
            if adj[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], "gray", alpha=0.1, linewidth=1, zorder=1)

    # Limit number of walks if requested
    walks_to_plot = walks[:max_walks] if max_walks else walks
    colors = plt.cm.tab10(np.linspace(0, 1, len(walks_to_plot)))

    # Plot each walk
    for walk_idx, walk in enumerate(walks_to_plot):
        # Extract location sequence
        if isinstance(walk.locations, Tensor):
            locs = walk.locations.tolist()
        else:
            locs = list(walk.locations)

        walk_x = [x[loc] for loc in locs]
        walk_y = [y[loc] for loc in locs]

        # Plot walk trajectory
        ax.plot(walk_x, walk_y, "-", color=colors[walk_idx], alpha=0.6, linewidth=2, zorder=5)
        ax.scatter(walk_x[0], walk_y[0], s=200, c=[colors[walk_idx]], marker="o", edgecolors="black", linewidths=2, zorder=10, label=f"Walk {walk_idx+1} start")
        ax.scatter(walk_x[-1], walk_y[-1], s=200, c=[colors[walk_idx]], marker="s", edgecolors="black", linewidths=2, zorder=10)

    # Plot location nodes
    ax.scatter(x, y, s=300, c="lightgray", edgecolors="black", linewidths=1.5, zorder=8, alpha=0.7)

    for i in range(n_locs):
        ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=8, zorder=9)

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend
    circle_patch = mpatches.Patch(color="gray", label="Start (circle)")
    square_patch = mpatches.Patch(color="gray", label="End (square)")
    ax.legend(handles=[circle_patch, square_patch], loc="upper right")

    plt.tight_layout()
    return fig


# ==============================================================================
# Walk Statistics
# ==============================================================================
def plot_walk_statistics(walks: List[WalkProtocol], figsize: Tuple[float, float] = (14, 10)) -> plt.Figure:
    """Plot statistics about generated walks.

    Analyzes walk length distribution, location visits, action distribution,
    and action repeat patterns.

    Args:
        walks: List of Walk objects to analyze
        figsize: Figure size

    Returns:
        matplotlib Figure with 2x2 subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract data from walks
    walk_lengths = []
    location_visits = []
    action_sequences = []

    for walk in walks:
        # Handle both Tensor and list formats
        if isinstance(walk.locations, Tensor):
            locs = walk.locations.tolist()
            acts = walk.actions.tolist()
        else:
            locs = list(walk.locations)
            acts = list(walk.actions)

        walk_lengths.append(len(locs))
        location_visits.extend(locs)
        action_sequences.extend(acts)

    # Plot 1: Walk lengths distribution
    axes[0, 0].hist(walk_lengths, bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].set_xlabel("Walk Length", fontsize=12)
    axes[0, 0].set_ylabel("Count", fontsize=12)
    axes[0, 0].set_title("Walk Length Distribution", fontsize=13, fontweight="bold")
    axes[0, 0].axvline(np.mean(walk_lengths), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(walk_lengths):.1f}")
    axes[0, 0].legend()

    # Plot 2: Location visit frequency
    unique_locs, counts = np.unique(location_visits, return_counts=True)
    axes[0, 1].bar(unique_locs, counts, edgecolor="black", alpha=0.7)
    axes[0, 1].set_xlabel("Location ID", fontsize=12)
    axes[0, 1].set_ylabel("Visit Count", fontsize=12)
    axes[0, 1].set_title("Location Visit Frequency", fontsize=13, fontweight="bold")

    # Plot 3: Action distribution
    unique_actions, action_counts = np.unique(action_sequences, return_counts=True)
    axes[1, 0].bar(unique_actions, action_counts, edgecolor="black", alpha=0.7, color="coral")
    axes[1, 0].set_xlabel("Action ID", fontsize=12)
    axes[1, 0].set_ylabel("Action Count", fontsize=12)
    axes[1, 0].set_title("Action Distribution", fontsize=13, fontweight="bold")

    # Plot 4: Action repeat analysis
    repeats = []
    for walk in walks:
        if isinstance(walk.actions, Tensor):
            acts = walk.actions.tolist()
        else:
            acts = list(walk.actions)

        current_repeat = 1
        for i in range(1, len(acts)):
            if acts[i] == acts[i - 1]:
                current_repeat += 1
            else:
                if current_repeat > 1:
                    repeats.append(current_repeat)
                current_repeat = 1

    if repeats:
        axes[1, 1].hist(repeats, bins=range(1, max(repeats) + 2), edgecolor="black", alpha=0.7, color="green")
        axes[1, 1].set_xlabel("Repeat Length", fontsize=12)
        axes[1, 1].set_ylabel("Count", fontsize=12)
        axes[1, 1].set_title("Action Repeat Analysis", fontsize=13, fontweight="bold")
        axes[1, 1].axvline(np.mean(repeats), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(repeats):.1f}")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No action repeats detected", ha="center", va="center", fontsize=12, transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("Action Repeat Analysis", fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig


# ==============================================================================
# Batch Tensor Visualization
# ==============================================================================
def plot_batch_tensors(obs: Tensor, actions: Tensor, locations: Tensor, figsize: Tuple[float, float] = (14, 10), max_walks: int = 5) -> plt.Figure:
    """Visualize batch tensor shapes and samples.

    Provides overview of batch structure with:
    - Observation heatmap for first walk
    - Action sequences overlayed
    - Location trajectories overlayed
    - Observation activation statistics

    Args:
        obs: Observation tensor [batch, walk_length, n_observations]
        actions: Action tensor [batch, walk_length]
        locations: Location tensor [batch, walk_length]
        figsize: Figure size
        max_walks: Maximum walks to overlay in trajectory plots

    Returns:
        matplotlib Figure with 2x2 subplots
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    batch_size, walk_length, n_obs = obs.shape

    # Plot 1: Observation heatmap (first walk in batch)
    im1 = axes[0, 0].imshow(obs[0].T.numpy(), aspect="auto", cmap="viridis")
    axes[0, 0].set_xlabel("Time Step", fontsize=12)
    axes[0, 0].set_ylabel("Observation ID", fontsize=12)
    axes[0, 0].set_title(f"Observations (Walk 0)\nShape: {tuple(obs.shape)}", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 0], label="Value")

    # Plot 2: Action sequence (overlay multiple walks)
    n_to_plot = min(max_walks, batch_size)
    for i in range(n_to_plot):
        axes[0, 1].plot(actions[i].numpy(), alpha=0.6, label=f"Walk {i}")
    axes[0, 1].set_xlabel("Time Step", fontsize=12)
    axes[0, 1].set_ylabel("Action ID", fontsize=12)
    axes[0, 1].set_title(f"Action Sequences\nShape: {tuple(actions.shape)}", fontsize=13, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Location trajectory (overlay multiple walks)
    for i in range(n_to_plot):
        axes[1, 0].plot(locations[i].numpy(), alpha=0.6, label=f"Walk {i}")
    axes[1, 0].set_xlabel("Time Step", fontsize=12)
    axes[1, 0].set_ylabel("Location ID", fontsize=12)
    axes[1, 0].set_title(f"Location Trajectories\nShape: {tuple(locations.shape)}", fontsize=13, fontweight="bold")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Observation activation statistics
    obs_activations = obs.sum(dim=1).mean(dim=0).numpy()  # Average across time and batch
    axes[1, 1].bar(range(n_obs), obs_activations, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("Observation ID", fontsize=12)
    axes[1, 1].set_ylabel("Mean Activation", fontsize=12)
    axes[1, 1].set_title("Observation Activation Statistics", fontsize=13, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    return fig
