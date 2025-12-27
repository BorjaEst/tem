"""Data generation visualization functions.

Provides plotting utilities for environments, policies, walks, and batches.
All functions use Protocol-based typing for flexibility and testability.
"""

from typing import Dict, List, Optional, Protocol, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torch_tem.types import Matrix, Vector
from torch_tem.utils import compute_graph_layout


# ==============================================================================
# Protocols
# ==============================================================================
class EnvironmentProtocol(Protocol):
    """Minimal environment interface for plotting."""

    n_locations: int
    n_observations: int
    n_actions: int
    adjacency: Matrix


class LocationProtocol(Protocol):
    """Minimal location interface for plotting."""

    id: int
    observation: int
    actions: List


class WalkProtocol(Protocol):
    """Minimal walk interface for plotting."""

    observations: Vector
    actions: Vector
    locations: Vector


# ==============================================================================
# Environment Visualization
# ==============================================================================
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
    x, y = compute_graph_layout(adj, n_locs)

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
def plot_walks(env: EnvironmentProtocol, walks: List[WalkProtocol], title: str = "Generated Walks", figsize: Tuple[float, float] = (12, 8)) -> plt.Figure:
    """Visualize multiple walks through the environment.

    Args:
        env: Environment providing structure
        walks: List of Walk objects to visualize
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_locs = env.n_locations

    # Get adjacency matrix (handle both Tensor and list)
    adj = env.adjacency
    if isinstance(adj, list):
        adj = torch.tensor(adj).numpy()
    else:
        adj = adj.numpy()

    # Compute optimal layout using automatic detection
    x, y = compute_graph_layout(adj, n_locs)

    for i in range(n_locs):
        for j in range(n_locs):
            if adj[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], "gray", alpha=0.1, linewidth=1, zorder=1)

    # Expand input into per-walk sequences (supports batched [T,B] walks)
    sequences: List[List[int]] = []
    for walk in walks:
        sequences.extend([locs for locs, _acts in _walk_sequences_time_major(walk)])

    # Plot each walk
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sequences))))
    for walk_idx, locs in enumerate(sequences):
        # Defensive: ignore invalid indices
        locs_valid = [int(loc) for loc in locs if 0 <= int(loc) < n_locs]
        if len(locs_valid) < 2:
            continue

        walk_x = [x[loc] for loc in locs_valid]
        walk_y = [y[loc] for loc in locs_valid]

        # Plot walk trajectory
        ax.plot(walk_x, walk_y, "-", color=colors[walk_idx % len(colors)], alpha=0.6, linewidth=2, zorder=5)
        ax.scatter(walk_x[0], walk_y[0], s=200, c=[colors[walk_idx % len(colors)]], marker="o", edgecolors="black", linewidths=2, zorder=10)
        ax.scatter(walk_x[-1], walk_y[-1], s=200, c=[colors[walk_idx % len(colors)]], marker="s", edgecolors="black", linewidths=2, zorder=10)

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


def _ensure_time_major(x: Tensor) -> Tensor:
    """Heuristically ensure time-major layout for batched sequences.

    The TEM DataModule yields time-major tensors: [T, B, ...].
    Some legacy code may provide batch-major tensors: [B, T, ...].

    This helper keeps [T, B, ...] unchanged and swaps the first two axes for
    likely [B, T, ...] inputs.
    """

    if x.ndim < 2:
        return x

    t_dim = int(x.shape[0])
    b_dim = int(x.shape[1])

    # Common case: B is small, T is larger. If the first dim is small and the
    # second dim is much larger, assume [B, T, ...] and transpose.
    if t_dim <= 64 and b_dim > t_dim:
        return x.transpose(0, 1)
    return x


def _walk_sequences_time_major(walk: WalkProtocol) -> List[Tuple[List[int], List[int]]]:
    """Convert a WalkProtocol into a list of (locations, actions) sequences.

    Supported inputs:
    - Per-walk sequences: locations/actions shaped [T]
    - Batched sequences:  locations/actions shaped [T, B] (preferred)
      (also accepts [B, T] and transposes heuristically)
    """

    def _to_tensor(v: Vector) -> Tensor:
        if isinstance(v, Tensor):
            return v
        return torch.as_tensor(v)

    loc = _to_tensor(walk.locations)
    act = _to_tensor(walk.actions)

    if loc.ndim == 1:
        return [(loc.detach().cpu().to(torch.int64).tolist(), act.detach().cpu().to(torch.int64).tolist())]

    loc_tm = _ensure_time_major(loc)
    act_tm = _ensure_time_major(act)

    if loc_tm.ndim != 2 or act_tm.ndim != 2:
        raise ValueError("Expected locations/actions to be [T], [T,B] (or [B,T])")

    if loc_tm.shape[:2] != act_tm.shape[:2]:
        raise ValueError(f"Locations/actions shape mismatch: {tuple(loc_tm.shape)} vs {tuple(act_tm.shape)}")

    t_steps, batch_size = int(loc_tm.shape[0]), int(loc_tm.shape[1])
    sequences: List[Tuple[List[int], List[int]]] = []
    for b in range(batch_size):
        locs_b = loc_tm[:, b].detach().cpu().to(torch.int64).tolist()
        acts_b = act_tm[:, b].detach().cpu().to(torch.int64).tolist()
        # Keep sequences length-consistent (defensive)
        if len(locs_b) != t_steps or len(acts_b) != t_steps:
            continue
        sequences.append((locs_b, acts_b))
    return sequences


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

    sequences_la: List[Tuple[List[int], List[int]]] = []
    for walk in walks:
        sequences_la.extend(_walk_sequences_time_major(walk))

    for locs, acts in sequences_la:
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
    for _locs, acts in sequences_la:

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
# Spatial Visit Visualization (Geometry-Agnostic)
# ==============================================================================
def plot_location_visit_map(
    env: EnvironmentProtocol,
    locations: Vector | List[Vector],
    title: str = "Location Visit Map",
    figsize: Tuple[float, float] = (10, 10),
    show_edges: bool = True,
    cmap: str = "viridis",
) -> plt.Figure:
    """Plot how often each location was visited, without assuming grid geometry.

    This function uses the environment adjacency matrix to compute a 2D layout via
    :func:`torch_tem.utils.compute_graph_layout`, then plots nodes sized/colored by
    visit count.

    Args:
        env: Environment providing adjacency and n_locations.
        locations: Either a single location tensor/array (e.g. [T] or [T, B] or
            [B, T]) or a list of such sequences (e.g. multiple walks).
        title: Plot title.
        figsize: Figure size.
        show_edges: If True, draw graph edges beneath the nodes.
        cmap: Matplotlib colormap name.

    Returns:
        matplotlib Figure object.
    """

    def _to_numpy_1d(seq: Vector) -> np.ndarray:
        if isinstance(seq, Tensor):
            arr = seq.detach().cpu().numpy()
        else:
            arr = np.asarray(seq)
        return arr.reshape(-1)

    if isinstance(locations, list):
        all_locations = np.concatenate([_to_numpy_1d(seq) for seq in locations], axis=0)
    else:
        all_locations = _to_numpy_1d(locations)

    # Defensive: ignore invalid indices (keeps the helper robust to future envs)
    all_locations = all_locations[(all_locations >= 0) & (all_locations < env.n_locations)]
    counts = np.bincount(all_locations.astype(int), minlength=env.n_locations)

    # Get adjacency matrix (handle both Tensor and list)
    adj = env.adjacency
    if isinstance(adj, list):
        adj = torch.tensor(adj).numpy()
    else:
        adj = adj.numpy()

    x, y = compute_graph_layout(adj, env.n_locations)

    fig, ax = plt.subplots(figsize=figsize)

    if show_edges:
        for i in range(env.n_locations):
            for j in range(env.n_locations):
                if adj[i, j] > 0:
                    ax.plot([x[i], x[j]], [y[i], y[j]], "k-", alpha=0.15, linewidth=1, zorder=1)

    max_count = float(np.max(counts)) if counts.size else 0.0
    if max_count <= 0:
        node_sizes = np.full(env.n_locations, 200.0)
        node_colors = np.zeros(env.n_locations)
    else:
        # Size scaling that stays readable across env sizes
        node_sizes = 200.0 + 800.0 * (counts / max_count)
        node_colors = counts

    sc = ax.scatter(
        x,
        y,
        s=node_sizes,
        c=node_colors,
        cmap=cmap,
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
    )

    for i in range(env.n_locations):
        ax.text(x[i], y[i], str(i), ha="center", va="center", fontsize=8, zorder=10)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="Visit count")
    plt.tight_layout()
    return fig


# ==============================================================================
# Split-Level Visit Statistics (Geometry-Agnostic)
# ==============================================================================
def plot_split_statistics(
    env: EnvironmentProtocol,
    datasets: Dict[str, Dataset],
    *,
    title: str = "Split Visitation Statistics",
    figsize: Tuple[float, float] = (14, 10),
    max_items_per_split: Optional[int] = None,
) -> plt.Figure:
    """Plot split-level visitation statistics from DataModule datasets.

    The `TEMDataModule` exposes `datamodule.datasets` as a mapping from split
    name ("fit", "validate", "test") to a `WalkDataset`.

    This convenience wrapper extracts the per-item location sequences from each
    dataset and forwards them to :func:`plot_split_location_visit_statistics`.

    Args:
        env: Environment providing `n_locations`.
        datasets: Mapping from split name to dataset. Each dataset item must
            provide a location sequence either as:
            - a tuple/list where the 3rd element is `locations`, or
            - an object with a `locations` attribute.
        title: Figure title.
        figsize: Figure size.
        max_items_per_split: Optional cap on how many dataset items to sample
            per split (useful for large datasets).

    Returns:
        A matplotlib Figure object.
    """

    def _extract_locations(sample) -> Vector:
        """Extract location sequence from a dataset sample."""
        if isinstance(sample, (tuple, list)) and len(sample) >= 3:
            return sample[2]
        if hasattr(sample, "locations"):
            return getattr(sample, "locations")
        raise TypeError("Dataset sample must be (obs, actions, locations) or have a " "`.locations` attribute")

    split_locations: Dict[str, List[Vector]] = {}
    for split_name, dataset in datasets.items():
        n_items = len(dataset)
        if max_items_per_split is not None:
            n_items = min(n_items, int(max_items_per_split))

        locations_list: List[Vector] = []
        for idx in range(n_items):
            sample = dataset[idx]
            locations_list.append(_extract_locations(sample))

        split_locations[split_name] = locations_list

    return plot_split_location_visit_statistics(
        env,
        split_locations,
        title=title,
        figsize=figsize,
    )


def plot_split_location_visit_statistics(
    env: EnvironmentProtocol,
    split_locations: Dict[str, Vector | List[Vector]],
    title: str = "Split Visitation Statistics",
    figsize: Tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Summarize visitation statistics across data splits (train/val/test).

    This helper intentionally does not assume a grid/hex geometry. It uses only
    location IDs and the environment's `n_locations`.

    Statistics per split:
    - total steps
    - unique visited locations + coverage
    - normalized entropy of visitation distribution
    Additionally, it visualizes split overlap via Jaccard similarity.

    Args:
        env: Environment providing `n_locations`.
        split_locations: Mapping from split name to location sequences. Each
            value can be a single sequence (e.g. [T], [T,B], [B,T]) or a list of
            sequences (e.g. multiple batches/walks).
        title: Figure title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """

    def _to_numpy_1d(seq: Vector) -> np.ndarray:
        if isinstance(seq, Tensor):
            arr = seq.detach().cpu().numpy()
        else:
            arr = np.asarray(seq)
        return arr.reshape(-1)

    def _flatten(value: Vector | List[Vector]) -> np.ndarray:
        if isinstance(value, list):
            if len(value) == 0:
                return np.asarray([], dtype=int)
            arr = np.concatenate([_to_numpy_1d(v) for v in value], axis=0)
        else:
            arr = _to_numpy_1d(value)
        if arr.size == 0:
            return np.asarray([], dtype=int)
        arr = arr[(arr >= 0) & (arr < env.n_locations)]
        return arr.astype(int, copy=False)

    def _normalized_entropy_from_counts(counts: np.ndarray) -> float:
        total = float(np.sum(counts))
        if total <= 0.0:
            return 0.0
        if env.n_locations <= 1:
            return 0.0
        p = counts / total
        p = p[p > 0]
        h = -float(np.sum(p * np.log(p)))
        return float(h / np.log(env.n_locations))

    split_names = list(split_locations.keys())
    if len(split_names) == 0:
        raise ValueError("split_locations must contain at least one split")

    # Compute per-split distributions
    counts_by_split: Dict[str, np.ndarray] = {}
    steps_by_split: Dict[str, int] = {}
    unique_by_split: Dict[str, int] = {}
    coverage_by_split: Dict[str, float] = {}
    entropy_by_split: Dict[str, float] = {}
    visited_sets: Dict[str, set[int]] = {}

    for split in split_names:
        arr = _flatten(split_locations[split])
        counts = np.bincount(arr, minlength=env.n_locations) if arr.size else np.zeros(env.n_locations, dtype=int)
        steps = int(arr.size)
        unique = int(np.count_nonzero(counts))

        counts_by_split[split] = counts
        steps_by_split[split] = steps
        unique_by_split[split] = unique
        coverage_by_split[split] = float(unique / env.n_locations) if env.n_locations > 0 else 0.0
        entropy_by_split[split] = _normalized_entropy_from_counts(counts)
        visited_sets[split] = set(np.nonzero(counts)[0].tolist())

    # Jaccard overlap matrix
    n = len(split_names)
    jaccard = np.zeros((n, n), dtype=float)
    for i, a in enumerate(split_names):
        for j, b in enumerate(split_names):
            sa = visited_sets[a]
            sb = visited_sets[b]
            union = sa | sb
            if len(union) == 0:
                jaccard[i, j] = 0.0
            else:
                jaccard[i, j] = len(sa & sb) / len(union)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Plot 1: total steps
    steps_vals = [steps_by_split[s] for s in split_names]
    axes[0, 0].bar(split_names, steps_vals, edgecolor="black", alpha=0.75)
    axes[0, 0].set_title("Total Steps", fontsize=13, fontweight="bold")
    axes[0, 0].set_ylabel("# steps")
    axes[0, 0].grid(True, alpha=0.25, axis="y")

    # Plot 2: unique locations + coverage
    uniq_vals = [unique_by_split[s] for s in split_names]
    bars = axes[0, 1].bar(split_names, uniq_vals, edgecolor="black", alpha=0.75, color="tab:green")
    axes[0, 1].set_title("Unique Locations (Coverage)", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("# unique locations")
    axes[0, 1].grid(True, alpha=0.25, axis="y")
    for bar, split in zip(bars, split_names):
        cov = 100.0 * coverage_by_split[split]
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(1.0, 0.02 * max(uniq_vals) if uniq_vals else 1.0),
            f"{cov:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Plot 3: normalized entropy
    ent_vals = [entropy_by_split[s] for s in split_names]
    axes[1, 0].bar(split_names, ent_vals, edgecolor="black", alpha=0.75, color="tab:purple")
    axes[1, 0].set_title("Visitation Entropy (Normalized)", fontsize=13, fontweight="bold")
    axes[1, 0].set_ylabel("entropy (0–1)")
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].grid(True, alpha=0.25, axis="y")

    # Plot 4: overlap heatmap
    im = axes[1, 1].imshow(jaccard, vmin=0.0, vmax=1.0, cmap="Blues")
    axes[1, 1].set_title("Split Overlap (Jaccard)", fontsize=13, fontweight="bold")
    axes[1, 1].set_xticks(range(n), labels=split_names, rotation=45, ha="right")
    axes[1, 1].set_yticks(range(n), labels=split_names)
    for i in range(n):
        for j in range(n):
            axes[1, 1].text(j, i, f"{jaccard[i, j]:.2f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Jaccard")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# Batch Tensor Visualization
# ==============================================================================
def plot_batch_tensors(obs: Vector, actions: Vector, locations: Vector, figsize: Tuple[float, float] = (14, 10), max_walks: int = 5) -> plt.Figure:
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


def plot_batch_tensors_time_major(
    obs: Vector,
    actions: Vector,
    locations: Vector,
    figsize: Tuple[float, float] = (14, 10),
    max_walks: int = 5,
) -> plt.Figure:
    """Wrapper for :func:`plot_batch_tensors` for time-major batches.

    Args:
        obs: Observation tensor shaped [T, B, n_observations].
        actions: Action tensor shaped [T, B].
        locations: Location tensor shaped [T, B].
        figsize: Figure size.
        max_walks: Maximum walks to overlay.

    Returns:
        matplotlib Figure object.
    """
    if isinstance(obs, Tensor) and obs.ndim == 3:
        obs_batched = obs.transpose(0, 1)
    else:
        raise ValueError(f"Expected obs with shape [T, B, n_o], got {getattr(obs, 'shape', None)}")

    if isinstance(actions, Tensor) and actions.ndim == 2:
        actions_batched = actions.transpose(0, 1)
    else:
        raise ValueError(f"Expected actions with shape [T, B], got {getattr(actions, 'shape', None)}")

    if isinstance(locations, Tensor) and locations.ndim == 2:
        locations_batched = locations.transpose(0, 1)
    else:
        raise ValueError(f"Expected locations with shape [T, B], got {getattr(locations, 'shape', None)}")

    return plot_batch_tensors(obs_batched, actions_batched, locations_batched, figsize=figsize, max_walks=max_walks)
