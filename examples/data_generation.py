#!/usr/bin/env python3
"""Data generation example with CLI configuration and visualizations.

This example demonstrates the torch_tem.data module capabilities:
- Environment loading and validation
- Policy generation (random, distance-based, Q-learning)
- Walk sampling with different strategies
- Shiny object environments
- Curriculum learning
- PyTorch Lightning DataModule integration

Usage:
    python examples/data_generation.py --env-path envs/4x4.json --n-walks 5
    python examples/data_generation.py --grid-size 6 --n-shiny 3 --walk-length 100
    python examples/data_generation.py --help
"""

from pathlib import Path
from typing import Literal, Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem.data import (
    Environment,
    PolicyGenerator,
    ShinyConfig,
    TEMDataModule,
    WalkGenerator,
)


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for data generation example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="data_generation")

    # Environment configuration
    env_path: Optional[str] = Field(default=None, description="Path to environment JSON file (overrides grid generation)")
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for programmatic environment (used if env_path not provided)")
    observation_mode: Literal["unique", "shared"] = Field(default="unique", description="Observation generation mode for grids")

    # Walk generation
    n_walks: int = Field(default=5, ge=1, le=20, description="Number of walks to generate and visualize")
    walk_length: int = Field(default=50, ge=10, le=200, description="Steps per walk")
    repeat_bias: float = Field(default=2.0, ge=1.0, le=10.0, description="Action repeat bias for straight-line movement")

    # Policy configuration
    policy_type: Literal["random", "distance", "q_learning", "mixed"] = Field(default="distance", description="Policy type for walk generation")
    beta: float = Field(default=2.0, ge=0.1, le=10.0, description="Inverse temperature for softmax policies")
    gamma: float = Field(default=0.9, ge=0.0, le=1.0, description="Discount factor for Q-learning")

    # Shiny objects
    n_shiny: int = Field(default=0, ge=0, le=10, description="Number of shiny reward objects (0 disables)")
    shiny_returns: int = Field(default=5, ge=1, le=20, description="Number of returns to shiny objects")
    shiny_separation: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum separation between shiny objects (as fraction of max distance)")

    # Output
    output_dir: Path = Field(default=Path("outputs/data_generation"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_environment_layout(env: Environment, title: str = "Environment Layout"):
    """Plot the environment structure showing locations and connections."""
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get adjacency matrix
    adj = env.adjacency.numpy()
    n_locs = env.n_locations

    # Create circular layout for locations
    angles = np.linspace(0, 2 * np.pi, n_locs, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

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


def plot_policy_comparison(env: Environment, policy_gen: PolicyGenerator, goal_location: int):
    """Compare different policy types side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    policies = [
        ("Random", policy_gen.random_policy()),
        ("Distance", policy_gen.distance_policy(goal_locations=goal_location, beta=2.0)),
        ("Q-learning", policy_gen.q_learning_policy(goal_locations=goal_location, gamma=0.9, beta=2.0, n_iterations=100)),
    ]

    for ax, (name, policy) in zip(axes, policies):
        # Extract action probabilities for each location
        n_locs = env.n_locations
        n_actions = env.n_actions
        prob_matrix = np.zeros((n_locs, n_actions))

        for loc_id, location in enumerate(policy):
            for action in location.actions:
                prob_matrix[loc_id, action.index] = action.probability

        im = ax.imshow(prob_matrix.T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xlabel("Location", fontsize=12)
        ax.set_ylabel("Action", fontsize=12)
        ax.set_title(f"{name} Policy (Goal: {goal_location})", fontsize=14, fontweight="bold")
        ax.set_xticks(range(0, n_locs, max(1, n_locs // 10)))
        ax.set_yticks(range(n_actions))
        plt.colorbar(im, ax=ax, label="Probability")

    plt.tight_layout()
    return fig


def plot_walks(env: Environment, walks: list, title: str = "Generated Walks"):
    """Visualize multiple walks through the environment."""
    fig, ax = plt.subplots(figsize=(12, 8))

    n_locs = env.n_locations

    # Create circular layout
    angles = np.linspace(0, 2 * np.pi, n_locs, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    # Plot connections (light gray)
    adj = env.adjacency.numpy()
    for i in range(n_locs):
        for j in range(n_locs):
            if adj[i, j] > 0:
                ax.plot([x[i], x[j]], [y[i], y[j]], "gray", alpha=0.1, linewidth=1, zorder=1)

    # Plot each walk with different color
    colors = plt.cm.tab10(np.linspace(0, 1, len(walks)))

    for walk_idx, walk in enumerate(walks):
        locs = walk.locations
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


def plot_walk_statistics(walks: list):
    """Plot statistics about generated walks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Extract data
    walk_lengths = [len(w.locations) for w in walks]
    location_visits = []
    action_sequences = []

    for walk in walks:
        location_visits.extend(walk.locations)
        action_sequences.extend(walk.actions)

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
        current_repeat = 1
        for i in range(1, len(walk.actions)):
            if walk.actions[i] == walk.actions[i - 1]:
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


def plot_batch_tensors(obs: torch.Tensor, actions: torch.Tensor, locations: torch.Tensor):
    """Visualize batch tensor shapes and samples."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    batch_size, walk_length, n_obs = obs.shape

    # Plot 1: Observation heatmap (first walk in batch)
    im1 = axes[0, 0].imshow(obs[0].T.numpy(), aspect="auto", cmap="viridis")
    axes[0, 0].set_xlabel("Time Step", fontsize=12)
    axes[0, 0].set_ylabel("Observation ID", fontsize=12)
    axes[0, 0].set_title(f"Observations (Walk 0)\nShape: {tuple(obs.shape)}", fontsize=13, fontweight="bold")
    plt.colorbar(im1, ax=axes[0, 0], label="Value")

    # Plot 2: Action sequence (all walks overlayed)
    for i in range(min(5, batch_size)):
        axes[0, 1].plot(actions[i].numpy(), alpha=0.6, label=f"Walk {i}")
    axes[0, 1].set_xlabel("Time Step", fontsize=12)
    axes[0, 1].set_ylabel("Action ID", fontsize=12)
    axes[0, 1].set_title(f"Action Sequences\nShape: {tuple(actions.shape)}", fontsize=13, fontweight="bold")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Location trajectory (all walks)
    for i in range(min(5, batch_size)):
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


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the data generation example with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("TEM Data Generation Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    for field, value in config.model_dump().items():
        print(f"  {field}: {value}")
    print()

    # Step 1: Load or create environment
    print("Step 1: Loading environment...")
    if config.env_path:
        env = Environment(config.env_path)
        print(f"  Loaded from: {config.env_path}")
    else:
        env = Environment.from_grid(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
        print(f"  Generated {config.grid_size}x{config.grid_size} grid")

    print(f"  Locations: {env.n_locations}")
    print(f"  Observations: {env.n_observations}")
    print(f"  Actions: {env.n_actions}")
    env.validate()
    print("  ✓ Environment validated")

    # Plot environment layout
    fig1 = plot_environment_layout(env, title=f"Environment ({env.n_locations} locations)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment_layout.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_environment_layout.png'}")

    # Step 2: Policy comparison
    print("\nStep 2: Generating policies...")
    policy_gen = PolicyGenerator(env)
    goal_location = env.n_locations - 1  # Use last location as goal

    fig2 = plot_policy_comparison(env, policy_gen, goal_location)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_policy_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_policy_comparison.png'}")

    # Step 3: Generate walks
    print(f"\nStep 3: Generating {config.n_walks} walks...")
    walk_gen = WalkGenerator(env, repeat_bias=config.repeat_bias)

    # Select policy based on config
    if config.policy_type == "random":
        policy = policy_gen.random_policy()
    elif config.policy_type == "distance":
        policy = policy_gen.distance_policy(goal_locations=goal_location, beta=config.beta)
    elif config.policy_type == "q_learning":
        policy = policy_gen.q_learning_policy(goal_locations=goal_location, gamma=config.gamma, beta=config.beta, n_iterations=100)
    elif config.policy_type == "mixed":
        random_pol = policy_gen.random_policy()
        distance_pol = policy_gen.distance_policy(goal_locations=goal_location, beta=config.beta)
        policy = policy_gen.mix_policies([random_pol, distance_pol], [0.3, 0.7])

    walks = walk_gen.generate_walks(config.n_walks, config.walk_length, policy)
    print(f"  Generated {len(walks)} walks of length {config.walk_length}")
    print(f"  Policy: {config.policy_type}")

    # Plot walks
    fig3 = plot_walks(env, walks, title=f"{config.n_walks} Walks ({config.policy_type} policy)")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_walk_trajectories.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_walk_trajectories.png'}")

    # Plot walk statistics
    fig4 = plot_walk_statistics(walks)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_walk_statistics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '04_walk_statistics.png'}")

    # Step 4: DataModule and batch generation
    print("\nStep 4: PyTorch Lightning DataModule...")

    # Setup shiny config if requested
    shiny_config = None
    if config.n_shiny > 0:
        shiny_config = ShinyConfig(n=config.n_shiny, returns=config.shiny_returns, min_separation=config.shiny_separation, gamma=config.gamma, beta=config.beta)
        print(f"  Shiny objects: {config.n_shiny}")

    dm = TEMDataModule(env_spec=env, batch_size=config.n_walks, walk_length=config.walk_length, shiny_config=shiny_config, repeat_bias=config.repeat_bias)

    if shiny_config:
        print(f"  Shiny locations: {dm.shiny_locations}")

    obs, actions, locations = dm.generate_batch()
    print(f"  Batch shapes:")
    print(f"    observations: {tuple(obs.shape)}")
    print(f"    actions: {tuple(actions.shape)}")
    print(f"    locations: {tuple(locations.shape)}")

    # Plot batch tensors
    fig5 = plot_batch_tensors(obs, actions, locations)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_batch_tensors.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '05_batch_tensors.png'}")

    # Final summary
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print(f"Output directory: {config.output_dir}")
    print("=" * 80)

    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
