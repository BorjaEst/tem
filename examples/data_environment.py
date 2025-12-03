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

import matplotlib.pyplot as plt
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures
from torch_tem.config import EnvironmentConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for data generation example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="data_generation")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for programmatic environment (used if env_path not provided)")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    n_walks: int = Field(default=5, ge=1, le=20, description="Number of walks to generate and visualize")
    walk_length: int = Field(default=50, ge=10, le=200, description="Steps per walk")
    repeat_bias: float = Field(default=2.0, ge=1.0, le=10.0, description="Action repeat bias for straight-line movement")

    # Policy configuration
    policy_type: Literal["random", "distance", "q_learning"] = Field(default="distance", description="Policy type for walk generation")
    beta: float = Field(default=2.0, ge=0.1, le=10.0, description="Inverse temperature for softmax policies")
    gamma: float = Field(default=0.9, ge=0.0, le=1.0, description="Discount factor for Q-learning")

    # Shiny objects
    n_shiny: int = Field(default=2, ge=0, le=10, description="Number of shiny reward objects")
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
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the data generation example with visualizations."""
    config = ExampleConfig()
    environment_config = EnvironmentConfig(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
    env = data.Environment(environment_config)
    env.validate()

    # Create policies for comparison
    policy_gen = data.PolicyGenerator(env)
    goal_location = env.n_locations - 1  # Use last location as goal
    policies = {
        "random": policy_gen.random_policy(),
        "distance": policy_gen.distance_policy(goal_location, config.beta),
        "q_learning": policy_gen.q_learning_policy(goal_location, config.gamma, config.beta, n_iterations=100),
    }

    # Generate walks
    walk_gen = data.WalkGenerator(env, repeat_bias=environment_config.explore_bias)
    policy = policies[config.policy_type]
    walks = walk_gen.generate_walks(config.n_walks, config.walk_length, policy)

    # DataModule and batch generation
    shiny_config = data.ShinyConfig.from_environment_config(environment_config, min_separation=config.shiny_separation)
    dm = data.TEMDataModule(env=env, batch_size=config.n_walks, walk_length=config.walk_length, shiny_config=shiny_config, env_config=environment_config)
    obs, actions, locations = dm.generate_batch()

    # Plot environment layout
    fig1 = figures.plot_environment_layout(env, title=f"Environment ({env.n_locations} locations)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment_layout.png", dpi=150, bbox_inches="tight")

    # Plot policy comparison
    fig2 = figures.plot_policy_comparison(env, list(policies.items()), goal_location)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_policy_comparison.png", dpi=150, bbox_inches="tight")

    # Plot walks
    fig3 = figures.plot_walks(env, walks, title=f"{config.n_walks} Walks ({config.policy_type} policy)")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_walk_trajectories.png", dpi=150, bbox_inches="tight")

    # Plot walk statistics
    fig4 = figures.plot_walk_statistics(walks)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_walk_statistics.png", dpi=150, bbox_inches="tight")

    # Plot batch tensors
    fig5 = figures.plot_batch_tensors(obs, actions, locations)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_batch_tensors.png", dpi=150, bbox_inches="tight")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
