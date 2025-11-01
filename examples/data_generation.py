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
        env = data.Environment(config.env_path)
        print(f"  Loaded from: {config.env_path}")
    else:
        env = data.Environment.from_grid(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
        print(f"  Generated {config.grid_size}x{config.grid_size} grid")

    print(f"  Locations: {env.n_locations}")
    print(f"  Observations: {env.n_observations}")
    print(f"  Actions: {env.n_actions}")
    env.validate()
    print("  ✓ Environment validated")

    # Plot environment layout
    fig1 = figures.plot_environment_layout(env, title=f"Environment ({env.n_locations} locations)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment_layout.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_environment_layout.png'}")

    # Step 2: Policy comparison
    print("\nStep 2: Generating policies...")
    policy_gen = figures.PolicyGenerator(env)
    goal_location = env.n_locations - 1  # Use last location as goal

    # Generate policies for comparison
    policies = [
        ("Random", policy_gen.random_policy()),
        ("Distance", policy_gen.distance_policy(goal_locations=goal_location, beta=2.0)),
        ("Q-learning", policy_gen.q_learning_policy(goal_locations=goal_location, gamma=0.9, beta=2.0, n_iterations=100)),
    ]

    fig2 = figures.plot_policy_comparison(env, policies, goal_location)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_policy_comparison.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_policy_comparison.png'}")

    # Step 3: Generate walks
    print(f"\nStep 3: Generating {config.n_walks} walks...")
    walk_gen = data.WalkGenerator(env, repeat_bias=config.repeat_bias)

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
    fig3 = figures.plot_walks(env, walks, title=f"{config.n_walks} Walks ({config.policy_type} policy)")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_walk_trajectories.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_walk_trajectories.png'}")

    # Plot walk statistics
    fig4 = figures.plot_walk_statistics(walks)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_walk_statistics.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '04_walk_statistics.png'}")

    # Step 4: DataModule and batch generation
    print("\nStep 4: PyTorch Lightning DataModule...")

    # Setup shiny config if requested
    shiny_config = None
    if config.n_shiny > 0:
        shiny_config = data.ShinyConfig(n=config.n_shiny, returns=config.shiny_returns, min_separation=config.shiny_separation, gamma=config.gamma, beta=config.beta)
        print(f"  Shiny objects: {config.n_shiny}")

    dm = data.TEMDataModule(env_spec=env, batch_size=config.n_walks, walk_length=config.walk_length, shiny_config=shiny_config, repeat_bias=config.repeat_bias)

    if shiny_config:
        print(f"  Shiny locations: {dm.shiny_locations}")

    obs, actions, locations = dm.generate_batch()
    print(f"  Batch shapes:")
    print(f"    observations: {tuple(obs.shape)}")
    print(f"    actions: {tuple(actions.shape)}")
    print(f"    locations: {tuple(locations.shape)}")

    # Plot batch tensors
    fig5 = figures.plot_batch_tensors(obs, actions, locations)
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
