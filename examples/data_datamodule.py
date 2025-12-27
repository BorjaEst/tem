#!/usr/bin/env python3
"""Data generation example with CLI configuration and visualizations.

This example demonstrates the torch_tem.data module capabilities:
- Environment loading and validation
- Policy generation (random, distance-based, Q-learning)
- Walk sampling with different strategies
- Shiny object environments
- PyTorch Lightning DataModule integration

Usage:
    python examples/data_datamodule.py --sequence-length 150 --batch-size 32 --n-train-batches 500
    python examples/data_datamodule.py --environment.width 10 --environment.height 10 --policy.type distance --policy.beta 2.0
    python examples/data_datamodule.py --help
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import config, data, figures


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for data generation example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="data_generation")

    # Environment and policy configuration
    environment: config.EnvironmentConfig = Field(default_factory=config.EnvironmentConfig, description="Environment configuration")
    policy: config.PolicyConfig = Field(default_factory=config.RandomPolicyConfig, description="Policy configuration for walk generation")

    # Sequence shape
    sequence_length: int = Field(default=100, ge=1, description="Number of timesteps per walk (T). Sole source of sequence length for the DataModule.")
    return_locations: bool = Field(default=True, description="If True, include location IDs as third element of the batch tuple.")

    # Data splits
    n_train_batches: int = Field(default=5, ge=1, le=5, description="Number of training batches per epoch (capped to keep this example quick)")
    n_val_batches: int = Field(default=1, ge=1, le=5, description="Number of validation batches (0 disables val plots)")
    n_test_batches: int = Field(default=1, ge=1, le=5, description="Number of test batches")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible environment/walk generation")

    # Dataloader settings
    batch_size: int = Field(default=2, ge=1, description="Number of walks per batch")
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader worker processes (0 = main process only)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")

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
    example_config = ExampleConfig()
    dm_config = config.DataModuleConfig.model_validate(example_config.model_dump())

    print("=" * 80)
    print("TEM DataModule Example")
    print("=" * 80)
    print(f"Grid: {dm_config.environment.width}×{dm_config.environment.height}")
    print(f"Policy: {dm_config.policy.type}")
    print(f"Sequence length (T): {dm_config.sequence_length}")
    print(f"Batch size (B): {dm_config.batch_size}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate TEMDataModule and build runtime objects
    # ------------------------------------------------------------------
    datamodule = data.TEMDataModule(dm_config)
    datamodule.setup(None)  # Use `None` so all configured splits are available.

    print("Step 1 → Environment and data generation setup complete.")
    print(f"  Environment: {datamodule.environment}")
    print(f"  Policy:      {datamodule.policy_gen}")
    print(f"  Walk Gen.:   {datamodule.walk_gen}")
    print()

    # ------------------------------------------------------------------
    # Step 2: Fetch a single time-major batch and print shapes
    # ------------------------------------------------------------------
    print("Step 2 → Inspecting a single batch (time-major tensors)")
    walks = obs, actions, locations = datamodule.sample_batch("validate")

    print(f"  observations: shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"  actions:       shape={tuple(actions.shape)}, dtype={actions.dtype}")
    print(f"  locations:     shape={tuple(locations.shape)}, dtype={locations.dtype}")
    print(f"  → Time-major contract: T={obs.shape[0]}, B={obs.shape[1]}, n_o={obs.shape[2]}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Additonal visualizations (environment, policies, walks, batch)
    # ------------------------------------------------------------------

    # Note: we intentionally do NOT plot policy comparisons here.
    # The purpose of this example is to demonstrate how the configured policy
    # (from ExampleConfig → DataModuleConfig) drives data generation.
    train_dataset = datamodule.train_dataloader().dataset
    val_dataset = datamodule.val_dataloader().dataset
    test_dataset = datamodule.test_dataloader().dataset

    batch_walk = data.Walk(observations=obs, actions=actions, locations=locations)
    env = datamodule.environment
    policy_type = dm_config.policy.type

    figs: list[tuple[str, plt.Figure]] = []
    figs.append(("01_environment_layout.png", figures.plot_environment_layout(env, title=f"Environment ({env.n_locations} locations)")))
    figs.append(("02_walk_trajectories.png", figures.plot_walks(env, [batch_walk], title=f"Val-Walks, policy={policy_type})")))
    figs.append(("03_walk_statistics.png", figures.plot_walk_statistics([batch_walk])))
    figs.append(("04_split_statistics.png", figures.plot_split_statistics(env, datamodule.datasets)))
    figs.append(("05_batch_tensors.png", figures.plot_batch_tensors_time_major(obs, actions, locations)))

    if example_config.save_plots:
        for filename, fig in figs:
            fig.savefig(example_config.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {example_config.output_dir}")

    if example_config.show_plots:
        plt.show()
    else:
        plt.close("all")
