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
from torch_tem.data.walks import Walk, WalkDataset


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
    n_val_batches: int = Field(default=1, ge=0, le=5, description="Number of validation batches (0 disables val plots)")
    n_test_batches: int = Field(default=0, ge=0, le=5, description="Number of test batches")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible environment/walk generation")

    # Dataloader settings
    batch_size: int = Field(default=16, ge=1, description="Number of walks per batch")
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader worker processes (0 = main process only)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")

    # Output
    output_dir: Path = Field(default=Path("outputs/data_generation"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    # Demo controls
    n_demo_walks: int = Field(default=2, ge=1, description="Number of single walks to generate for trajectory plots")

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
    cfg = ExampleConfig()

    # DataModuleConfig is strict about unknown fields; ExampleConfig also contains
    # plotting-only settings (e.g., output_dir). Filter to the DataModule surface.
    cfg_dict = cfg.model_dump()
    dm_payload = {k: v for k, v in cfg_dict.items() if k in config.DataModuleConfig.model_fields}
    dm_cfg = config.DataModuleConfig.model_validate(dm_payload)

    print("=" * 80)
    print("TEM DataModule Example")
    print("=" * 80)
    print(f"Grid: {dm_cfg.environment.width}×{dm_cfg.environment.height}")
    print(f"Policy: {dm_cfg.policy.type}")
    print(f"Sequence length (T): {dm_cfg.sequence_length}")
    print(f"Batch size (B): {dm_cfg.batch_size}")
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate TEMDataModule and build runtime objects
    # ------------------------------------------------------------------
    datamodule = data.TEMDataModule(dm_cfg)
    # Use `None` so all configured splits are available.
    datamodule.setup(None)

    env = datamodule.environment
    policy_gen = datamodule.policy_gen
    walk_gen = datamodule.walk_gen
    if env is None or policy_gen is None or walk_gen is None:
        raise RuntimeError("TEMDataModule.setup() did not initialize required components")

    # ------------------------------------------------------------------
    # Step 2: Fetch a single time-major batch and print shapes
    # ------------------------------------------------------------------
    print("Step 2 → Inspecting a single batch (time-major tensors)")
    train_loader = datamodule.train_dataloader()
    obs, actions, locations = next(iter(train_loader))

    print(f"  observations: shape={tuple(obs.shape)}, dtype={obs.dtype}")
    print(f"  actions:       shape={tuple(actions.shape)}, dtype={actions.dtype}")
    print(f"  locations:     shape={tuple(locations.shape)}, dtype={locations.dtype}")
    print(f"  → Time-major contract: T={obs.shape[0]}, B={obs.shape[1]}, n_x={obs.shape[2]}")
    print()

    # ------------------------------------------------------------------
    # Step 3: Optional visualizations (environment, policies, walks, batch)
    # ------------------------------------------------------------------
    figs: list[tuple[str, plt.Figure]] = []
    figs.append(("01_environment_layout.png", figures.plot_environment_layout(env, title=f"Environment ({env.n_locations} locations)")))

    # Note: we intentionally do NOT plot policy comparisons here.
    # The purpose of this example is to demonstrate how the configured policy
    # (from ExampleConfig → DataModuleConfig) drives data generation.
    policy_cfg = dm_cfg.policy

    # Sample a few walks directly from the same dataset configuration so we can
    # visualize trajectories and statistics.
    walk_dataset = WalkDataset(n_items=cfg.n_demo_walks, env=env, policy_gen=policy_gen, walk_gen=walk_gen, params=dm_cfg)
    walks: list[Walk] = []
    for i in range(cfg.n_demo_walks):
        obs_i, act_i, loc_i = walk_dataset[i]
        walks.append(Walk(observations=obs_i, actions=act_i, locations=loc_i))

    def collect_split_locations(loader, n_batches: int) -> list:
        collected = []
        if n_batches <= 0:
            return collected
        for _, batch in zip(range(n_batches), loader):
            _, _, locs_b = batch
            collected.append(locs_b)
        return collected

    train_locations = collect_split_locations(datamodule.train_dataloader(), dm_cfg.n_train_batches)
    val_locations_list = collect_split_locations(datamodule.val_dataloader(), dm_cfg.n_val_batches)
    test_locations_list = collect_split_locations(datamodule.test_dataloader(), dm_cfg.n_test_batches)

    figs.append(("02_walk_trajectories.png", figures.plot_walks(env, walks, title=f"{cfg.n_demo_walks} Walks (policy={policy_cfg.type})")))
    figs.append(("03_walk_statistics.png", figures.plot_walk_statistics(walks)))

    split_locations = {"train": train_locations}
    if dm_cfg.n_val_batches > 0:
        split_locations["val"] = val_locations_list
    if dm_cfg.n_test_batches > 0:
        split_locations["test"] = test_locations_list
    figs.append(("04_split_visit_statistics.png", figures.plot_split_location_visit_statistics(env, split_locations)))

    # Batch tensors are time-major [T, B, ...] from the DataModule.
    figs.append(("05_train_batch_tensors.png", figures.plot_batch_tensors_time_major(obs, actions, locations)))

    # Optional: visualize a single validation batch if configured.
    if dm_cfg.n_val_batches > 0:
        val_loader = datamodule.val_dataloader()
        val_obs, val_actions, val_locations = next(iter(val_loader))
        figs.append(("06_val_batch_tensors.png", figures.plot_batch_tensors_time_major(val_obs, val_actions, val_locations)))

    if cfg.save_plots:
        for filename, fig in figs:
            fig.savefig(cfg.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {cfg.output_dir}")

    if cfg.show_plots:
        plt.show()
    else:
        plt.close("all")
