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

from torch_tem import data, figures
from torch_tem.data.datamodule import DataConfig
from torch_tem.settings import CurriculumSettings, EnvironmentSettings, EnvSamplingSettings, RolloutStreamSettings, SpaceContractSettings


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleArguments(BaseSettings):
    """Configuration for data generation example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="data_generation")

    # General settings
    drop_last: bool = Field(
        default=False,
        description="Drop last incomplete batch",
    )

    # Data configuration (leaves)
    space: SpaceContractSettings = Field(
        default_factory=SpaceContractSettings,
        description="Space contract: observation and action space dimensions.",
    )
    env: EnvironmentSettings = Field(
        default_factory=EnvironmentSettings,
        description="Environment generation settings.",
    )
    iterator: RolloutStreamSettings = Field(
        default_factory=RolloutStreamSettings,
        description="Iterator protocol settings (rollout chunking + eval protocol).",
    )
    policy: EnvSamplingSettings = Field(
        default_factory=EnvSamplingSettings,
        description="Data generation policies (exploration + shiny).",
    )
    walk: CurriculumSettings = Field(
        default_factory=CurriculumSettings,
        description="Walk length curriculum settings.",
    )

    # Output
    output_dir: Path = Field(
        default=Path("outputs/data_generation"),
        description="Directory for saving plots",
    )
    show_plots: bool = Field(
        default=True,
        description="Display plots interactively",
    )
    save_plots: bool = Field(
        default=True,
        description="Save plots to output directory",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def data(self) -> DataConfig:
        """Compose DataConfig from leaf settings.

        Creates the aggregate data configuration consumed by DataModule.
        The walk settings are shared with trainer to maintain single source of truth.
        """
        return DataConfig(
            space=self.space,
            env=self.env,
            iterator=self.iterator,
            policy=self.policy,
            walk=self.walk,
        )


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the data generation example with visualizations."""

    # Step 1: Parse all settings from CLI and environment
    # Pydantic Settings will automatically parse sys.argv when cli_parse_args=True
    args = ExampleArguments()

    print("=" * 80)
    print("TEM DataModule Example")
    print("=" * 80)
    # TODO: Add more detailed argument summaries
    print()

    # ------------------------------------------------------------------
    # Step 1: Instantiate TEMDataModule and build runtime objects
    # ------------------------------------------------------------------
    datamodule = data.DataModule(args.data)
    datamodule.setup(None)  # Use `None` so all configured splits are available.

    print("Step 1 → Environment and data generation setup complete.")
    # TODO: Add more detailed datamodule summaries
    print()

    # ------------------------------------------------------------------
    # Step 2: Fetch a single time-major batch and print shapes
    # ------------------------------------------------------------------
    walk, visited = datamodule.sample_batch("validate")
    locations, observation_0, action_0 = walk[0]

    print("Step 2 → Inspecting a single batch (time-major tensors)")
    # TODO: Add more detailed datamodule summaries
    print()

    # ------------------------------------------------------------------
    # Step 3: Additonal visualizations (environment, policies, walks, batch)
    # ------------------------------------------------------------------
    figs: list[tuple[str, plt.Figure]] = []
    figs.append(("01_environment_layout.png", figures.environment.layout.plot(...)))  # TODO: complete inputs
    figs.append(("02_walk_trajectories.png", figures.walk.trajectories.plot(...)))  # TODO: complete inputs
    figs.append(("03_walk_statistics.png", figures.walk.statistics.plot(...)))  # TODO: complete inputs
    figs.append(("04_split_statistics.png", figures.split.statistics.plot(...)))  # TODO: complete inputs

    print("Step 3 → Generated visualizations.")
    # TODO: Add more detailed figure summaries
    print()

    # ------------------------------------------------------------------
    # Step 4: Save and/or show plots
    # ------------------------------------------------------------------

    # Save and/or show plots as per user configuration
    if args.save_plots:
        for filename, fig in figs:
            fig.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
        print(f"Saved {len(figs)} figure(s) to: {args.output_dir}")
    plt.show() if args.show_plots else plt.close("all")

    print("Example complete.")
    print("=" * 80)
