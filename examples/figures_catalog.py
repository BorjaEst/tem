#!/usr/bin/env python3
"""Render a catalog of TEM figures from a rollout trace.

This example builds a DataModule and TEM model, collects a rollout trace, and
then renders a configurable set of built-in figures for quick inspection.

Examples:
    # Full catalog (interactive).
    python examples/figures_catalog.py

    # Headless run (no display).
    python examples/figures_catalog.py --show_plots false --save_plots true

    # Focus on model diagnostics only.
    python examples/figures_catalog.py --figure_profile diagnostics
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures
from torch_tem.data.datamodule import DataConfig
from torch_tem.diagnostics.traces import RolloutTrace, WorldTrace
from torch_tem.figures.register import register_builtin_figures
from torch_tem.figures.registry import REGISTRY, FigureContext
from torch_tem.model import Model as TEMModel
from torch_tem.model import TEMConfig
from torch_tem.settings import (
    AutoencoderSettings,
    CurriculumSettings,
    EnvironmentSettings,
    EnvSamplingSettings,
    HPCSettings,
    LECProjectionSettings,
    LECSettings,
    MECProjectionSettings,
    MECSettings,
    RolloutStreamSettings,
    SpaceContractSettings,
)

NAME = __file__.split("/")[-1].replace(".py", "")
logger = logging.getLogger(NAME)


class ExampleArguments(BaseSettings):
    """CLI configuration for the figure catalog example."""

    model_config = SettingsConfigDict(
        extra="forbid",
        cli_parse_args=True,
        cli_prog_name=NAME,
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
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

    # Model configuration
    autoencoder: AutoencoderSettings = Field(
        default_factory=AutoencoderSettings,
        description="Autoencoder (observation embedding) settings.",
    )
    lec: LECSettings = Field(
        default_factory=LECSettings,
        description="LEC (sensory features) settings.",
    )
    lec_proj: LECProjectionSettings = Field(
        default_factory=LECProjectionSettings,
        description="LEC projection settings.",
    )
    mec: MECSettings = Field(
        default_factory=MECSettings,
        description="MEC (abstract location) settings.",
    )
    mec_proj: MECProjectionSettings = Field(
        default_factory=MECProjectionSettings,
        description="MEC projection settings.",
    )
    hpc: HPCSettings = Field(
        default_factory=HPCSettings,
        description="HPC (grounded location memory) settings.",
    )

    # Checkpoint
    checkpoint: Path | None = Field(
        default=None,
        description="Path to model checkpoint (optional).",
    )

    # Rollout settings
    max_rollout_steps: int = Field(
        default=100,
        ge=10,
        description="Maximum rollout steps to collect.",
    )
    downsample_stride: int = Field(
        default=1,
        ge=1,
        description="Downsampling stride for trace (1 = keep all steps).",
    )

    # Figure selection
    figure_profile: str = Field(
        default="full",
        description="Profile: full, world, model, diagnostics, or cells.",
    )
    figure_names: list[str] = Field(
        default_factory=list,
        description="Explicit figure names (overrides figure_profile).",
    )
    env_idx: int = Field(
        default=0,
        ge=0,
        description="Environment index to visualize.",
    )
    freq_idx: int = Field(
        default=0,
        ge=0,
        description="Frequency module index to visualize.",
    )

    # Output
    output_dir: Path = Field(
        default=Path("outputs/figure_catalog"),
        description="Directory for saving plots.",
    )
    show_plots: bool = Field(
        default=True,
        description="Display plots interactively.",
    )
    save_plots: bool = Field(
        default=True,
        description="Save plots to output directory.",
    )

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output_dir if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def data(self) -> DataConfig:
        """Create the aggregate DataConfig consumed by the DataModule."""
        return DataConfig.model_validate(self, from_attributes=True)

    @property
    def model(self) -> TEMConfig:
        """Create the aggregate TEMConfig for model construction."""
        return TEMConfig.model_validate(self, from_attributes=True)

    def resolved_figures(self) -> list[str]:
        """Resolve the figure names to render based on the profile."""
        # Prefer explicit names when provided.
        if self.figure_names:
            return self.figure_names

        profiles = {
            "world": [
                "environment.layout",
                "walk.trajectories",
                "walk.statistics",
                "split.statistics",
                "coverage.occupancy_map",
                "coverage.action_bias_map",
            ],
            "model": [
                "overview",
                "overview.rate_maps",
                "cells.place_rate_maps",
                "cells.place_rate_maps.frequencies",
                "cells.place_rate_maps.pathways",
            ],
            "diagnostics": [
                "decoding.location_error_map",
                "dynamics.path_integration_drift",
                "dynamics.sequence_consistency",
                "representation.freq_similarity",
                "memory.retrieval_error_by_location",
                "uncertainty.calibration",
                "cells.place_field_summary",
                "cells.remapping_correlation",
                "cells.spatial_autocorrelogram",
            ],
            "cells": [
                "cells.place_rate_maps",
                "cells.place_rate_maps.frequencies",
                "cells.place_rate_maps.pathways",
                "cells.place_field_summary",
                "cells.remapping_correlation",
                "cells.spatial_autocorrelogram",
            ],
        }

        if self.figure_profile == "full":
            names = []
            for key in ["world", "model", "diagnostics"]:
                names.extend(profiles[key])
            return names

        if self.figure_profile not in profiles:
            raise ValueError(f"Unknown profile '{self.figure_profile}'")
        return profiles[self.figure_profile]


def _build_rollout_trace(args: ExampleArguments) -> RolloutTrace:
    """Create a rollout trace from a DataModule and TEM model."""
    # Build the datamodule and model to generate a rollout trace.
    datamodule = data.DataModule(args.data)
    datamodule.setup(None)

    model = TEMModel(args.model)
    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        state_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
    model.eval()

    dataset = datamodule.get_dataset("validate")
    trace = RolloutTrace.from_batch(
        batch=datamodule.sample_batch(split="validate"),
        environments=dataset.environments,
        model=model,
        stop=args.max_rollout_steps,
        meta={"split": "validate"},
    ).downsample_time(args.downsample_stride)
    return trace


def _resolve_trace(spec, rollout: RolloutTrace, world: WorldTrace):
    """Pick the correct trace for a figure spec."""
    # Match registry trace requirements to available traces.
    if issubclass(spec.accepts, RolloutTrace):
        return rollout
    if issubclass(spec.accepts, WorldTrace):
        return world
    return None


def main() -> None:
    """Run the figure catalog example."""
    # Parse CLI arguments and render the selected figure catalog.
    args = ExampleArguments()
    logging.basicConfig(format="%(levelname)s:%(message)s", level=args.log_level.upper())

    print("=" * 80)
    print("TEM Figure Catalog")
    print("=" * 80)
    print(f" - Log level: {args.log_level.upper()}")
    print(f" - Checkpoint: {args.checkpoint or 'None (random init)'}")
    print(f" - Output directory: {args.output_dir}")
    print(f" - Save plots: {args.save_plots}")
    print(f" - Show plots: {args.show_plots}")
    print(f" - Figure profile: {args.figure_profile}")
    print()

    register_builtin_figures()
    names = args.resolved_figures()
    REGISTRY.validate(names)

    rollout = _build_rollout_trace(args)
    world_trace = rollout.world_step
    ctx = FigureContext(
        env_idx=args.env_idx,
        freq_idx=args.freq_idx,
        figsize=(14, 10),
        split_name="validate",
    )

    saved = 0
    for index, name in enumerate(names, start=1):
        spec = REGISTRY.get(name)
        trace = _resolve_trace(spec, rollout, world_trace)
        if trace is None:
            logger.warning("Skipping %s (unsupported trace type)", name)
            continue

        fig = spec.plot(trace, ctx)
        if args.save_plots:
            filename = f"{index:02d}_{name.replace('.', '_')}.png"
            fig.savefig(args.output_dir / filename, dpi=150, bbox_inches="tight")
            saved += 1
        if not args.show_plots:
            plt.close(fig)

    if args.save_plots:
        print(f"Saved {saved} figure(s) to: {args.output_dir}")
    plt.show() if args.show_plots else plt.close("all")
    print("Catalog completed.")


if __name__ == "__main__":
    main()
