"""
TEM training entrypoint (PyTorch Lightning).

This module provides a Lightning-native interface for training TEM models:
- RunSettings: Pydantic settings with CLI support
- Lightning Trainer: Handles training loop, checkpointing, logging
- TEMLightningModule: Wraps TEM model
- TEMDataModule: Manages environment data generation

All paths, logging, and checkpointing are handled by Lightning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import training
from torch_tem.core.model import Parameters
from torch_tem.settings import DataSettings, ScheduleSettings


class RunSettings(BaseSettings):
    """Settings for a training run (Lightning-native)."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="run")

    # Model params
    model_params: Parameters = Field(default_factory=Parameters, description="Model parameters (Pydantic TEM Parameters).")

    # Data/environment settings
    data: DataSettings = Field(default_factory=DataSettings, description="Data generation and environment settings.")

    # Training schedule settings
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings, description="Training schedules (loss weights, LR, etc).")

    # Runtime
    seed: int = Field(default=0, description="Random seed.")

    # Lightning paths/logging
    root_dir: Optional[Path] = Field(default=None, description="Root directory for Lightning outputs (default: ./lightning_logs).")
    experiment: str = Field(default="tem", description="Experiment name for logger.")
    version: Optional[str] = Field(default=None, description="Version/run identifier (auto-increments if None).")

    # Lightning Trainer config (passed to Trainer())
    trainer: dict[str, Any] = Field(
        default_factory=lambda: {"max_steps": 20000, "log_every_n_steps": 10, "enable_progress_bar": True},
        description="PyTorch Lightning Trainer kwargs.",
    )

    # Checkpoint config (passed to ModelCheckpoint callback)
    checkpoint: dict[str, Any] = Field(
        default_factory=lambda: {"every_n_train_steps": 1000, "save_last": True},
        description="PyTorch Lightning ModelCheckpoint kwargs.",
    )

    # Resume
    ckpt_path: Optional[Path] = Field(default=None, description="Path to checkpoint file to resume from.")


def main():
    """Main training routine."""

    settings = RunSettings()
    seed_everything(settings.seed, workers=True)

    # Merge all settings into single params dict for model/datamodule/training
    params = {
        **settings.model_params.model_dump(),
        **settings.data.model_dump(),
        **settings.schedule.model_dump(),
        "max_steps": int(settings.trainer["max_steps"]),
    }

    # Setup Lightning components
    logger = TensorBoardLogger(
        save_dir=str(settings.root_dir) if settings.root_dir else "lightning_logs",
        name=settings.experiment,
        version=settings.version,
    )

    # Create Trainer
    trainer = Trainer(
        logger=logger,
        callbacks=[ModelCheckpoint(**settings.checkpoint)],
        default_root_dir=str(settings.root_dir) if settings.root_dir else None,
        **settings.trainer,
    )

    # Train
    trainer.fit(
        model=training.TEMLightningModule(params),
        datamodule=training.TEMDataModule(settings.data.envs, params),
        ckpt_path=str(settings.ckpt_path) if settings.ckpt_path else None,
    )


if __name__ == "__main__":
    main()
