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
from torch_tem.core.model import Model, Parameters
from torch_tem.settings import CheckpointSettings, DataSettings, LoggerSettings, ScheduleSettings, TrainerSettings


class RunSettings(BaseSettings):
    """Settings for a training run (Lightning-native)."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="run")

    # Model, data and environment settings
    model_params: Parameters = Field(default_factory=Parameters, description="Model parameters (Pydantic TEM Parameters).")
    data: DataSettings = Field(default_factory=DataSettings, description="Data generation and environment settings.")
    seed: int = Field(default=0, description="Random seed.")

    # Trainer config
    trainer: TrainerSettings = Field(default_factory=TrainerSettings, description="PyTorch Lightning Trainer kwargs.")
    schedule: ScheduleSettings = Field(default_factory=ScheduleSettings, description="Training schedules (loss weights, LR, etc).")
    logger: LoggerSettings = Field(default_factory=LoggerSettings, description="Logger settings for TensorBoard logger.")

    # Checkpoint and path settings
    checkpoint: CheckpointSettings = Field(default_factory=CheckpointSettings, description="PyTorch Lightning ModelCheckpoint kwargs.")
    root_dir: Path = Field(default=Path("./logs"), description="Root directory for Lightning outputs (default: ./lightning_logs).")
    ckpt_path: Optional[Path] = Field(default=None, description="Path to checkpoint file to resume from.")


if __name__ == "__main__":
    """Main training routine."""

    settings = RunSettings()
    seed_everything(settings.seed, workers=True)

    # Create the TEM model
    tem_model = Model(settings.model_params.model_dump())

    # Create DataModule (receives data + schedule for walk curriculum bounds)
    datamodule = training.TEMDataModule(settings.data, settings.schedule)

    # Create LightningModule (Option A: explicit dependencies, no data_settings)
    lightning_module = training.TEMLightningModule(
        tem_model=tem_model,
        schedule_settings=settings.schedule,
        trainer_settings=settings.trainer,
    )

    # Create Trainer
    trainer = Trainer(
        logger=TensorBoardLogger(**settings.logger.model_dump()),
        callbacks=[ModelCheckpoint(**settings.checkpoint.model_dump())],
        **settings.trainer.model_dump(),
    )

    # Train
    trainer.fit(
        lightning_module,
        datamodule=datamodule,
        ckpt_path=str(settings.ckpt_path) if settings.ckpt_path else None,
    )
