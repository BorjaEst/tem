"""TEM training entrypoint.

This module provides a minimal, Lightning-native CLI entrypoint for training the
Tolman-Eichenbaum Machine (TEM) implementation in this repository.

It wires together:

- Settings parsing via Pydantic Settings (`RunSettings`).
- Model construction (`torch_tem.core.model.Model`).
- Lightning `Trainer`, logger, and checkpoint callback.
- Training loop defined in `torch_tem.training`.

The CLI is driven by `pydantic-settings` with `cli_parse_args=True`, so nested
configuration can be overridden with dot-notation arguments.

Examples:
    Run a short training session:

        python run.py --trainer.max_steps 20

    Override schedule values:

        python run.py --schedule.eta 0.6 --schedule.lr_max 0.0015
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import core, data, training
from torch_tem.core import Parameters
from torch_tem.settings import CheckpointSettings, DataSettings, LoggerSettings, ScheduleSettings, TrainerSettings

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")


# ============================================================================
# Settings Model
# ============================================================================
class RunSettings(BaseSettings):
    """Settings for a TEM training run.

    This settings model is designed to be used as a CLI interface via Pydantic Settings.
    It supports nested overrides via dot-notation flags (e.g., `--trainer.max_steps 1000`).
    All settings have sensible defaults and can be overridden via CLI arguments or
    environment variables prefixed with the setting path.
    """

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


# ============================================================================
# Main Entrypoint
# ============================================================================
if __name__ == "__main__":
    """Run a TEM training session.

    This is the main entrypoint for TEM training. It orchestrates the following steps:

    1. Parses configuration from CLI arguments and environment variables using Pydantic Settings.
    2. Seeds all random number generators (PyTorch, NumPy, Python random) for reproducibility.
    3. Constructs the TEM neural network model from architecture parameters.
    4. Creates a PyTorch Lightning Trainer with TensorBoard logging and checkpointing.
    5. Starts training via `Trainer.fit()` with the Lightning DataModule and LightningModule.

    The training loop will run until `trainer.max_steps` is reached. Logs are written to
    `logger.save_dir`, and checkpoints are saved every `checkpoint.every_n_train_steps`.

    Raises:
        FileNotFoundError: If `ckpt_path` is specified but does not exist.
        ValidationError: If settings validation fails (e.g., invalid parameter values).
    """
    # Step 1: Parse all settings from CLI and environment
    # Pydantic Settings will automatically parse sys.argv when cli_parse_args=True
    settings = RunSettings()

    # Step 2: Seed all RNGs for deterministic training
    # workers=True ensures DataLoader workers are also seeded
    seed_everything(settings.seed, workers=True)

    # Step 3: Construct the TEM model from architecture parameters
    # model_dump() converts the Pydantic Parameters model to a plain dict
    tem_model = core.Model(settings.model_params.model_dump())

    # Step 4: Build the PyTorch Lightning Trainer
    # This wires together logging, checkpointing, and training control
    trainer = Trainer(
        # TensorBoard logger for metrics and hyperparameters
        logger=TensorBoardLogger(**settings.logger.model_dump()),
        # Checkpoint callback to save model state periodically
        callbacks=[ModelCheckpoint(**settings.checkpoint.model_dump())],
        # Trainer settings (max_steps, log_every_n_steps, etc.)
        **settings.trainer.model_dump(),
    )

    # Step 5: Start training
    # The LightningModule wraps the TEM model and defines the training loop
    # The DataModule generates batches of walk data on-the-fly
    trainer.fit(
        # Lightning module: training step, optimizer, schedule computation
        training.TEMLightningModule(tem_model, settings.schedule, settings.trainer),
        # Data module: generates environment walks and batches
        datamodule=data.TEMDataModule(settings.data, settings.schedule),
        # Optional: resume from checkpoint
        ckpt_path=str(settings.ckpt_path) if settings.ckpt_path else None,
    )
