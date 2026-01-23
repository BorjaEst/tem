"""TEM training entrypoint.

This module provides a minimal, Lightning-native CLI entrypoint for training the
Tolman-Eichenbaum Machine (TEM) implementation in this repository.

It wires together:

- Settings parsing via Pydantic Settings (`RunArguments`).
- Model construction (`torch_tem.core.model.TEMModel`).
- Lightning `Trainer`, logger, and checkpoint callback.
- Training loop defined in `torch_tem.training`.

The CLI is driven by `pydantic-settings` with `cli_parse_args=True`, so nested
configuration can be overridden with dot-notation arguments.

Examples:
    Run a short training session:

        python run.py --max_steps 20

    Override schedule values:

        python run.py --scheduler.memory.eta 0.6 --optimizer.lr 0.0015
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

from torch_tem import callbacks, core, data, losses, settings, training
from torch_tem.callbacks import FigureCallbackSettings, FiguresCallback
from torch_tem.data.datamodule import DataConfig
from torch_tem.data.env_validation import validate_envs_against_contract
from torch_tem.settings import CheckpointSettings, LoggerSettings, TEMSettings
from torch_tem.training import TrainerConfig

# Configure PyTorch for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")


# ============================================================================
# Settings Model
# ============================================================================
class RunArguments(BaseSettings):
    """Entry-point settings for a TEM training run.

    This settings model is designed to be used as a CLI interface via Pydantic Settings.
    It supports nested overrides via dot-notation flags (e.g., `--env.randomise_observations false`).
    All settings have sensible defaults and can be overridden via CLI arguments.

    Settings Architecture:
        This class demonstrates the settings composition pattern used throughout torch_tem:

        1. **Low-level settings** (from settings.py) are exposed as individual fields
        2. **Composite configs** (DataConfig, TrainerConfig) are built from these fields
        3. **Single source of truth**: Shared settings (e.g., walk curriculum) are defined
           once and referenced by multiple components, preventing parameter duplication

    Benefits:
        - No parameter duplication: walk_it_min, walk_it_max defined once, used by both
          data generation (DataConfig) and training (TrainerConfig)
        - Type safety: Pydantic validation at both settings and config levels
        - CLI composability: Override any nested setting via dot notation
        - Clear dependency: settings.py → module configs → run.py

    Example:
        Override walk curriculum and learning rate from CLI:
        ```bash
        python run.py --walk.walk_it_min 30 --walk.walk_it_max 250 --optimizer.lr 0.001
        ```
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="run")

    # =========================================================================
    # Core settings
    # =========================================================================
    model: TEMSettings = Field(
        default_factory=TEMSettings,
        description="Model architecture parameters.",
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility.",
    )

    # =========================================================================
    # Leaf settings (data generation)
    # =========================================================================
    space: settings.SpaceContractSettings = Field(
        default_factory=settings.SpaceContractSettings,
        description="Space contract: observation and action space dimensions.",
    )
    env: settings.EnvironmentSettings = Field(
        default_factory=settings.EnvironmentSettings,
        description="Environment generation settings.",
    )
    rollout: settings.RolloutSettings = Field(
        default_factory=settings.RolloutSettings,
        description="Batch and rollout chunking settings.",
    )
    eval: settings.EvalSettings = Field(
        default_factory=settings.EvalSettings,
        description="Validation and test dataset settings.",
    )
    exploration: settings.ExplorationSettings = Field(
        default_factory=settings.ExplorationSettings,
        description="World exploration behavior settings.",
    )
    shiny: settings.ShinySettings = Field(
        default_factory=settings.ShinySettings,
        description="Shiny environment generation settings.",
    )

    # =========================================================================
    # Leaf settings (training)
    # =========================================================================
    loss: settings.LossSettings = Field(
        default_factory=settings.LossSettings,
        description="Loss settings including weights for each component.",
    )
    optimizer: settings.OptimizerSettings = Field(
        default_factory=settings.OptimizerSettings,
        description="Optimizer settings (e.g., Adam hyperparameters).",
    )
    scheduler: settings.SchedulerSettings = Field(
        default_factory=settings.SchedulerSettings,
        description="Schedulers grouped by what they control (lr/memory/uncertainty).",
    )

    # =========================================================================
    # Shared settings (consumed by both data and training)
    # =========================================================================
    walk: settings.WalkCurriculumSettings = Field(
        default_factory=settings.WalkCurriculumSettings,
        description="Walk length curriculum settings (shared by data and training).",
    )

    # =========================================================================
    # Lightning infrastructure
    # =========================================================================
    max_steps: int = Field(
        default=20000,
        description="Maximum training steps.",
    )
    log_every_n_steps: int = Field(
        default=10,
        description="Log metrics every N steps.",
    )
    enable_progress_bar: bool = Field(
        default=True,
        description="Show progress bar during training.",
    )
    logger: LoggerSettings = Field(
        default_factory=LoggerSettings,
        description="TensorBoard logger settings.",
    )
    checkpoint: CheckpointSettings = Field(
        default_factory=CheckpointSettings,
        description="Model checkpoint settings.",
    )
    figures: FigureCallbackSettings = Field(
        default_factory=FigureCallbackSettings,
        description="Figure generation callback settings.",
    )
    ckpt_path: Optional[Path] = Field(
        default=None,
        description="Path to checkpoint file to resume from.",
    )

    # =========================================================================
    # Aggregate settings (compose leaf settings for modules)
    # =========================================================================
    @property
    def data(self) -> DataConfig:
        """Compose DataConfig from leaf settings.

        Creates the aggregate data configuration consumed by TEMDataModule.
        The walk settings are shared with trainer to maintain single source of truth.
        """
        return DataConfig(
            space=self.space,  # Shared reference
            env=self.env,
            rollout=self.rollout,
            eval=self.eval,
            exploration=self.exploration,
            shiny=self.shiny,
            walk=self.walk,  # Shared reference
        )

    @property
    def trainer(self) -> TrainerConfig:
        """Compose TrainerConfig from leaf settings and Lightning kwargs.

        Creates the aggregate training configuration consumed by TEMLightningModule.
        The walk settings are shared with data to maintain single source of truth.
        """
        return TrainerConfig(
            max_steps=self.max_steps,
            log_every_n_steps=self.log_every_n_steps,
            enable_progress_bar=self.enable_progress_bar,
            loss=self.loss,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            walk=self.walk,  # Shared reference
        )


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
    args = RunArguments()

    # Step 2: Validate environment files against space contract (fast fail)
    # This ensures all env JSON files match the expected observation/action dimensions
    # before we spend time initializing the model and trainer
    validate_envs_against_contract(args.env.envs, args.space)

    # Step 3: Seed all RNGs for deterministic training
    # workers=True ensures DataLoader workers are also seeded
    seed_everything(args.seed, workers=True)

    # Step 4: Construct the TEM model from architecture parameters and space contract
    # Use contract dimensions (not hardcoded values) for observation/action spaces
    tem_model = core.TEMModel(
        n_observations=args.space.n_observations,
        n_actions=args.space.n_actions_move,
        settings=args.model,
    )

    # Step 5: Build the PyTorch Lightning Trainer
    # This wires together logging, checkpointing, and training control
    callbacks_list = [ModelCheckpoint(**args.checkpoint.model_dump())]

    # Add FiguresCallback if enabled
    if args.figures.enabled:
        callbacks_list.append(FiguresCallback(args.figures))

    trainer = Trainer(
        # TensorBoard logger for metrics and hyperparameters
        logger=TensorBoardLogger(**args.logger.model_dump()),
        # Callbacks: checkpointing + optional figure generation
        callbacks=callbacks_list,
        # Lightning Trainer kwargs (extracted from config)
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=args.enable_progress_bar,
    )

    # Step 6: Start training
    # The LightningModule wraps the TEM model and defines the training loop
    # The DataModule generates batches of walk data on-the-fly
    trainer.fit(
        # Lightning module: training step, optimizer, schedule computation
        training.TEMLightningModule(tem_model, args.trainer),
        # Data module: generates environment walks and batches
        datamodule=data.TEMDataModule(args.data),
        # Optional: resume from checkpoint
        ckpt_path=str(args.ckpt_path) if args.ckpt_path else None,
    )
