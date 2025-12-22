"""Experiment 1: TEM Training with TensorBoard Logging.

This experiment trains the Tolman-Eichenbaum Machine (TEM) model on a gridworld
environment with configurable architecture and training parameters. All metrics
and training progress are logged to TensorBoard for visualization.

The experiment pipeline consists of:
    1. Environment and data setup (gridworld configuration)
    2. Model initialization (TEM architecture with multiple frequency modules)
    3. Training configuration (learning rates, loss weights, walk parameters)
    4. Trainer setup (PyTorch Lightning with TensorBoard logging)
    5. Training execution with automatic checkpointing

Usage:
    python experiments/experiment_1.py [OPTIONS]

Examples:
    # Run with defaults (10x10 grid, 100 steps)
    python experiments/experiment_1.py

    # Longer training on larger grid
    python experiments/experiment_1.py --grid_size 15 --max_steps 5000

    # View logs
    tensorboard --logdir logs
"""

from pathlib import Path
from typing import List, Literal

import lightning as L
import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures
from torch_tem.config import EnvironmentConfig, ModelConfig, TrainingConfig
from torch_tem.core.model import TEMModel
from torch_tem.data import TEMDataModule
from torch_tem.training import TEMLightningModule


# ==============================================================================
# Configuration
# ==============================================================================
class ExperimentConfig(BaseSettings):

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name=Path(__file__).stem)

    # Environment configuration
    grid_size: int = Field(default=10, ge=5, le=15, description="Grid size for square environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation assignment strategy")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01], description="Initial frequencies for each spatial module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [10, 10, 8, 6, 6], description="Grid cells per frequency module")
    n_x_c: int = Field(default=15, ge=2, le=30, description="Compressed sensory dimension (two-hot)")

    # Training configuration
    max_steps: int = Field(default=100, ge=100, le=50000, description="Maximum training steps")
    batch_size: int = Field(default=4, ge=1, le=64, description="Batch size for training")
    n_rollout: int = Field(default=20, ge=5, le=100, description="BPTT rollout length (steps per backward pass)")

    # Learning rate configuration
    lr_max: float = Field(default=9.4e-4, gt=0, description="Maximum learning rate")
    lr_decay_rate: float = Field(default=0.5, gt=0, le=1, description="Learning rate decay factor")
    lr_decay_steps: int = Field(default=400, ge=1, description="Steps between LR decay")

    # Walk generation
    walk_length_min: int = Field(default=25, ge=10, le=100, description="Minimum walk length")
    walk_length_max: int = Field(default=300, ge=50, le=1000, description="Maximum walk length")

    # Loss weights
    loss_weights_x: float = Field(default=1.0, ge=0, description="Weight for sensory loss")
    loss_weights_p: float = Field(default=1.0, ge=0, description="Weight for grounded location loss")
    loss_weights_g: float = Field(default=1.0, ge=0, description="Weight for abstract location loss")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Shiny objects (optional reward-based learning)
    shiny_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability of shiny objects in environment")

    # Logging and output
    checkpoint_dir: Path = Field(default=Path("checkpoints"), description="Directory for model checkpoints")
    output_dir: Path = Field(default=Path("outputs"), description="Directory for plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    # Debugging and monitoring
    log_every_n_steps: int = Field(default=12, ge=1, description="Logging frequency")
    val_check_interval: int = Field(default=12, ge=1, description="Validation check interval (number of batches)")

    @field_validator("checkpoint_dir", "output_dir")
    @classmethod
    def create_dir(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Main Training Script
# ==============================================================================
if __name__ == "__main__":
    """Run complete TEM training pipeline with visualizations."""
    config = ExperimentConfig()
    config_kwargs = config.model_dump(mode="python")

    print("=" * 80)
    print("TEM Training Example")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode})")
    print(f"  Architecture: {len(config.f_initial)} frequency modules")
    print(f"  Grid cells: {config.n_g_subsampled}")
    print(f"  Frequencies: {config.f_initial}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Max steps: {config.max_steps}")
    print(f"  BPTT rollout: {config.n_rollout}")
    print(f"  Learning rate: {config.lr_max}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Data Setup
    # =========================================================================
    print("Phase 1: Setting up environment and data generation...")

    # Create environment configuration
    env_config = EnvironmentConfig.model_validate(config.model_dump(), extra="ignore")
    env = data.Environment(env_config)
    env.validate()
    print(f"  Environment: {env.n_locations} locations, {env.n_observations} observations")

    # Create data module (full-walk, time-major batches)
    dm_config = DataModuleConfig(
        environment=env_config,
        batch_size=config.batch_size,
        sequence_length=(config.walk_length_min + config.walk_length_max) // 2,  # Keep prior behavior
    )
    datamodule = TEMDataModule(dm_config, env=env)
    print(f"  DataModule: batch_size={dm_config.batch_size}, walk_length={dm_config.sequence_length}")

    # =========================================================================
    # PHASE 2: Model Initialization
    # =========================================================================
    print("\nPhase 2: Initializing TEM model...")

    # Calculate total actions
    total_actions = env_config.n_actions + (1 if env_config.has_static_action else 0)

    # Create model configuration
    model_config = ModelConfig(
        n_x=env.n_observations,
        n_x_c=config.n_x_c,
        n_g_subsampled=config.n_g_subsampled,
        f_initial=config.f_initial,
        n_actions=total_actions,
        eta=config.eta,
        kappa=config.kappa,
        batch_size=config.batch_size,
    )

    # Create TEM model
    tem_model = TEMModel(model_config)
    print(f"  Model architecture:")
    print(f"    - Observations: {model_config.n_x}")
    print(f"    - Compressed sensory: {model_config.n_x_c}")
    print(f"    - Grid cells (subsampled): {model_config.n_g_subsampled}")
    print(f"    - Place cells: {model_config.n_p}")
    print(f"    - Actions: {total_actions}")

    # =========================================================================
    # PHASE 3: Training Configuration
    # =========================================================================
    print("\nPhase 3: Configuring training...")

    # Create training configuration
    training_config = TrainingConfig(
        n_rollout=config.n_rollout,
        lr_max=config.lr_max,
        lr_decay_rate=config.lr_decay_rate,
        lr_decay_steps=config.lr_decay_steps,
        loss_weights_x=config.loss_weights_x,
        loss_weights_p=config.loss_weights_p,
        loss_weights_g=config.loss_weights_g,
    )

    # Wrap in Lightning module
    lightning_module = TEMLightningModule(tem_model, training_config)
    print(f"  Training configuration:")
    print(f"    - Max steps: {config.max_steps}")
    print(f"    - BPTT rollout: {training_config.n_rollout}")
    print(f"    - Learning rate: {training_config.lr_max}")
    print(f"    - LR decay: {training_config.lr_decay_rate} every {training_config.lr_decay_steps} steps")

    # =========================================================================
    # PHASE 4: Trainer Setup
    # =========================================================================
    print("\nPhase 4: Initializing PyTorch Lightning Trainer...")

    # Configure logger (TensorBoard always enabled)
    logger = L.pytorch.loggers.TensorBoardLogger("logs", name=Path(__file__).stem)
    print(f"  TensorBoard logging enabled: logs/{Path(__file__).stem}")

    # Configure callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="tem-step={step:06d}-train_loss={train/loss:.4f}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
            every_n_train_steps=100,  # Save checkpoint every 50 training steps
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Calculate expected batches (each batch processes walk_length/n_rollout optimizer steps)
    # With manual optimization, max_steps counts optimizer steps, not batches
    steps_per_batch = (datamodule.walk_length + config.n_rollout - 1) // config.n_rollout  # Ceiling division
    target_batches = (config.max_steps + steps_per_batch - 1) // steps_per_batch  # Ceiling division

    # Create trainer
    trainer = L.Trainer(
        max_epochs=1,  # Single epoch with limit_train_batches
        limit_train_batches=target_batches,  # Control actual training duration
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        num_sanity_val_steps=0,  # Skip sanity validation to prevent early stopping
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator="auto",  # Use GPU if available
        devices=1,
    )
    print(f"  Trainer configured for approx. {target_batches} batches (~{config.max_steps} steps)")

    # =========================================================================
    # PHASE 5: Training
    # =========================================================================
    print("\nPhase 5: Starting training...")
    print("=" * 80)

    trainer.fit(lightning_module, datamodule)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)
