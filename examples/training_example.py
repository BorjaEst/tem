#!/usr/bin/env python3
"""Complete TEM training example using PyTorch Lightning.

This example demonstrates the full training pipeline for the Tolman-Eichenbaum Machine,
showing how to:
- Configure training hyperparameters and architecture
- Initialize the TEM model and Lightning module
- Set up data generation with environment and DataModule
- Execute training with automatic checkpointing and logging
- Visualize training progress and learned representations

The training process implements Backpropagation Through Time (BPTT) with truncation
to handle long temporal sequences while maintaining computational efficiency.

Training Pipeline:
------------------
1. Environment Setup: Create grid world with observations
2. Data Generation: TEMDataModule generates random walks
3. Model Initialization: TEMModel with specified architecture
4. Lightning Wrapper: TEMLightningModule handles optimization
5. Training Loop: PyTorch Lightning Trainer with BPTT
6. Visualization: Plot training metrics and learned representations

Key Components:
---------------
- EnvironmentConfig: Defines the spatial environment
- ModelConfig: Specifies neural architecture (grid/place cells, frequencies)
- TrainingConfig: Sets learning rates, loss weights, BPTT rollout
- TEMDataModule: Generates training data (random walks)
- TEMLightningModule: Handles training loop with manual BPTT
- Trainer: PyTorch Lightning trainer with logging/checkpointing

Usage Examples:
---------------
    # Default: 10x10 grid, 1000 steps, save checkpoints
    python examples/training_example.py

    # Longer training with larger architecture
    python examples/training_example.py --max_steps 5000 --grid_size 8

    # Custom learning rate and batch size
    python examples/training_example.py --lr_max 0.001 --batch_size 32

    # Enable TensorBoard logging
    python examples/training_example.py --use_tensorboard true

    # Different frequency configuration
    python examples/training_example.py --f_initial "[0.95, 0.7, 0.4, 0.2]"

    # Full help
    python examples/training_example.py --help

Outputs:
--------
When save_plots=true, generates visualizations in outputs/training/:
    1. 01_environment.png - Grid layout and observation mapping
    2. 02_loss_curves.png - Training loss over iterations
    3. 03_grid_rate_maps.png - Learned grid cell firing patterns
    4. 04_place_rate_maps.png - Learned place cell firing fields
    5. 05_memory_structure.png - Final Hebbian memory matrices

When use_tensorboard=true, logs to logs/ for TensorBoard visualization:
    - Loss components (total, sensory, abstract, grounded)
    - Learning rate schedule
    - Gradient norms
    - Model parameters
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
class ExampleConfig(BaseSettings):
    """Configuration for TEM training example.

    Defines environment, architecture, training hyperparameters, and output
    options for demonstrating the complete training pipeline.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="training_example")

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
    use_tensorboard: bool = Field(default=True, description="Enable TensorBoard logging")
    checkpoint_dir: Path = Field(default=Path("checkpoints/training"), description="Directory for model checkpoints")
    output_dir: Path = Field(default=Path("outputs/training"), description="Directory for plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    # Debugging and monitoring
    log_every_n_steps: int = Field(default=10, ge=1, description="Logging frequency")
    val_check_interval: int = Field(default=50, ge=1, description="Validation check interval (number of batches)")

    @field_validator("checkpoint_dir", "output_dir")
    @classmethod
    def create_dir(cls, v: Path) -> Path:
        """Create directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Helper Functions
# ==============================================================================
def plot_loss_curves(trainer: L.Trainer, output_path: Path) -> plt.Figure:
    """Plot training loss curves from trainer metrics.

    Note: Only shows final values. For full training curves, use TensorBoard.

    Args:
        trainer: PyTorch Lightning trainer with logged metrics.
        output_path: Path to save the figure.

    Returns:
        Matplotlib figure with loss summary.
    """
    # Extract final metrics from logger
    metrics = trainer.logged_metrics

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Training Loss Summary (Final Values)", fontsize=16, fontweight="bold")

    # Create bar plot of final loss components
    loss_names = []
    loss_values = []

    if "train_loss" in metrics:
        loss_names.append("Total")
        loss_values.append(float(metrics["train_loss"]))
    if "train_lx" in metrics:
        loss_names.append("Sensory (L_x)")
        loss_values.append(float(metrics["train_lx"]))
    if "train_lg" in metrics:
        loss_names.append("Abstract (L_g)")
        loss_values.append(float(metrics["train_lg"]))
    if "train_lp" in metrics:
        loss_names.append("Grounded (L_p)")
        loss_values.append(float(metrics["train_lp"]))

    if loss_names:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        bars = ax.bar(loss_names, loss_values, color=colors[: len(loss_names)])
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.set_title("Loss Components at Final Step")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.4f}", ha="center", va="bottom", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No loss metrics available", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.annotate("Note: For full training curves, view TensorBoard logs", xy=(0.5, -0.15), xycoords="axes fraction", ha="center", fontsize=9, style="italic")

    plt.tight_layout()
    return fig


# ==============================================================================
# Main Training Script
# ==============================================================================
if __name__ == "__main__":
    """Run complete TEM training pipeline with visualizations."""
    config = ExampleConfig()

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
    env_config = EnvironmentConfig(
        width=config.grid_size,
        height=config.grid_size,
        observation_mode=config.observation_mode,
        shiny_rate=config.shiny_rate,
    )

    # Create environment
    env = data.Environment(env_config)
    env.validate()
    print(f"  Environment: {env.n_locations} locations, {env.n_observations} observations")

    # Create data module
    datamodule = TEMDataModule(
        env=env,
        batch_size=config.batch_size,
        walk_length=(config.walk_length_min + config.walk_length_max) // 2,  # Use average for simplicity
        env_config=env_config,
    )
    print(f"  DataModule: batch_size={config.batch_size}, walk_length={datamodule.walk_length}")

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
        train_it=config.max_steps,
        n_rollout=config.n_rollout,
        batch_size=config.batch_size,
        walk_it_min=config.walk_length_min,
        walk_it_max=config.walk_length_max,
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
    print(f"    - Max steps: {training_config.train_it}")
    print(f"    - BPTT rollout: {training_config.n_rollout}")
    print(f"    - Learning rate: {training_config.lr_max}")
    print(f"    - LR decay: {training_config.lr_decay_rate} every {training_config.lr_decay_steps} steps")

    # =========================================================================
    # PHASE 4: Trainer Setup
    # =========================================================================
    print("\nPhase 4: Initializing PyTorch Lightning Trainer...")

    # Configure logger
    if config.use_tensorboard:
        logger = L.pytorch.loggers.TensorBoardLogger("logs", name="tem_training")
        print(f"  TensorBoard logging enabled: logs/tem_training")
    else:
        logger = None
        print(f"  Logging disabled")

    # Configure callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="tem-{epoch:02d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_steps=config.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator="auto",  # Use GPU if available
        devices=1,
    )
    print(f"  Trainer ready: max_steps={config.max_steps}")

    # =========================================================================
    # PHASE 5: Training
    # =========================================================================
    print("\nPhase 5: Starting training...")
    print("=" * 80)

    trainer.fit(lightning_module, datamodule)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # =========================================================================
    # PHASE 6: Post-Training Visualization
    # =========================================================================
    print("\nPhase 6: Generating visualizations...")

    # Plot environment
    fig1 = figures.plot_environment_layout(env, title=f"Training Environment ({env.n_locations} locations)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '01_environment.png'}")

    # Plot loss curves (simplified - use TensorBoard for detailed metrics)
    fig2 = plot_loss_curves(trainer, config.output_dir / "02_loss_curves.png")
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_loss_curves.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '02_loss_curves.png'}")

    # Generate test walk to visualize learned representations
    print("\n  Generating test walk for visualization...")
    tem_model.eval()
    with torch.no_grad():
        # Generate a single test walk
        # Returns: observations [T, B, n_x], actions [T, B], locations [T, B]
        test_obs, test_actions, test_locations = datamodule.generate_batch()

        # Initialize state with first observation: [B, n_x]
        state = tem_model.init_state(test_obs[0])

        # Process through walk (limit to 100 steps)
        max_steps = min(100, test_obs.shape[0])
        batch_size = test_obs.shape[1]
        step_locations = [{"shiny": None}] * batch_size

        for t in range(max_steps):
            # Extract observation and action for timestep t
            obs_t = test_obs[t]  # [B, n_x]
            act_t = test_actions[t] if t > 0 else None  # [B] or None
            state = tem_model(obs_t, step_locations, act_t, state)

    # Plot abstract location snapshot
    fig3 = figures.plot_abstract_location_snapshot(
        state.mec.abstract_location,
        config.f_initial,
        batch_idx=0,
        title="Abstract Location Snapshot (Final State)",
    )
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_abstract_location.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {config.output_dir / '03_abstract_location.png'}")

    # Plot memory structure
    M_gen = state.hpc.memory[0] if state.hpc.memory else None
    M_inf = state.hpc.memory[1] if len(state.hpc.memory) > 1 and state.hpc.memory[1] is not None else None

    if M_gen is not None:
        # Extract first batch item but keep as tensor
        if M_gen.ndim > 2:
            M_gen = M_gen[0]
        if M_inf is not None and M_inf.ndim > 2:
            M_inf = M_inf[0]

        fig4 = figures.plot_memory_matrices(M_gen, M_inf, title="Final Hebbian Memory Matrices")
        if config.save_plots:
            fig4.savefig(config.output_dir / "04_memory_structure.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: {config.output_dir / '04_memory_structure.png'}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"  Training steps completed: {trainer.global_step}")
    if config.use_tensorboard:
        print(f"  TensorBoard logs: logs/tem_training")
        print(f"    View detailed metrics with: tensorboard --logdir logs")
    print(f"  Best checkpoint: {config.checkpoint_dir}")
    if config.save_plots:
        print(f"  Plots saved to: {config.output_dir}")
    print("=" * 80)

    # Show plots if requested
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
