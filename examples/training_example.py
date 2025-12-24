"""Complete TEM training example using PyTorch Lightning."""

from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures
from torch_tem.config import DataModuleConfig, TrainingConfig
from torch_tem.core.model import TEMModel
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

    # TrainingConfig knobs (mirrors torch_tem.config.TrainingConfig)
    n_rollout: int = Field(default=TrainingConfig().n_rollout, ge=5, le=100, description="BPTT rollout length (steps per backward pass)")
    lr_max: float = Field(default=TrainingConfig().lr_max, gt=0, description="Maximum learning rate")
    lr_decay_rate: float = Field(default=TrainingConfig().lr_decay_rate, gt=0, le=1, description="StepLR decay factor (gamma)")
    lr_decay_steps: int = Field(default=TrainingConfig().lr_decay_steps, ge=1, description="StepLR step_size (optimizer steps between decays)")

    loss_weights_x: float = Field(default=TrainingConfig().loss_weights_x, ge=0, description="Weight for sensory loss")
    loss_weights_p: float = Field(default=TrainingConfig().loss_weights_p, ge=0, description="Weight for grounded location loss")
    loss_weights_g: float = Field(default=TrainingConfig().loss_weights_g, ge=0, description="Weight for abstract location loss")
    loss_weights_reg_g: float = Field(default=TrainingConfig().loss_weights_reg_g, ge=0, description="Weight for abstract location regularization")
    loss_weights_reg_p: float = Field(default=TrainingConfig().loss_weights_reg_p, ge=0, description="Weight for grounded location regularization")

    # Trainer control (example-only)
    max_steps: int = Field(default=10, ge=1, description="Maximum number of optimizer steps")

    # Logging and output
    checkpoint_dir: Path = Field(default=Path("checkpoints/training"), description="Directory for model checkpoints")
    output_dir: Path = Field(default=Path("outputs/training"), description="Directory for plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    # Debugging and monitoring
    log_every_n_steps: int = Field(default=1, ge=1, description="Logging frequency")
    val_check_interval: int = Field(default=10, ge=1, description="Validation check interval (number of batches)")

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
    example_config = ExampleConfig()  # Load from CLI args if provided
    dm_config = DataModuleConfig()  # Default Environment and DataModule settings
    model_config = dm_config.build_model_config()
    training_config = TrainingConfig.model_validate(example_config.model_dump())

    print("=" * 80)
    print("TEM Training Example")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  BPTT rollout: {training_config.n_rollout}")
    print(f"  Learning rate: {training_config.lr_max}")
    print(f"  LR decay: gamma={training_config.lr_decay_rate}, step_size={training_config.lr_decay_steps}")
    print(
        "  Loss weights: "
        f"x={training_config.loss_weights_x}, p={training_config.loss_weights_p}, g={training_config.loss_weights_g}, "
        f"reg_g={training_config.loss_weights_reg_g}, reg_p={training_config.loss_weights_reg_p}"
    )
    print()

    # =========================================================================
    # PHASE 1: Environment and Data Setup
    # =========================================================================
    print("Phase 1: Setting up environment and data generation...")

    # Create environment
    datamodule = data.TEMDataModule(dm_config)
    print(f"  DataModule: batch_size={dm_config.batch_size}, walk_length={dm_config.sequence_length}")
    datamodule.setup(stage=None)  # Setup all splits

    # =========================================================================
    # PHASE 2: Model Initialization
    # =========================================================================
    print("\nPhase 2: Initializing TEM model...")

    # Create TEM model
    tem_model = TEMModel(model_config)
    print(f"  Model architecture:")
    print(f"    - Observations: {model_config.n_x}")
    print(f"    - Compressed sensory: {model_config.n_x_c}")
    print(f"    - Grid cells (subsampled): {model_config.n_g_subsampled}")
    print(f"    - Place cells: {model_config.n_p}")
    print(f"    - Actions: {model_config.n_actions}")

    # =========================================================================
    # PHASE 3: Training Configuration
    # =========================================================================
    print("\nPhase 3: Initializing training...")

    # Wrap in Lightning module
    lightning_module = TEMLightningModule(tem_model, training_config)
    print(f"  Training configuration:")
    print(f"    - BPTT rollout: {training_config.n_rollout}")
    print(f"    - Learning rate: {training_config.lr_max}")
    print(f"    - LR decay: {training_config.lr_decay_rate} every {training_config.lr_decay_steps} steps")

    # =========================================================================
    # PHASE 4: Trainer Setup
    # =========================================================================
    print("\nPhase 4: Initializing PyTorch Lightning Trainer...")

    # Configure logger
    logger = L.pytorch.loggers.TensorBoardLogger("logs", name=Path(__file__).stem)
    print(f"  TensorBoard logging enabled: logs/tem_training")

    # Configure callbacks
    callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=example_config.checkpoint_dir,
            filename="tem-{epoch:02d}-{step:06d}",
            save_top_k=3,
            monitor="train/loss",
            mode="min",
        ),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_steps=example_config.max_steps,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=example_config.log_every_n_steps,
        val_check_interval=example_config.val_check_interval,
        enable_progress_bar=True,
        enable_model_summary=True,
        accelerator="auto",  # Use GPU if available
        devices=1,
    )
    print(f"  Trainer ready: max_steps={example_config.max_steps}")
    print(f"    - Checkpoints: {example_config.checkpoint_dir}")
    print(f"    - log_every_n_steps: {example_config.log_every_n_steps}")
    print(f"    - val_check_interval: {example_config.val_check_interval}")

    # =========================================================================
    # PHASE 5: Training
    # =========================================================================
    print("\nPhase 5: Starting training...")
    print("=" * 80)

    # Start training
    trainer.fit(lightning_module, datamodule)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # =========================================================================
    # PHASE 6: Post-Training Visualization
    # =========================================================================
    print("\nPhase 6: Generating visualizations...")

    # Plot environment
    environment, n_locations = datamodule.environment, datamodule.environment.n_locations
    fig1 = figures.plot_environment_layout(environment, title=f"Training Environment ({n_locations} locations)")
    if example_config.save_plots:
        fig1.savefig(example_config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {example_config.output_dir / '01_environment.png'}")

    # Plot loss curves (simplified - use TensorBoard for detailed metrics)
    loss_curves_path = example_config.output_dir / "02_loss_curves.png" if example_config.save_plots else None
    fig2 = figures.plot_loss_curves(trainer, loss_curves_path)
    if example_config.save_plots:
        print(f"  Saved: {example_config.output_dir / '02_loss_curves.png'}")

    # Generate test walk to visualize learned representations
    print("\n  Generating test walk for visualization...")
    tem_model.eval()
    with torch.no_grad():
        # Fetch a single time-major batch from the test DataLoader.
        test_obs, test_actions, _ = datamodule.sample_batch("test")

        # Keep tensors on the same device as the trained model.
        model_device = next(tem_model.parameters()).device
        test_obs = test_obs.to(model_device)
        test_actions = test_actions.to(model_device)

        # Initialize state with first observation: [B, n_x]
        state = tem_model.init_state(test_obs[0])

        # Process through walk (limit to 100 steps)
        max_steps = min(100, test_obs.shape[0])
        batch_size = test_obs.shape[1]
        step_locations = lightning_module.create_step_locations(batch_size)

        for t in range(max_steps):
            # Extract observation and action for timestep t
            obs_t = test_obs[t]  # [B, n_x]
            act_t = test_actions[t]  # [B]
            state = tem_model(obs_t, step_locations, act_t, state)

    # Plot abstract location snapshot
    fig3 = figures.plot_abstract_location_snapshot(
        state.mec.abstract_location,
        model_config.f_initial,
        batch_idx=0,
        title="Abstract Location Snapshot (Final State)",
    )
    if example_config.save_plots:
        fig3.savefig(example_config.output_dir / "03_abstract_location.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: {example_config.output_dir / '03_abstract_location.png'}")

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
        if example_config.save_plots:
            fig4.savefig(example_config.output_dir / "04_memory_structure.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: {example_config.output_dir / '04_memory_structure.png'}")

    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    print(f"  Training steps completed: {trainer.global_step}")
    print(f"  TensorBoard logs: logs/tem_training")
    print(f"    View detailed metrics with: tensorboard --logdir logs")
    print(f"  Best checkpoint: {example_config.checkpoint_dir}")
    if example_config.save_plots:
        print(f"  Plots saved to: {example_config.output_dir}")
    print("=" * 80)

    # Show plots if requested
    if example_config.show_plots:
        plt.show()
    else:
        plt.close("all")
