"""
Sensory Processing Example: Multi-Scale Temporal Filtering

This example demonstrates the SensoryProcessor's multi-scale temporal filtering
on a simple sequence prediction task. The processor learns to maintain multiple
representations of observations at different time scales.

What This Example Shows:
- How the SensoryProcessor compresses observations (one-hot → two-hot)
- How temporal filters adapt at different rates (fast vs slow)
- How filter parameters evolve during training
- The frequency response of learned filters

Task:
Given a sequence of observations, predict the next observation. This simple
task allows the temporal filters to learn appropriate smoothing rates.

Usage:
    python examples/sensory.py                    # Default: 10 epochs
    python examples/sensory.py --n_epochs=20      # More training
    python examples/sensory.py --n_sequences=500  # More data

Output:
    outputs/sensory/sensory_analysis.png - 4-panel visualization
    outputs/sensory/checkpoints/ - Model checkpoints
"""

import math
from pathlib import Path
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, nn
from torch.optim import Adam

from torch_tem.data import MultiScaleSequenceDataset
from torch_tem.data.datamodule import BaseDataModule, DataModuleParams
from torch_tem.figures.sensory import plot_sensory_analysis, print_processor_summary
from torch_tem.modules.sensory import SensoryProcessor, SensoryState
from torch_tem.utils import generate_two_hot_codes

# ==============================================================================
# Configuration
# ==============================================================================


class ExperimentConfig(BaseSettings):
    """Configuration for sensory processing experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Data generation
    n_observations: int = Field(default=20, ge=10, description="Number of distinct observations")
    n_sequences: int = Field(default=500, ge=100, description="Number of training sequences")
    sequence_length: int = Field(default=30, ge=10, description="Length of each sequence")

    # Model architecture
    n_compressed: int = Field(default=8, ge=5, description="Compressed observation dimension")
    n_frequencies: int = Field(default=3, ge=2, description="Number of frequency modules")

    # Training
    batch_size: int = Field(default=16, ge=1, description="Training batch size")
    n_epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=0.01, gt=0, description="Learning rate")

    # DataModule settings
    val_split: float = Field(default=0.15, ge=0.05, le=0.3, description="Validation split fraction")

    # Output
    output_dir: str = Field(default="outputs/sensory", description="Output directory")
    seed: int = Field(default=42, ge=0, description="Random seed")


# ==============================================================================
# Model Wrapper
# ==============================================================================


class SensoryLearner(pl.LightningModule):
    """Simple learner demonstrating SensoryProcessor on sequence prediction.

    Task: Given observations x[0], x[1], ..., x[t], predict x[t+1]

    The SensoryProcessor maintains multi-scale filtered representations that
    help predict the next observation by capturing both recent changes (high
    frequency) and longer-term trends (low frequency).
    """

    def __init__(self, n_obs: int, n_comp: int, n_freq: int, lr: float = 0.01):
        super().__init__()
        self.save_hyperparameters()

        self.n_observations = n_obs
        self.n_compressed = n_comp
        self.n_frequencies = n_freq
        self.learning_rate = lr

        # Create two-hot encoding table

        two_hot_codes = generate_two_hot_codes(n_bits=n_comp, n_codes=n_obs)
        two_hot_table = torch.tensor(two_hot_codes, dtype=torch.float)

        # Initialize filter frequencies (high to low)
        initial_frequencies = [0.9 * (0.1 ** (f / (n_freq - 1))) for f in range(n_freq)]

        # Create sensory processor
        self.processor = SensoryProcessor(
            n_frequencies=n_freq,
            initial_frequencies=initial_frequencies,
            two_hot_table=two_hot_table,
            tile_matrices=[torch.randn(n_comp, n_comp * 2) / 10 for _ in range(n_freq)],
        )

        # Simple prediction head: use highest frequency (most recent) representation
        self.predictor = nn.Linear(n_comp, n_obs)

        # Track training history
        self.training_history = {
            "loss": [],
            "alpha_values": [[] for _ in range(n_freq)],
        }

    def forward(self, sequence: Tensor) -> Tuple[List[SensoryState], Tensor]:
        """Process sequence through sensory processor and make predictions.

        Args:
            sequence: [seq_len, n_obs] one-hot encoded observations

        Returns:
            states: SensoryState for each time step
            predictions: [seq_len-1, n_obs] predicted next observations
        """
        sequence = sequence.to(self.device)
        seq_len = sequence.shape[0]

        states = []
        predictions = []

        # Initialize with zeros (no prior information)
        x_prev = [torch.zeros(1, self.n_compressed, device=self.device) for _ in range(self.n_frequencies)]

        for t in range(seq_len):
            # Process current observation through sensory pipeline
            x_raw = sequence[t : t + 1]
            state = self.processor(x_raw, x_prev)
            states.append(state)

            # Predict next observation using high-frequency representation
            # (high frequency captures recent patterns best for next-step prediction)
            if t < seq_len - 1:
                pred = self.predictor(state.filtered[0])  # Use filter 0 (highest frequency)
                predictions.append(pred)

            # Update for next step
            x_prev = state.filtered

        predictions = torch.cat(predictions, dim=0) if predictions else torch.zeros(0, self.n_observations, device=self.device)
        return states, predictions

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step: predict next observation in sequence.

        Args:
            batch: [batch_size, seq_len, n_obs] batch of sequences
            batch_idx: Batch index

        Returns:
            Loss value
        """
        batch_loss = 0.0

        for seq in batch:
            states, predictions = self(seq)
            targets = seq[1:].argmax(dim=-1)  # Next observation indices
            loss = F.cross_entropy(predictions, targets)
            batch_loss += loss

        batch_loss /= len(batch)
        self.log("train_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)

        return batch_loss

    def on_train_epoch_end(self) -> None:
        """Track filter parameters after each epoch."""
        with torch.no_grad():
            current_loss = self.trainer.callback_metrics.get("train_loss", 0.0)
            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()
            self.training_history["loss"].append(current_loss)

            for f in range(self.n_frequencies):
                alpha = torch.sigmoid(self.processor.alpha[f]).item()
                self.training_history["alpha_values"][f].append(alpha)

    def configure_optimizers(self):
        """Configure Adam optimizer with higher learning rate for faster convergence."""
        return Adam(self.parameters(), lr=self.learning_rate)

    def get_history(self) -> dict:
        """Get training history for visualization."""
        return self.training_history


# ==============================================================================
# Main Experiment
# ==============================================================================


def main():
    """Run the sensory processing experiment."""

    print("=" * 80)
    print("Sensory Processing: Multi-Scale Temporal Filtering")
    print("=" * 80)

    config = ExperimentConfig()

    print(f"\nConfiguration:")
    print(f"  • Observations: {config.n_observations}")
    print(f"  • Sequences: {config.n_sequences} × {config.sequence_length} steps")
    print(f"  • Frequencies: {config.n_frequencies} (high → low)")
    print(f"  • Training: {config.n_epochs} epochs, batch size {config.batch_size}, lr={config.learning_rate}")
    print(f"  • Output: {config.output_dir}")

    pl.seed_everything(config.seed)

    # Dataset generator
    def dataset_generator(n_samples: int) -> MultiScaleSequenceDataset:
        return MultiScaleSequenceDataset(
            n_sequences=n_samples,
            sequence_length=config.sequence_length,
            n_observations=config.n_observations,
            seed=config.seed,
        )

    # DataModule
    print(f"\n📊 Creating data...")
    dm_params = DataModuleParams(
        num_samples=config.n_sequences,
        val_split=config.val_split,
        test_split=0.0,
        n_predict=0,
        batch_size=config.batch_size,
        num_workers=0,
        pin_memory=False,
        drop_last=True,
    )
    datamodule = BaseDataModule(generator=dataset_generator, params=dm_params)

    # Model
    print(f"\n🔧 Initializing SensoryProcessor...")
    model = SensoryLearner(
        n_observations=config.n_observations,
        n_compressed=config.n_compressed,
        n_frequencies=config.n_frequencies,
        learning_rate=config.learning_rate,
    )

    # Show initial filter rates
    with torch.no_grad():
        print(f"  Initial filter rates (α):")
        for f in range(config.n_frequencies):
            alpha = torch.sigmoid(model.processor.alpha[f]).item()
            print(f"    Filter {f}: α = {alpha:.3f} ({'fast' if alpha > 0.5 else 'slow'})")

    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best-{epoch:02d}-{train_loss:.3f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    # Trainer
    print(f"\n🚀 Training...")
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=False,  # Keep output clean
        log_every_n_steps=10,
        logger=False,
        enable_checkpointing=True,
        accelerator="cpu",
    )

    trainer.fit(model, datamodule)

    # Visualize processor behavior
    print(f"\n📈 Generating visualization...")
    datamodule.setup("fit")
    train_dataset = datamodule.datasets["fit"][0]  # Get the actual dataset (returns tuple)
    sample_sequence = train_dataset[0]  # Get first sequence as tensor

    output_path = plot_sensory_analysis(model.processor, sample_sequence, output_dir)
    print(f"✅ Saved: {output_path}")

    print_processor_summary(model.processor)

    print(f"\n✅ Complete! Results in: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
