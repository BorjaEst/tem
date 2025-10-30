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

import warnings
from pathlib import Path

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
from torch_tem.modules.sensory import SensoryProcessor
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
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    n_epochs: int = Field(default=10, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=0.01, gt=0, description="Learning rate")

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

    def configure_optimizers(self):
        """Configure Adam optimizer with higher learning rate for faster convergence."""
        return Adam(self.parameters(), lr=self.learning_rate)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence.to(self.device)
        seq_len = sequence.shape[0]

        # Pre-allocate predictions tensor (seq_len - 1 predictions)
        predictions = torch.zeros(seq_len - 1, self.n_observations, device=self.device)

        # Initialize with zeros (no prior information)
        x_prev = [torch.zeros(1, self.n_compressed, device=self.device) for _ in range(self.n_frequencies)]

        for t in range(seq_len - 1):
            # Process current observation through sensory pipeline
            x_raw = sequence[t : t + 1]
            state = self.processor(x_raw, x_prev)

            # Predict next observation using high-frequency representation
            # (high frequency captures recent patterns best for next-step prediction)
            predictions[t] = self.predictor(state.filtered[0])  # Use filter 0 (highest frequency)

            # Update for next step
            x_prev = state.filtered

        return predictions

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        batch_loss = 0.0

        for seq in batch:
            predictions = self(seq)
            targets = seq[1:].argmax(dim=-1)  # Next observation indices
            loss = F.cross_entropy(predictions, targets)
            batch_loss += loss

        batch_loss /= len(batch)
        self.log("train_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)

        return batch_loss


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the sensory processing experiment."""
    config = ExperimentConfig()
    pl.seed_everything(config.seed)
    torch.set_float32_matmul_precision("medium")

    # Dataset generator
    def dataset_generator(n_samples: int) -> MultiScaleSequenceDataset:
        return MultiScaleSequenceDataset(
            n_sequences=n_samples,
            sequence_length=config.sequence_length,
            n_observations=config.n_observations,
            seed=config.seed,
        )

    # DataModule
    dm_params = DataModuleParams(
        num_samples=config.n_sequences,
        batch_size=config.batch_size,
        val_split=0.0,  # No validation split
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True,
    )
    datamodule = BaseDataModule(generator=dataset_generator, params=dm_params)

    # Model
    model = SensoryLearner(
        n_obs=config.n_observations,
        n_comp=config.n_compressed,
        n_freq=config.n_frequencies,
        lr=config.learning_rate,
    )

    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training
    pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=config.n_epochs,
        callbacks=[ModelCheckpoint(mode="min", save_weights_only=True)],
    ).fit(model, datamodule)

    # Visualize processor behavior
    datamodule.setup("fit")
    train_dataset = datamodule.datasets["fit"][0]  # Get the dataset (returns tuple)
    sample_sequence = train_dataset[0]  # Get first sequence as tensor
    plot_sensory_analysis(model.processor, sample_sequence, output_dir)
    print_processor_summary(model.processor)
