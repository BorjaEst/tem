"""
Sensory Processing Example: Multi-Scale Temporal Filtering

This example demonstrates the SensoryProcessor's ability to learn and apply
multi-scale temporal filtering to sensory observations. We generate synthetic
sequences with different temporal dynamics and visualize how the processor
captures patterns at multiple time scales.

Key Concepts Demonstrated:
1. Two-hot encoding for dimensionality reduction
2. Multi-frequency temporal filtering (fast to slow)
3. Normalization for stable representations
4. Memory projection for downstream processing

The example trains a SensoryProcessor with 5 frequency modules on synthetic
sequences containing slow trends, medium oscillations, and fast changes. The
learned filter parameters (α) determine how quickly each frequency adapts to
new observations, creating a hierarchy from rapid (α≈0.99) to slow (α≈0.01).

Visualization:
A single comprehensive figure with 4 panels showing:
- A. Training convergence over epochs
- B. Multi-scale filtered representations vs input observations
- C. Evolution of learned filter parameters during training
- D. Frequency response characteristics of the learned filters

Usage:
    python examples/sensory.py --n_epochs=10 --n_sequences=500

Output:
    outputs/sensory/sensory_analysis.png - Comprehensive 4-panel analysis
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
from torch_tem.figures import plot_sensory_analysis, print_sensory_summary
from torch_tem.modules.sensory import SensoryProcessor, SensoryState

# ==============================================================================
# Configuration
# ==============================================================================


class ExperimentConfig(BaseSettings):
    """Configuration for sensory processing experiment."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Data generation
    n_observations: int = Field(default=45, ge=10, description="Number of distinct observations")
    n_sequences: int = Field(default=1000, ge=100, description="Number of training sequences")
    sequence_length: int = Field(default=50, ge=10, description="Length of each sequence")

    # Model architecture
    n_compressed: int = Field(default=10, ge=5, description="Compressed observation dimension")
    n_frequencies: int = Field(default=5, ge=2, description="Number of frequency modules")
    n_grid: int = Field(default=30, ge=10, description="Grid cell dimension for memory")

    # Training
    batch_size: int = Field(default=32, ge=1, description="Training batch size")
    n_epochs: int = Field(default=5, ge=1, description="Number of training epochs")
    learning_rate: float = Field(default=0.001, gt=0, description="Learning rate")

    # DataModule settings
    val_split: float = Field(default=0.1, ge=0.05, le=0.3, description="Validation split fraction")

    # Output
    output_dir: str = Field(default="outputs/sensory", description="Output directory")
    seed: int = Field(default=42, ge=0, description="Random seed")


# ==============================================================================
# Model Wrapper
# ==============================================================================


class SensoryLearner(pl.LightningModule):
    """Lightning module wrapper around SensoryProcessor for end-to-end training.

    Adds a simple prediction head to demonstrate that the learned
    representations are useful for downstream tasks.
    """

    def __init__(
        self,
        processor: SensoryProcessor,
        n_compressed: int,
        n_observations: int,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["processor"])

        self.processor = processor
        self.learning_rate = learning_rate

        # Prediction head: predict next observation from filtered state
        self.predictor = nn.Linear(n_compressed, n_observations)

        # Storage for history tracking
        self.training_history = {
            "loss": [],
            "alpha_values": [[] for _ in range(processor.n_freq)],
        }

    def forward(self, sequence: Tensor) -> Tuple[List[SensoryState], Tensor]:
        """Process sequence and predict next observations.

        Args:
            sequence: [seq_len, n_obs] one-hot sequence

        Returns:
            states: List of SensoryState for each time step
            predictions: [seq_len-1, n_obs] predicted next observations
        """
        # Ensure sequence is on the correct device
        sequence = sequence.to(self.device)

        seq_len = sequence.shape[0]
        states = []
        predictions = []

        # Initialize previous state with correct dimensions (n_compressed, not n_freq)
        n_compressed = self.processor.two_hot_table.shape[1]
        x_prev = [torch.zeros(1, n_compressed, device=self.device) for _ in range(self.processor.n_freq)]

        for t in range(seq_len):
            # Process current observation
            x_raw = sequence[t : t + 1]  # Add batch dim
            state = self.processor(x_raw, x_prev)
            states.append(state)

            # Predict next observation (using high-frequency filtered state)
            if t < seq_len - 1:
                pred = self.predictor(state.filtered[0])
                predictions.append(pred)

            # Update previous state
            x_prev = state.filtered

        predictions = torch.cat(predictions, dim=0) if predictions else torch.zeros(0, sequence.shape[1], device=self.device)

        return states, predictions

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """Training step for a batch of sequences.

        Args:
            batch: Batch of sequences [batch_size, seq_len, n_obs]
            batch_idx: Batch index

        Returns:
            Loss value
        """
        # Process each sequence independently (no batch processing for sequences)
        batch_loss = 0.0
        for seq in batch:
            states, predictions = self(seq)

            # Target: next observations in sequence
            targets = seq[1:].argmax(dim=-1)

            # Prediction loss
            loss = F.cross_entropy(predictions, targets)
            batch_loss += loss

        batch_loss /= len(batch)

        # Log metrics
        self.log("train_loss", batch_loss, prog_bar=True, on_step=False, on_epoch=True)

        return batch_loss

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch to track filter parameters."""
        # Record alpha values for visualization
        with torch.no_grad():
            current_loss = self.trainer.callback_metrics.get("train_loss", 0.0)
            if isinstance(current_loss, torch.Tensor):
                current_loss = current_loss.item()
            self.training_history["loss"].append(current_loss)

            for f in range(self.processor.n_freq):
                alpha = torch.sigmoid(self.processor.alpha[f]).item()
                self.training_history["alpha_values"][f].append(alpha)

    def configure_optimizers(self):
        """Configure optimizer."""
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
    print("Sensory Processing Experiment: Multi-Scale Temporal Filtering")
    print("=" * 80)

    # Load configuration
    config = ExperimentConfig()

    print(f"\nConfiguration:")
    print(f"  Observations: {config.n_observations}")
    print(f"  Sequences: {config.n_sequences} × {config.sequence_length} steps")
    print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Training: {config.n_epochs} epochs, batch size {config.batch_size}")
    print(f"  Validation split: {config.val_split:.1%}")
    print(f"  Output: {config.output_dir}")

    # Set random seed
    pl.seed_everything(config.seed)

    # Create dataset generator
    def dataset_generator(n_samples: int) -> MultiScaleSequenceDataset:
        return MultiScaleSequenceDataset(
            n_sequences=n_samples,
            sequence_length=config.sequence_length,
            n_observations=config.n_observations,
            seed=config.seed,
        )

    # Create DataModule
    print(f"\n📊 Setting up data module...")
    dm_params = DataModuleParams(
        num_samples=config.n_sequences,
        val_split=config.val_split,
        test_split=0.0,  # No test split for this example
        n_predict=0,
        batch_size=config.batch_size,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,
        drop_last=True,
    )
    datamodule = BaseDataModule(generator=dataset_generator, params=dm_params)

    # Create two-hot encoding table
    from torch_tem.utils import generate_two_hot_codes

    two_hot_codes = generate_two_hot_codes(n_bits=config.n_compressed, n_codes=config.n_observations)
    two_hot_table = torch.tensor(two_hot_codes, dtype=torch.float)

    # Create tile matrices
    tile_matrices = [torch.randn(config.n_compressed, config.n_grid * config.n_compressed) / 10 for _ in range(config.n_frequencies)]

    # Initialize frequencies (high to low)
    initial_frequencies = [0.99 * (0.1 ** (f / (config.n_frequencies - 1))) for f in range(config.n_frequencies)]

    print(f"\n🔧 Initializing model...")
    print(f"  Initial α values: {[f'{a:.3f}' for a in initial_frequencies]}")

    # Create processor
    processor = SensoryProcessor(
        n_frequencies=config.n_frequencies,
        initial_frequencies=initial_frequencies,
        two_hot_table=two_hot_table,
        tile_matrices=tile_matrices,
    )

    # Wrap in Lightning module
    model = SensoryLearner(
        processor=processor,
        n_compressed=config.n_compressed,
        n_observations=config.n_observations,
        learning_rate=config.learning_rate,
    )

    # Setup callbacks
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="sensory-{epoch:02d}-{train_loss:.4f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )

    # Create trainer
    print(f"\n🚀 Training...")
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        logger=False,  # Disable default logger for cleaner output
        enable_checkpointing=True,
        accelerator="cpu",  # Use CPU to avoid device transfer issues
    )

    # Train the model
    trainer.fit(model, datamodule)

    # Get training history
    history = model.get_history()

    # Generate visualizations
    print(f"\n📈 Generating visualizations...")

    # Get a sample dataset for visualization
    datamodule.setup("fit")
    sample_dataset = datamodule.datasets["fit"][0]

    output_path = plot_sensory_analysis(model, sample_dataset, history, output_dir)
    print(f"✅ Saved comprehensive analysis: {output_path}")

    # Print summary statistics
    print_sensory_summary(history, config.n_frequencies)

    # Print final filter values
    print(f"\n📊 Final Filter Parameters:")
    with torch.no_grad():
        for f in range(config.n_frequencies):
            alpha = torch.sigmoid(model.processor.alpha[f]).item()
            print(f"  Frequency {f}: α = {alpha:.4f}")

    print(f"\n✅ Experiment complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
