import math
from typing import Any, List, Optional, Tuple

import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from pydantic import Field, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor, nn
from torch.optim import Adam, Optimizer

import torch_tem as tem


# -----------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    # Model architecture parameters
    example_model_parameter: int = Field(default=42, ge=1, le=100, description="An example integer parameter")

    # Data and augmentation parameters
    seed: int = Field(default=0, ge=0, description="Random seed for reproducibility")
    example_data_param: float = Field(default=0.5, ge=0.0, le=1.0, description="An example float parameter")

    # Training Settings
    max_epochs: PositiveInt = Field(default=200, ge=1, le=1000, description="Maximum training epochs")
    batch_size: PositiveInt = Field(default=32, gt=0, le=1024, description="Batch size for training")

    # Logging and Output Settings
    log_dir: str = Field(default="logs", description="Directory for experiment logs")
    experiment_name: str = Field(__file__.split("/")[-1].replace(".py", ""), description="Experiment name")
    checkpoint_freq: PositiveInt = Field(default=50, ge=1, le=50, description="Checkpoint frequency")


# -------------------------------------------------------------------------------------------
class Layer(nn.Linear):
    def __init__(self, n_in: int, n_out: int):
        super().__init__(in_features=n_in, out_features=n_out, bias=True)
        self.register_buffer("activations", None)  # Starts without activation values

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        currents = super().forward(*args, **kwargs)
        self.activations = nn.functional.gelu(currents)
        return self.activations


# -------------------------------------------------------------------------------------------
class Autoencoder(pl.LightningModule):
    def __init__(self, **kwargs: Any):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize model layers
        self.layer1 = Layer(625, 256)
        self.layer2 = Layer(256, 64)

        # Input and output reshaping layers (25x25 maps)
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1, (25, 25))

    def configure_optimizers(self) -> Optimizer:
        optimizer_parameters = [{"params": self.parameters(), "lr": 1e-3}]
        return Adam(optimizer_parameters)

    # -----------------------------------------------------------------------------------
    def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
        sensors, _ = batch
        inputs = self.flatten(sensors)
        h1 = self.layer1(inputs)
        h2 = self.layer2(h1)
        return self.unflatten(h2)

    # -----------------------------------------------------------------------------------
    def compute_loss(self, batch: Tuple[Tensor, Tensor], output: Tensor) -> Tensor:
        _, targets = batch
        return nn.functional.mse_loss(output, targets, reduction="mean")

    # -----------------------------------------------------------------------------------
    def training_step(self, batch: Tensor, batch_idx: int) -> None:
        self.optimizers().zero_grad()
        output = self.forward(batch)
        global_loss = self.compute_loss(batch, output)
        self.manual_backward(global_loss)
        self.optimizers().step()
        self.log("train/loss", global_loss, on_step=True, on_epoch=True, prog_bar=True)

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        output = self.forward(batch)
        val_loss = self.compute_loss(batch, output)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)


# -------------------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n--- Running Baseline Sparse BP Experiment ---")

    # Initialize experiment configuration
    experiment = Experiment()

    # Initialize model with specified architecture
    model = tem.model.TEM(**experiment.model_dump())

    # Initialize data module
    datamodule = tem.data.DataModule()

    # Initialize trainer
    trainer = pl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=experiment.max_epochs,
        callbacks=[ModelCheckpoint(every_n_epochs=experiment.checkpoint_freq, save_weights_only=True)],
        logger=TensorBoardLogger(experiment.log_dir, name=experiment.experiment_name),
        profiler=None,  # "simple" for basic profiling and "advanced" for detailed profiling
    )

    try:  # Train till end of training or keyboard interup
        trainer.fit(model, datamodule=datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        print("Experiment completed.")
