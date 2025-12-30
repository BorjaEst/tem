"""PyTorch Lightning DataModule for TEM training.

This module generates synthetic walk data on the fly and yields *time-major*
full-walk tensors suitable for manual truncated BPTT in the Lightning module.

Batch contract (time-major):
    - observations: float32 [T, B, n_o]
    - actions: int64 [T, B]
    - locations: int64 [T, B] (auxiliary)
"""

import functools
from typing import Optional

import lightning as L
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader

from torch_tem.data.environment import Environment, EnvironmentConfig
from torch_tem.data.policies import PolicyConfig, PolicyGenerator, RandomPolicyConfig
from torch_tem.data.walks import WalkDataset, WalkGenerator


class DataModuleConfig(BaseModel):
    """DataModule configuration: environment, walks, policies, batching, and data splits.

    This is the single source of truth for all data generation parameters.
    Defines how training/validation/test data is generated and delivered to the model.
    """

    model_config = ConfigDict(extra="ignore", strict=False)

    # ===================================================================================
    # ENVIRONMENT and POLICY
    # ===================================================================================

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration")
    policy: PolicyConfig = Field(default_factory=RandomPolicyConfig, description="Action selection policy for walk generation")

    # ===================================================================================
    # SEQUENCE SHAPE (BPTT)
    # ===================================================================================

    sequence_length: int = Field(default=100, ge=1, description="Number of timesteps per walk (T). Sole source of sequence length for the DataModule.")
    return_locations: bool = Field(default=True, description="If True, include location IDs as third element of the batch tuple.")

    # ===================================================================================
    # DATA SPLITS
    # ===================================================================================

    n_train_batches: int = Field(default=1000, ge=1, description="Number of training batches per epoch")
    n_val_batches: int = Field(default=10, ge=0, description="Number of validation batches")
    n_test_batches: int = Field(default=10, ge=0, description="Number of test batches")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible environment/walk generation")

    # ===================================================================================
    # DATALOADER SETTINGS
    # ===================================================================================

    batch_size: int = Field(default=16, ge=1, description="Number of walks per batch")
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader worker processes (0 = main process only)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")


class TEMDataModule(L.LightningDataModule):
    """Lightning DataModule that yields time-major full-walk batches.

    The DataLoader returns tuples `(observations, actions, locations)` with the
    following shapes:

    - observations: float32 tensor of shape (T, B, n_o)
    - actions: int64 tensor of shape (T, B)
    - locations: int64 tensor of shape (T, B) (auxiliary)

    Notes:
        - The walk length `T` is determined solely by `config.sequence_length`.
        - Truncated BPTT chunking (TBPTT) is performed in the LightningModule,
          not in the DataModule.
        - Epoch sizing is controlled by `n_*_batches`: the dataset length is
          `n_batches * batch_size`, so the DataLoader produces exactly
          `n_batches` batches (assuming `batch_size` is unchanged).
    """

    def __init__(self, config: DataModuleConfig):
        """Initialize the DataModule.

        Args:
            config: Data generation and dataloader configuration.
        """
        super().__init__()
        self.config = config
        self.environment: Optional[Environment] = None
        self.policy_gen: Optional[PolicyGenerator] = None
        self.walk_gen: Optional[WalkGenerator] = None
        self.datasets: dict[str, WalkDataset] = {}

    def setup(self, stage: Optional[str] = None) -> None:
        """Build runtime objects and create datasets for the given stage.

        Args:
            stage: One of `None`, "fit", "validate", or "test".
        """
        # Build environment (once per datamodule instance).
        if self.environment is None:
            self.environment = Environment(self.config.environment)
            self.environment.validate()

        # Build policy and walk generators (once per datamodule instance).
        if self.policy_gen is None or self.walk_gen is None:
            self.policy_gen = PolicyGenerator(self.environment)
            self.walk_gen = WalkGenerator(self.environment, repeat_bias=self.config.environment.explore_bias)

        if stage in (None, "fit"):
            self.datasets["fit"] = self._make_dataset(n_batches=self.config.n_train_batches)

        if stage in (None, "fit", "validate"):
            self.datasets["validate"] = self._make_dataset(n_batches=self.config.n_val_batches)

        if stage in (None, "test"):
            self.datasets["test"] = self._make_dataset(n_batches=self.config.n_test_batches)

    def _make_dataset(self, n_batches: int) -> WalkDataset:
        """Create a `WalkDataset` sized to yield exactly `n_batches` batches.

        Args:
            n_batches: Number of DataLoader batches to produce.

        Returns:
            A `WalkDataset` exposing `n_batches * batch_size` items.
        """
        n_items = n_batches * self.config.batch_size
        return WalkDataset(n_items, self.environment, self.policy_gen, self.walk_gen, params=self.config)

    def sample_batch(self, stage: str) -> tuple:
        """Fetch a single batch from the specified stage's DataLoader.

        Args:
            stage: Dataset stage key ("fit", "validate", "test").

        Returns:
            A single batch tuple `(observations, actions, locations)`.
        """
        loader = self.dataloader_for(stage)
        return next(iter(loader))

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self.dataloader_for("fit")

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self.dataloader_for("validate")

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self.dataloader_for("test")

    def dataloader_for(self, stage: str) -> DataLoader:
        """Create a DataLoader for a previously created dataset stage.

        Args:
            stage: Dataset stage key ("fit", "validate", "test").

        Returns:
            A PyTorch DataLoader yielding time-major full-walk batches.

        Raises:
            RuntimeError: If `setup()` has not been called for the requested stage.
        """
        dataset = self.datasets.get(stage)
        if dataset is None:
            raise RuntimeError(f"DataModule not set up for stage '{stage}'. Call setup() first.")

        collate = functools.partial(WalkGenerator.collate, return_locations=self.config.return_locations)
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=collate,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last,
        )
