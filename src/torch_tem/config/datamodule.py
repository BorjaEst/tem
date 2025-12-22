"""DataModule configuration for the Temporal Experience Model (TEM).

This config is the *single source of truth* for data shapes and dataloader
behavior.

In particular, when training TEM with (truncated) BPTT, the canonical batch
layout is time-major full walks:

- observations: float32 tensor [T, B, n_x]
- actions: int64 tensor [T, B]
- locations: int64 tensor [T, B] (auxiliary)

The walk length T is defined solely by :attr:`sequence_length`.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from torch_tem.config.environment import EnvironmentConfig
from torch_tem.config.policies import PolicyConfig, RandomPolicyConfig


class DataModuleConfig(BaseModel):
    """DataModule configuration: environment, walks, policies, batching, and data splits.

    This is the single source of truth for all data generation parameters.
    Defines how training/validation/test data is generated and delivered to the model.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # ENVIRONMENT and POLICY
    # ===================================================================================

    environment: EnvironmentConfig = Field(description="Environment structure and observation mode")
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
    n_val_batches: int = Field(default=100, ge=0, description="Number of validation batches")
    n_test_batches: int = Field(default=100, ge=0, description="Number of test batches")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible environment/walk generation")

    # ===================================================================================
    # DATALOADER SETTINGS
    # ===================================================================================

    batch_size: int = Field(default=16, ge=1, description="Number of walks per batch")
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader worker processes (0 = main process only)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")
