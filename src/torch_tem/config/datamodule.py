"""DataModule configuration for the Temporal Experience Model (TEM)."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

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
    # MULTI-ENVIRONMENT DIVERSITY
    # ===================================================================================

    n_environments: int = Field(default=1, ge=1, description="Number of distinct environments to train on")
    environments_per_batch: int = Field(default=1, ge=1, description="Number of different environments per batch (enables multi-env batches)")
    environment_regeneration_interval: Optional[int] = Field(default=None, ge=1, description="Regenerate environments every N batches (None = never)")

    @model_validator(mode="after")
    def validate_environment_batching(self) -> "DataModuleConfig":
        """Ensure environments_per_batch doesn't exceed batch_size or n_environments."""
        if self.environments_per_batch > self.batch_size:
            raise ValueError(f"environments_per_batch ({self.environments_per_batch}) cannot exceed batch_size ({self.batch_size})")
        if self.environments_per_batch > self.n_environments:
            raise ValueError(f"environments_per_batch ({self.environments_per_batch}) cannot exceed n_environments ({self.n_environments})")
        return self

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

    # ===================================================================================
    # COMPUTED FIELDS
    # ===================================================================================

    @computed_field(description="Walk length curriculum window width (20% of range)")
    @property
    def walk_length_window(self) -> float:
        """Curriculum sampling window width (from policy config)."""
        walk_length_min = self.policy.walk_length_min
        walk_length_max = self.policy.walk_length_max
        return 0.2 * (walk_length_max - walk_length_min)

    @computed_field(description="Current walk length for a given training iteration")
    @property
    def curriculum_steps(self) -> int:
        """Total steps for curriculum completion (from policy config)."""
        policy_steps = self.policy.walk_length_curriculum_steps
        return policy_steps if policy_steps is not None else self.n_train_batches
