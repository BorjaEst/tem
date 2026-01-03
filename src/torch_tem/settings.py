"""Settings for TEM training and data generation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field


class DataSettings(BaseModel):
    """Data/environment generation settings (Lightning datamodule + dataset)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    envs: list[Path] = Field(default_factory=lambda: [Path("./envs/10x10.json")], description="Environment JSON files for training.")
    randomise_observations: bool = Field(default=True, description="Randomise observations in environments.")

    # Walk generation
    batch_size: int = Field(default=16, description="Number of parallel walks.")
    n_rollout: int = Field(default=20, description="Steps per truncated BPTT chunk.")

    # Validation and test settings (optional, disabled by default)
    enable_validation: bool = Field(default=False, description="Enable validation loop during training.")
    enable_test: bool = Field(default=False, description="Enable test loop after training.")
    val_batches: int = Field(default=10, description="Number of batches per validation epoch (finite).")
    test_batches: int = Field(default=10, description="Number of batches per test epoch (finite).")
    val_seed: int = Field(default=42, description="Random seed for validation dataset (deterministic).")
    test_seed: int = Field(default=43, description="Random seed for test dataset (deterministic).")

    # World exploration
    explore_bias: int = Field(default=2, description="Bias for explorative behaviour to pick the same action again, to encourage straight walks")

    # Shiny env sampling
    shiny_rate: int = Field(default=0, description="Rate at which shiny environments occur during training.")
    shiny_gamma: float = Field(default=0.7, description="Shiny policy gamma.")
    shiny_beta: float = Field(default=1.5, description="Shiny policy beta.")
    shiny_n: int = Field(default=2, description="Number of shiny objects.")
    shiny_returns: int = Field(default=15, description="Returns to shiny object after finding.")

    @computed_field
    @property
    def shiny(self) -> dict:
        """Dict consumed by torch_tem.data.World(shiny=...)."""
        return {"gamma": self.shiny_gamma, "beta": self.shiny_beta, "n": self.shiny_n, "returns": self.shiny_returns}


class LoggerSettings(BaseModel):
    """Logging settings for TensorBoard logger."""

    model_config = ConfigDict(extra="forbid")

    save_dir: Path = Field(default=Path("./logs"), description="Directory to save logs (default: ./lightning_logs).")
    name: Optional[str] = Field(default=None, description="Experiment name for logger.")
    version: Optional[str] = Field(default=None, description="Version/run identifier (auto-increments if None).")
    log_graph: bool = Field(default=False, description="Log model graph to TensorBoard.")
    prefix: str = Field(default="", description="Prefix for all logged metrics.")


class CheckpointSettings(BaseModel):
    """Settings for model checkpointing."""

    model_config = ConfigDict(extra="forbid")

    every_n_train_steps: int = Field(default=1000, description="Save a checkpoint every N training steps.")
    save_last: bool = Field(default=True, description="Whether to always save the last checkpoint.")


class TrainerSettings(BaseModel):
    """Subset of Lightning Trainer kwargs with support for extra keys."""

    model_config = ConfigDict(extra="allow")

    max_steps: int = Field(default=20000)
    log_every_n_steps: int = Field(default=10)
    enable_progress_bar: bool = Field(default=True)


class ScheduleSettings(BaseModel):
    """Training schedules (loss weights, lr, inference variance offset, etc)."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    # Loss weights
    loss_weights_x: float = Field(default=1.0)
    loss_weights_p: float = Field(default=1.0)
    loss_weights_g: float = Field(default=1.0)
    loss_weights_reg_g: float = Field(default=0.01)
    loss_weights_reg_p: float = Field(default=0.02)

    loss_weights_p_g_it: int = Field(default=2000)
    loss_weights_reg_p_it: int = Field(default=4000)
    loss_weights_reg_g_it: int = Field(default=40000000)

    # Memory schedules
    eta: float = Field(default=0.5, description="Base Hebbian rate of remembering")
    eta_it: int = Field(default=16000)
    hebbian_decay: float = Field(default=0.9999, description="Base Hebbian decay factor (rate of forgetting)")
    lambda_it: int = Field(default=200)

    # p->g variance offset schedule
    p2g_sig_half_it: int = Field(default=400)
    p2g_sig_scale_it: int = Field(default=200)

    # LR schedule
    lr_max: float = Field(default=9.4e-4)
    lr_min: float = Field(default=8e-5)
    lr_decay_rate: float = Field(default=0.5)
    lr_decay_steps: int = Field(default=4000)

    # Walk length curriculum (annealing schedule)
    walk_it_min: int = Field(default=25, description="Minimum walk length multiplier (curriculum endpoint)")
    walk_it_max: int = Field(default=300, description="Maximum walk length multiplier (curriculum start)")

    @computed_field
    @property
    def walk_it_window(self) -> float:
        """Width of window from which walk lengths are sampled."""
        return 0.2 * (self.walk_it_max - self.walk_it_min)

    @computed_field
    @property
    def loss_weights_base(self) -> torch.Tensor:
        """Base loss weights vector in model.loss() order."""
        return torch.tensor(
            [
                self.loss_weights_p,
                self.loss_weights_p,
                self.loss_weights_x,
                self.loss_weights_x,
                self.loss_weights_x,
                self.loss_weights_g,
                self.loss_weights_reg_g,
                self.loss_weights_reg_p,
            ],
            dtype=torch.float,
        )
