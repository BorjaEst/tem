"""Training configuration for the Temporal Experience Model (TEM).

This module defines training-time hyperparameters consumed by the PyTorch Lightning
training loop (see `torch_tem.training.TEMLightningModule`).
"""

from pydantic import BaseModel, ConfigDict, Field


class LossConfig(BaseModel):
    """Loss weight configuration for TEM training.

    This configuration encapsulates the weights assigned to each loss component
    during training. These weights influence the relative importance of different
    objectives such as reconstruction accuracy, location grounding, and regularization.
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # ===================================================================================
    # LOSS WEIGHTS
    # ===================================================================================

    weights_x: float = Field(default=1.0, ge=0, description="Weight of reconstruction/prediction losses on x")
    weights_p: float = Field(default=1.0, ge=0, description="Weight of grounded location losses on p")
    weights_g: float = Field(default=1.0, ge=0, description="Weight of abstract location losses on g")
    weights_reg_g: float = Field(default=0.01, ge=0, description="Weight of regularisation loss on abstract location")
    weights_reg_p: float = Field(default=0.02, ge=0, description="Weight of regularisation loss on grounded location")


class TrainingConfig(BaseModel):
    """Training and optimization configuration.

    This configuration is intentionally scoped to parameters that are currently
    used by the Lightning training loop:

    - Truncated BPTT rollout length.
    - Learning rate schedule parameters.
    - Loss term weights.

    Model structure and inference behaviour belong in `ModelConfig`.
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # ===================================================================================
    # ROLLOUT LENGTH
    # ===================================================================================

    n_rollout: int = Field(default=20, ge=1, description="Unroll length for BPTT (timesteps per optimisation step)")

    # ===================================================================================
    # LEARNING RATE SCHEDULE
    # ===================================================================================

    lr_max: float = Field(default=9.4e-4, gt=0, description="Maximum learning rate")
    lr_decay_rate: float = Field(default=0.5, gt=0, le=1, description="StepLR decay factor (gamma)")
    lr_decay_steps: int = Field(default=400, ge=1, description="StepLR step_size (number of optimizer steps between decays)")

    # ===================================================================================
    # LOSS
    # ===================================================================================

    loss: LossConfig = Field(default_factory=LossConfig, description="Loss weight configuration")
