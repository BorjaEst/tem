"""Training configuration for the Temporal Experience Model (TEM)."""

from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch import Tensor, zeros


class TrainingConfig(BaseModel):
    """Training and optimization configuration: schedules, learning rates, loss weights, and curricula.

    This defines how the model is trained but does not affect its structure.
    Can be changed between training runs on the same architecture.
    """

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    # ===================================================================================
    # TRAINING SCHEDULE
    # ===================================================================================

    train_it: int = Field(default=20000, ge=1, description="Number of training walks (environments × walks) to generate")
    n_rollout: int = Field(default=20, ge=1, description="Unroll length for BPTT (steps per optimisation step)")
    batch_size: int = Field(default=16, ge=1, description="Number of walks processed in parallel during training")

    # ===================================================================================
    # WALK LENGTH CURRICULUM
    # ===================================================================================

    walk_it_min: int = Field(default=25, ge=1, description="Minimum walk length at end of training")
    walk_it_max: int = Field(default=300, ge=1, description="Maximum walk length at start of training")

    @computed_field(description="Width of the walk-length sampling window (used for curriculum over time)")
    @property
    def walk_it_window(self) -> float:
        return 0.2 * (self.walk_it_max - self.walk_it_min)

    # ===================================================================================
    # LEARNING RATE SCHEDULE
    # ===================================================================================

    lr_max: float = Field(default=9.4e-4, gt=0, description="Maximum learning rate")
    lr_min: float = Field(default=8e-5, gt=0, description="Minimum learning rate")
    lr_decay_rate: float = Field(default=0.5, gt=0, le=1, description="Exponential decay factor for the learning rate")
    lr_decay_steps: int = Field(default=4000, ge=1, description="Number of steps over which decay is applied")

    # ===================================================================================
    # LOSS WEIGHTS
    # ===================================================================================

    loss_weights_x: float = Field(default=1.0, ge=0, description="Weight of reconstruction/prediction losses on x")
    loss_weights_p: float = Field(default=1.0, ge=0, description="Weight of grounded location losses on p")
    loss_weights_g: float = Field(default=1.0, ge=0, description="Weight of abstract location losses on g")
    loss_weights_reg_g: float = Field(default=0.01, ge=0, description="Weight of regularisation loss on abstract location")
    loss_weights_reg_p: float = Field(default=0.02, ge=0, description="Weight of regularisation loss on grounded location")
    loss_weights: Tensor = Field(default_factory=lambda: zeros(8), description="Combined loss weights [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]")

    # ===================================================================================
    # CURRICULUM SCHEDULES
    # ===================================================================================

    loss_weights_p_g_it: int = Field(default=2000, ge=1, description="Iterations until latent losses (L_p_g, L_p_x, L_g) are fully weighted")
    loss_weights_reg_p_it: int = Field(default=4000, ge=1, description="Iterations until grounded-location regularisation is fully weighted")
    loss_weights_reg_g_it: int = Field(default=40000000, ge=1, description="Iterations until abstract-location regularisation is fully weighted")
    eta_it: int = Field(default=16000, ge=1, description="Iterations until η (rate of remembering) reaches its full value")
    lambda_it: int = Field(default=200, ge=1, description="Iterations until λ (rate of forgetting) reaches its full value")

    # ===================================================================================
    # PRECISION-WEIGHTED MEAN SCHEDULE (p→g inference)
    # ===================================================================================

    p2g_scale_offset: float = Field(default=0.0, ge=0, description="Scaling factor for variance offset on inferred p in p→g")
    p2g_sig_val: float = Field(default=10000.0, ge=0, description="Magnitude of variance offset for inferred grounded location")
    p2g_sig_half_it: int = Field(default=400, ge=1, description="Iteration at which offset scaling is 0.5")
    p2g_sig_scale_it: int = Field(default=200, ge=1, description="Controls steepness of offset scaling decay over iterations")
