"""Inference configuration for the Temporal Experience Model (TEM)."""

from pydantic import BaseModel, ConfigDict, Field


class InferenceConfig(BaseModel):
    """Inference and generation behaviour configuration: runtime switches and memory dynamics.

    These parameters can be changed at test/inference time without retraining.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # INFERENCE BEHAVIOUR
    # ===================================================================================

    do_sample: bool = Field(default=False, description="If False, use distribution means instead of sampling (no observation noise)")
    use_p_inf: bool = Field(default=True, description="Use inferred ground location p_inf when inferring new abstract location")

    # ===================================================================================
    # MEMORY DYNAMICS (can be tuned at inference time)
    # ===================================================================================

    eta: float = Field(default=0.5, ge=0, le=1, description="Hebbian rate of remembering (η in memory update)")
    kappa: float = Field(default=0.8, ge=0, le=1, description="Hebbian retrieval decay term κ")
