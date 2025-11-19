"""Environment configuration for the Temporal Experience Model (TEM)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field


class EnvironmentConfig(BaseModel):
    """Environment and task configuration: action space, exploration, and reward-driven behaviour.

    This defines the task distribution the model is trained on but does not affect model structure.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # ACTION SPACE
    # ===================================================================================

    has_static_action: bool = Field(default=True, description="Include standing-still action in the action set")
    n_actions: int = Field(default=4, ge=1, description="Number of non-static actions (cardinal movements, etc.)")
    explore_bias: float = Field(default=2.0, ge=0, description="Bias to repeat last action to encourage straight walks")

    # ===================================================================================
    # SHINY OBJECTS (reward-driven behaviour)
    # ===================================================================================

    shiny_rate: float = Field(default=0.0, ge=0, description="Fraction of training environments with shiny objects (0 disables them)")
    shiny_gamma: float = Field(default=0.7, ge=0, le=1, description="Discount factor for shiny-object Q-values")
    shiny_beta: float = Field(default=1.5, ge=0, description="Inverse temperature for shiny-object action selection")
    shiny_n: int = Field(default=2, ge=0, description="Number of shiny objects placed in the arena")
    shiny_returns: int = Field(default=15, ge=0, description="Number of revisits to a shiny object after discovery")
    shiny: dict[str, Any] = Field(default_factory=dict, description="Legacy-style shiny parameter dict used by the world object")

    @computed_field(description="Grouped shiny parameters dictionary, matching legacy 'shiny' field")
    @property
    def shiny_dict(self) -> dict[str, Any]:
        return {"gamma": self.shiny_gamma, "beta": self.shiny_beta, "n": self.shiny_n, "returns": self.shiny_returns}
