"""Environment configuration for the Temporal Experience Model (TEM)."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field


class EnvironmentConfig(BaseModel):
    """Environment and task configuration: action space, exploration, and reward-driven behaviour.

    This defines the task distribution the model is trained on but does not affect model structure.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # GEOMETRY / GRID LAYOUT
    # ===================================================================================

    width: int = Field(default=5, ge=1, description="Grid width (number of columns)")
    height: int = Field(default=5, ge=1, description="Grid height (number of rows)")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation assignment strategy; mirrors Environment.from_grid observation_mode")
    randomize_observations: bool = Field(default=False, description="Randomize observation assignments after environment construction")

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

    # ===================================================================================
    # DERIVED GEOMETRY (for wiring into data/architecture)
    # ===================================================================================

    @computed_field(description="Total number of discrete locations in the grid")
    @property
    def n_locations(self) -> int:
        return self.width * self.height

    @computed_field(description="Number of unique sensory observations implied by geometry and observation_mode")
    @property
    def n_observations(self) -> int:
        if self.observation_mode == "unique":
            return self.n_locations
        if self.observation_mode == "tiled":
            return 4
        # "random": mirror Environment.from_grid behaviour
        return max(4, self.n_locations // 4)
