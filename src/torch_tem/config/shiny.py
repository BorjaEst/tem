"""Shiny object configuration and placement.

Provides configuration for reward-driven behavior with goal-switching:
- ShinyConfig: Placement parameters, Q-learning settings, and goal-switching behavior
"""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from torch_tem.config import EnvironmentConfig


class ShinyConfig(BaseModel):
    """Configuration for shiny object environments with goal-switching behavior.

    Shiny objects are special reward locations that drive goal-directed navigation
    with automatic goal switching during walks. Once an agent discovers a shiny object,
    it will revisit it multiple times before switching to another goal.

    This configuration controls shiny object placement (count and spatial separation),
    Q-learning parameters for goal-directed navigation (discount factor and temperature),
    and goal-switching behavior (number of returns before switching).
    """

    model_config = ConfigDict(extra="ignore", strict=False)

    # ===================================================================================
    # SHINY OBJECT PLACEMENT
    # ===================================================================================

    n: int = Field(gt=0, description="Number of shiny objects to place in the environment")
    min_separation: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum distance ratio between shiny objects (relative to max graph distance)")

    # ===================================================================================
    # GOAL-SWITCHING BEHAVIOR
    # ===================================================================================

    returns: int = Field(gt=0, description="Number of revisits to a shiny object after discovery before switching to another goal")

    # ===================================================================================
    # Q-LEARNING NAVIGATION PARAMETERS
    # ===================================================================================

    gamma: float = Field(default=0.9, ge=0.0, le=1.0, description="Discount factor for Q-learning value iteration")
    beta: float = Field(default=1.0, gt=0.0, description="Softmax inverse temperature for action selection (higher = more deterministic)")

    # ===================================================================================
    # VALIDATION
    # ===================================================================================

    @model_validator(mode="after")
    def validate_feasibility(self) -> "ShinyConfig":
        """Validate that configuration parameters are internally consistent.

        Checks:
            - Multiple shiny objects cannot be placed with very high separation requirement
              (min_separation > 0.9 would force shinies to opposite corners of the environment)

        Raises:
            ValueError: If configuration is infeasible
        """
        if self.n > 1 and self.min_separation > 0.9:
            raise ValueError("Cannot place multiple shiny objects with min_separation > 0.9")
        return self

    # ===================================================================================
    # FACTORY METHODS
    # ===================================================================================

    @classmethod
    def from_environment_config(cls, env_config: EnvironmentConfig, min_separation: float = 0.3) -> "ShinyConfig":
        """Create shiny configuration from an EnvironmentConfig.

        Extracts shiny-related parameters from the global environment configuration
        and constructs a dedicated ShinyConfig object.

        Args:
            env_config: Global environment configuration containing shiny parameters
            min_separation: Minimum distance ratio between shiny objects (default: 0.3)

        Returns:
            ShinyConfig: Constructed shiny configuration with parameters from env_config
        """
        return cls(
            n=env_config.shiny_n,
            returns=env_config.shiny_returns,
            gamma=env_config.shiny_gamma,
            beta=env_config.shiny_beta,
            min_separation=min_separation,
        )
