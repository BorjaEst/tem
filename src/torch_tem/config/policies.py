"""Policy configurations for the Temporal Experience Model (TEM).

Provides action selection policies for walk generation:
- RandomPolicyConfig: Uniform random action selection
- DistancePolicyConfig: Graph-distance based goal-directed navigation
- QLearningPolicyConfig: Q-learning value iteration for goal-directed navigation
- ShinyPolicyConfig: Goal-switching policy with multiple reward locations
- MixedPolicyConfig: Weighted mixture of multiple policies
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ===================================================================================
# RANDOM EXPLORATION POLICY
# ===================================================================================


class RandomPolicyConfig(BaseModel):
    """Random uniform action selection policy for unbiased spatial exploration.

    This policy selects actions uniformly at random, providing comprehensive coverage
    of the environment without goal-directed behavior. Uses longer walk lengths to
    ensure adequate spatial sampling.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["random"] = "random"

    # Note: Walk length is owned by DataModuleConfig.sequence_length.


# ===================================================================================
# DISTANCE-BASED GOAL-DIRECTED POLICY
# ===================================================================================


class DistancePolicyConfig(BaseModel):
    """Distance-based goal-directed policy using graph shortest paths.

    Navigates to goal locations using softmax action selection over graph distances.
    Actions are weighted by their progress toward the goal along the shortest path.
    Uses shorter walk lengths since optimal paths reach goals efficiently.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["distance"] = "distance"

    # ===================================================================================
    # NAVIGATION BEHAVIOR
    # ===================================================================================

    beta: float = Field(default=1.0, gt=0, description="Softmax inverse temperature (higher = more deterministic)")
    goal_mode: Literal["random", "fixed"] = Field(default="random", description="Goal selection: random location per walk or fixed goals")
    n_goals: int = Field(default=1, ge=1, description="Number of goal locations (random or fixed)")

    # Note: Walk length is owned by DataModuleConfig.sequence_length.


# ===================================================================================
# Q-LEARNING GOAL-DIRECTED POLICY
# ===================================================================================


class QLearningPolicyConfig(BaseModel):
    """Q-learning value iteration policy for goal-directed navigation.

    Computes optimal action-value functions via value iteration and selects actions
    using softmax over Q-values. Supports discounted rewards for multi-step planning.
    Uses shorter walk lengths since optimal policies reach goals efficiently.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["q_learning"] = "q_learning"

    # ===================================================================================
    # REINFORCEMENT LEARNING PARAMETERS
    # ===================================================================================

    gamma: float = Field(default=0.9, ge=0, le=1, description="Discount factor for future rewards")
    beta: float = Field(default=1.0, gt=0, description="Softmax inverse temperature (higher = more deterministic)")
    n_iterations: Optional[int] = Field(default=None, ge=1, description="Value iteration steps (default: 10 × n_locations)")

    # ===================================================================================
    # GOAL CONFIGURATION
    # ===================================================================================

    goal_mode: Literal["random", "fixed"] = Field(default="random", description="Goal selection: random location per walk or fixed goals")
    n_goals: int = Field(default=1, ge=1, description="Number of goal locations (random or fixed)")

    # Note: Walk length is owned by DataModuleConfig.sequence_length.


# ===================================================================================
# MIXED POLICY
# ===================================================================================


class MixedPolicyConfig(BaseModel):
    """Weighted mixture of multiple policies for diverse behavior.

    Combines multiple policies with specified weights, sampling from each according
    to the weight distribution.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["mixed"] = "mixed"

    # ===================================================================================
    # POLICY MIXTURE CONFIGURATION
    # ===================================================================================

    policies: List[Union[RandomPolicyConfig, DistancePolicyConfig, QLearningPolicyConfig, "ShinyPolicyConfig"]] = Field(description="Policies to mix")
    weights: List[float] = Field(description="Mixing weights (must sum to 1.0)")

    @model_validator(mode="after")
    def validate_weights(self) -> "MixedPolicyConfig":
        """Ensure weights sum to 1.0 and match policy count."""
        if len(self.weights) != len(self.policies):
            raise ValueError(f"Number of weights ({len(self.weights)}) must match number of policies ({len(self.policies)})")
        total = sum(self.weights)
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Policy weights must sum to 1.0, got {total}")
        return self

    # Note: Walk length is owned by DataModuleConfig.sequence_length.


# ===================================================================================
# SHINY GOAL-SWITCHING POLICY
# ===================================================================================


class ShinyPolicyConfig(BaseModel):
    """Goal-switching policy with shiny objects for reward-driven navigation.

    Automatically switches between multiple goal locations (shiny objects) with
    a lingering period at each goal.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["shiny"] = "shiny"

    # ===================================================================================
    # NAVIGATION BEHAVIOR
    # ===================================================================================

    beta: float = Field(default=1.5, gt=0, description="Softmax inverse temperature for action selection")

    # ===================================================================================
    # SHINY OBJECT PLACEMENT
    # ===================================================================================

    n: int = Field(default=2, ge=1, description="Number of shiny objects in environment")
    min_separation: float = Field(default=0.3, ge=0, le=1, description="Minimum distance ratio between shiny objects")

    # ===================================================================================
    # GOAL-SWITCHING BEHAVIOR
    # ===================================================================================

    returns: int = Field(default=15, ge=1, description="Steps to linger at shiny object before switching goals")

    # Note: Walk length is owned by DataModuleConfig.sequence_length.


# ===================================================================================
# POLICY UNION TYPE
# ===================================================================================

PolicyConfig = Union[RandomPolicyConfig, DistancePolicyConfig, QLearningPolicyConfig, ShinyPolicyConfig, MixedPolicyConfig]
