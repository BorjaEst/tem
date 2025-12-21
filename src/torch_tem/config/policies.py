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

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["random"] = "random"

    # ===================================================================================
    # WALK LENGTH CURRICULUM (long walks for comprehensive exploration)
    # ===================================================================================

    walk_length_min: int = Field(default=50, ge=1, description="Minimum walk length (curriculum end target)")
    walk_length_max: int = Field(default=300, ge=1, description="Maximum walk length (curriculum start target)")
    walk_length_curriculum: bool = Field(default=True, description="Enable walk length curriculum")
    walk_length_curriculum_steps: Optional[int] = Field(default=None, ge=1, description="Steps to complete curriculum (default: n_train_batches)")


# ===================================================================================
# DISTANCE-BASED GOAL-DIRECTED POLICY
# ===================================================================================


class DistancePolicyConfig(BaseModel):
    """Distance-based goal-directed policy using graph shortest paths.

    Navigates to goal locations using softmax action selection over graph distances.
    Actions are weighted by their progress toward the goal along the shortest path.
    Uses shorter walk lengths since optimal paths reach goals efficiently.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

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

    # ===================================================================================
    # WALK LENGTH CURRICULUM (short walks - goals reached quickly)
    # ===================================================================================

    walk_length_min: int = Field(default=25, ge=1, description="Minimum walk length (curriculum end target)")
    walk_length_max: int = Field(default=100, ge=1, description="Maximum walk length (curriculum start target)")
    walk_length_curriculum: bool = Field(default=True, description="Enable walk length curriculum")
    walk_length_curriculum_steps: Optional[int] = Field(default=None, ge=1, description="Steps to complete curriculum (default: n_train_batches)")


# ===================================================================================
# Q-LEARNING GOAL-DIRECTED POLICY
# ===================================================================================


class QLearningPolicyConfig(BaseModel):
    """Q-learning value iteration policy for goal-directed navigation.

    Computes optimal action-value functions via value iteration and selects actions
    using softmax over Q-values. Supports discounted rewards for multi-step planning.
    Uses shorter walk lengths since optimal policies reach goals efficiently.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

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

    # ===================================================================================
    # WALK LENGTH CURRICULUM (short walks - goals reached quickly)
    # ===================================================================================

    walk_length_min: int = Field(default=25, ge=1, description="Minimum walk length (curriculum end target)")
    walk_length_max: int = Field(default=100, ge=1, description="Maximum walk length (curriculum start target)")
    walk_length_curriculum: bool = Field(default=True, description="Enable walk length curriculum")
    walk_length_curriculum_steps: Optional[int] = Field(default=None, ge=1, description="Steps to complete curriculum (default: n_train_batches)")


# ===================================================================================
# MIXED POLICY
# ===================================================================================


class MixedPolicyConfig(BaseModel):
    """Weighted mixture of multiple policies for diverse behavior.

    Combines multiple policies with specified weights, sampling from each according
    to the weight distribution. Walk length parameters can be inherited from the
    first policy or explicitly overridden for the mixture.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

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

    # ===================================================================================
    # HELPER METHODS
    # ===================================================================================

    @property
    def walk_length_min(self) -> int:
        """Get effective walk_length_min."""
        return min(policy.walk_length_min for policy in self.policies)

    @property
    def walk_length_max(self) -> int:
        """Get effective walk_length_max."""
        return max(policy.walk_length_max for policy in self.policies)

    @property
    def walk_length_curriculum(self) -> bool:
        """Get effective walk_length_curriculum."""
        return any(policy.walk_length_curriculum for policy in self.policies)

    @property
    def walk_length_curriculum_steps(self) -> Optional[int]:
        """Get effective walk_length_curriculum_steps."""
        steps = [policy.walk_length_curriculum_steps for policy in self.policies if policy.walk_length_curriculum_steps is not None]
        return max(steps) if steps else None


# ===================================================================================
# SHINY GOAL-SWITCHING POLICY
# ===================================================================================


class ShinyPolicyConfig(BaseModel):
    """Goal-switching policy with shiny objects for reward-driven navigation.

    Automatically switches between multiple goal locations (shiny objects) with
    a lingering period at each goal. Uses distance-based or Q-learning navigation
    between goals. Medium walk lengths accommodate multiple goal visits within
    a single episode.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    # ===================================================================================
    # POLICY TYPE
    # ===================================================================================

    type: Literal["shiny"] = "shiny"

    # ===================================================================================
    # NAVIGATION BEHAVIOR
    # ===================================================================================

    navigation_policy: Literal["distance", "q_learning"] = Field(default="distance", description="Policy type for navigation to shiny objects")
    beta: float = Field(default=1.5, gt=0, description="Softmax inverse temperature for action selection")
    gamma: float = Field(default=0.7, ge=0, le=1, description="Discount factor (for Q-learning navigation)")

    # ===================================================================================
    # SHINY OBJECT PLACEMENT
    # ===================================================================================

    n: int = Field(default=2, ge=1, description="Number of shiny objects in environment")
    min_separation: float = Field(default=0.3, ge=0, le=1, description="Minimum distance ratio between shiny objects")

    # ===================================================================================
    # GOAL-SWITCHING BEHAVIOR
    # ===================================================================================

    returns: int = Field(default=15, ge=1, description="Steps to linger at shiny object before switching goals")

    # ===================================================================================
    # WALK LENGTH CURRICULUM (medium walks for multiple goal visits)
    # ===================================================================================

    walk_length_max: int = Field(default=200, ge=1, description="Maximum walk length (curriculum start target)")
    walk_length_curriculum: bool = Field(default=True, description="Enable walk length curriculum")
    walk_length_curriculum_steps: Optional[int] = Field(default=None, ge=1, description="Steps to complete curriculum (default: n_train_batches)")


# ===================================================================================
# POLICY UNION TYPE
# ===================================================================================

PolicyConfig = Union[RandomPolicyConfig, DistancePolicyConfig, QLearningPolicyConfig, ShinyPolicyConfig, MixedPolicyConfig]
