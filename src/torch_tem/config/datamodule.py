"""DataModule configuration for the Temporal Experience Model (TEM).

This config is the *single source of truth* for data shapes and dataloader
behavior.

In particular, when training TEM with (truncated) BPTT, the canonical batch
layout is time-major full walks:

- observations: float32 tensor [T, B, n_x]
- actions: int64 tensor [T, B]
- locations: int64 tensor [T, B] (auxiliary)

The walk length T is defined solely by :attr:`sequence_length`.
"""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from torch_tem.config.architecture import ModelConfig


class EnvironmentConfig(BaseModel):
    """Environment and task configuration: action space, exploration, and reward-driven behaviour.

    This defines the task distribution the model is trained on but does not affect model structure.
    """

    model_config = ConfigDict(extra="ignore", strict=False)

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

    # ===================================================================================
    # HELPER PROPERTIES
    # ===================================================================================

    def build_model_config(self, base: Optional[ModelConfig] = None, **overrides) -> ModelConfig:
        """Construct ModelConfig matching this environment configuration.

        Args:
            base: Optional base ModelConfig to override. If None, uses default ModelConfig.
            **overrides: Additional ModelConfig fields to override.

        Returns:
            ModelConfig instance with n_locations and n_observations set.
        """
        base = base or ModelConfig()
        model_dict = base.model_dump()
        n_actions = self.n_actions + (1 if self.has_static_action else 0)
        model_dict.update({"n_x": self.n_observations, "n_actions": n_actions})
        model_dict.update(overrides)
        return ModelConfig.model_validate(model_dict)


# ===================================================================================
# RANDOM EXPLORATION POLICY
# ===================================================================================
class RandomPolicyConfig(BaseModel):
    """Random uniform action selection policy for unbiased spatial exploration.

    This policy selects actions uniformly at random, providing comprehensive coverage
    of the environment without goal-directed behavior. Uses longer walk lengths to
    ensure adequate spatial sampling.
    """

    model_config = ConfigDict(extra="ignore", strict=False)

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

    model_config = ConfigDict(extra="ignore", strict=False)

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

    model_config = ConfigDict(extra="ignore", strict=False)

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

    model_config = ConfigDict(extra="ignore", strict=False)

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

    model_config = ConfigDict(extra="ignore", strict=False)

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


class DataModuleConfig(BaseModel):
    """DataModule configuration: environment, walks, policies, batching, and data splits.

    This is the single source of truth for all data generation parameters.
    Defines how training/validation/test data is generated and delivered to the model.
    """

    model_config = ConfigDict(extra="ignore", strict=False)

    # ===================================================================================
    # ENVIRONMENT and POLICY
    # ===================================================================================

    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration")
    policy: PolicyConfig = Field(default_factory=RandomPolicyConfig, description="Action selection policy for walk generation")

    # ===================================================================================
    # SEQUENCE SHAPE (BPTT)
    # ===================================================================================

    sequence_length: int = Field(default=100, ge=1, description="Number of timesteps per walk (T). Sole source of sequence length for the DataModule.")
    return_locations: bool = Field(default=True, description="If True, include location IDs as third element of the batch tuple.")

    # ===================================================================================
    # DATA SPLITS
    # ===================================================================================

    n_train_batches: int = Field(default=1000, ge=1, description="Number of training batches per epoch")
    n_val_batches: int = Field(default=10, ge=0, description="Number of validation batches")
    n_test_batches: int = Field(default=10, ge=0, description="Number of test batches")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducible environment/walk generation")

    # ===================================================================================
    # DATALOADER SETTINGS
    # ===================================================================================

    batch_size: int = Field(default=16, ge=1, description="Number of walks per batch")
    num_workers: int = Field(default=0, ge=0, description="Number of DataLoader worker processes (0 = main process only)")
    pin_memory: bool = Field(default=False, description="Pin memory for faster GPU transfer")
    drop_last: bool = Field(default=False, description="Drop last incomplete batch")

    # ===================================================================================
    # HELPER PROPERTIES
    # ===================================================================================

    def build_model_config(self, base: Optional[ModelConfig] = None, **overrides) -> ModelConfig:
        """Construct ModelConfig matching this DataModule configuration.

        Args:
            base: Optional base ModelConfig to override. If None, uses default ModelConfig.
            **overrides: Additional ModelConfig fields to override.

        Returns:
            ModelConfig instance with n_locations and n_observations set.
        """
        overrides["batch_size"] = self.batch_size
        return self.environment.build_model_config(base=base, **overrides)
