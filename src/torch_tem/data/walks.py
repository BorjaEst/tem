"""Walk sequence generation and batching.

This module provides:

- `WalkGenerator`: generates synthetic trajectories (walks) through an
    `Environment`, optionally conditioned on a location policy.
- `WalkDataset`: a map-style dataset that generates a single walk per index.
- `collate_walk_samples`: a collate function that stacks individual walks into
    *time-major* batches for TEM training.

Time-major batch contract:
        - observations: float tensor of shape (T, B, n_x)
        - actions: int tensor of shape (T, B)
        - locations: int tensor of shape (T, B) (auxiliary)

Determinism:
        `WalkDataset.__getitem__` seeds NumPy and PyTorch RNGs using
        `seed + index` (or just `index` when no seed is provided). This makes items
        reproducible across runs, but will repeat the same samples across epochs
        unless an epoch component is added externally.
"""

from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import Dataset

from torch_tem.config.datamodule import DistancePolicyConfig, EnvironmentConfig, MixedPolicyConfig, PolicyConfig, QLearningPolicyConfig, RandomPolicyConfig, ShinyPolicyConfig
from torch_tem.data.environment import Environment, Location
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyEnvironmentBuilder
from torch_tem.types import Vector, WalkBatch, WalkSample


class Walk(BaseModel):
    """Single walk trajectory.

    Represents a sampled trajectory through the environment with observations,
    actions, and location markers.

    Attributes:
        observations: [walk_length, n_observations] one-hot tensors
        actions: [walk_length] action indices
        locations: [walk_length] location IDs (for analysis)
        shiny_markers: Optional[Tensor] [walk_length] boolean shiny flags
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    observations: Vector = Field(description="One-hot observation tensors")
    actions: Vector = Field(description="Action indices")
    locations: Vector = Field(description="Location IDs")
    shiny_markers: Optional[Tensor] = Field(default=None, description="Shiny location flags")

    def __len__(self) -> int:
        """Return walk length."""
        return len(self.observations)


class WalkGenerator:
    """Generates walk sequences from environment and policy.

    Supports standard exploration and shiny goal-switching modes.
    Applies repeat-action bias for straight-line movement.
    """

    def __init__(self, environment: Environment, repeat_bias: float = 2.0):
        """Initialize walk generator.

        Args:
            environment: Environment parameters implementing WalkEnvironmentParams
            repeat_bias: Bias factor for repeating previous action
        """
        self.env = environment
        self.repeat_bias = repeat_bias

    def generate_walk(self, walk_length: int, policy: Optional[List[Location]] = None) -> Walk:
        """Generate single walk from optional policy.

        Args:
            walk_length: Number of steps
            policy: Optional location-specific policies (uses env default if None)

        Returns:
            Walk: Generated walk trajectory
        """
        if policy is None:
            policy = self.env.locations

        observations = []
        actions = []
        location_ids = []
        shiny_flags = []

        # Start at random location
        current_location_id = np.random.randint(self.env.n_locations)
        prev_action_id = None

        for step in range(walk_length):
            current_location = policy[current_location_id]

            # Get observation at current location
            obs_id = current_location.observation
            obs_one_hot = torch.zeros(self.env.n_observations, dtype=torch.float)
            obs_one_hot[obs_id] = 1.0
            observations.append(obs_one_hot)

            # Record location ID and shiny flag
            location_ids.append(current_location_id)
            shiny_flags.append(current_location.shiny if current_location.shiny is not None else False)

            # Select action with repeat bias
            action_probs = np.array([action.probability for action in current_location.actions])

            # Apply repeat bias to previous action if we actually moved
            # (don't bias toward stay action or repeating when stuck at boundaries)
            if prev_action_id is not None and len(location_ids) > 1:
                if location_ids[-1] != location_ids[-2]:  # Only if moved
                    action_probs[prev_action_id] *= self.repeat_bias

            # Renormalize
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)
            else:
                # No valid actions, uniform
                action_probs = np.ones(len(action_probs)) / len(action_probs)

            # Sample action
            action_id = np.random.choice(len(action_probs), p=action_probs)
            actions.append(action_id)
            prev_action_id = action_id

            # Transition to next location
            transition_probs = current_location.actions[action_id].transition
            cumsum = np.cumsum(transition_probs)
            next_location_id = int(np.searchsorted(cumsum, np.random.rand()))
            current_location_id = next_location_id

        # Stack into tensors
        observations_tensor = torch.stack(observations)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        locations_tensor = torch.tensor(location_ids, dtype=torch.long)
        shiny_tensor = torch.tensor(shiny_flags, dtype=torch.bool)

        return Walk(observations=observations_tensor, actions=actions_tensor, locations=locations_tensor, shiny_markers=shiny_tensor if any(shiny_flags) else None)

    def generate_walks(self, n_walks: int, walk_length: int, policy: Optional[List[Location]] = None) -> List[Walk]:
        """Generate multiple independent walks.

        Args:
            n_walks: Number of walks to generate
            walk_length: Steps per walk
            policy: Optional policy

        Returns:
            List[Walk]: Generated walks
        """
        return [self.generate_walk(walk_length, policy) for _ in range(n_walks)]

    def generate_shiny_walk(self, walk_length: int, shiny_locations: List[int], shiny_policies: List[List[Location]], returns: int) -> Walk:
        """Generate walk with automatic goal switching.

        Agent approaches shiny objects sequentially, switching goals after
        reaching each object and lingering for 'returns' steps.

        Args:
            walk_length: Total walk steps
            shiny_locations: List of shiny object locations
            shiny_policies: Goal-directed policies for each shiny object
            returns: Steps to linger at each shiny before switching goals

        Returns:
            Walk: Generated walk with goal-switching behavior
        """
        n_shiny = len(shiny_locations)

        observations = []
        actions = []
        location_ids = []
        shiny_flags = []

        # Start at random location
        current_location_id = np.random.randint(self.env.n_locations)
        prev_action_id = None

        # Pick initial shiny target
        current_shiny_idx = np.random.randint(n_shiny)
        current_policy = shiny_policies[current_shiny_idx]
        shiny_returns_remaining = returns

        for step in range(walk_length):
            current_location = current_policy[current_location_id]

            # Check if reached current shiny target
            if current_location_id == shiny_locations[current_shiny_idx]:
                shiny_returns_remaining -= 1

            # Check if time to switch targets
            if shiny_returns_remaining < 0:
                current_shiny_idx = np.random.randint(n_shiny)
                current_policy = shiny_policies[current_shiny_idx]
                shiny_returns_remaining = returns

            # Get observation
            obs_id = current_location.observation
            obs_one_hot = torch.zeros(self.env.n_observations, dtype=torch.float)
            obs_one_hot[obs_id] = 1.0
            observations.append(obs_one_hot)

            # Record location and shiny flag
            location_ids.append(current_location_id)
            shiny_flags.append(current_location.shiny if current_location.shiny is not None else False)

            # Select action with repeat bias
            action_probs = np.array([action.probability for action in current_location.actions])

            # Apply repeat bias
            if prev_action_id is not None and len(location_ids) > 1:
                if location_ids[-1] != location_ids[-2]:  # Only if moved
                    action_probs[prev_action_id] *= self.repeat_bias

            # Renormalize
            if np.sum(action_probs) > 0:
                action_probs = action_probs / np.sum(action_probs)
            else:
                action_probs = np.ones(len(action_probs)) / len(action_probs)

            # Sample action
            action_id = np.random.choice(len(action_probs), p=action_probs)
            actions.append(action_id)
            prev_action_id = action_id

            # Transition
            transition_probs = current_location.actions[action_id].transition
            cumsum = np.cumsum(transition_probs)
            next_location_id = int(np.searchsorted(cumsum, np.random.rand()))
            current_location_id = next_location_id

        # Stack into tensors
        observations_tensor = torch.stack(observations)
        actions_tensor = torch.tensor(actions, dtype=torch.long)
        locations_tensor = torch.tensor(location_ids, dtype=torch.long)
        shiny_tensor = torch.tensor(shiny_flags, dtype=torch.bool)

        return Walk(observations=observations_tensor, actions=actions_tensor, locations=locations_tensor, shiny_markers=shiny_tensor)

    def generate_shiny_walks(self, n_walks: int, walk_length: int, shiny_locations: List[int], shiny_policies: List[List[Location]], returns: int) -> List[Walk]:
        """Generate multiple shiny walks.

        Args:
            n_walks: Number of walks
            walk_length: Steps per walk
            shiny_locations: Shiny object locations
            shiny_policies: Policies for each shiny object
            returns: Linger steps at shiny objects

        Returns:
            List[Walk]: Generated shiny walks
        """
        return [self.generate_shiny_walk(walk_length, shiny_locations, shiny_policies, returns) for _ in range(n_walks)]

    def batch_walks(self, walks: List[Walk]) -> Tuple[Vector, Vector, Vector]:
        """Stack walks into batched tensors for TEM.

        Args:
            walks: List of Walk objects

        Returns:
            observations: [walk_length, batch, n_observations]
            actions: [walk_length, batch]
            locations: [walk_length, batch]
        """
        # Stack walks: [batch, walk_length, ...] then transpose to [walk_length, batch, ...]
        observations = torch.stack([walk.observations for walk in walks]).transpose(0, 1)
        actions = torch.stack([walk.actions for walk in walks]).transpose(0, 1)
        locations = torch.stack([walk.locations for walk in walks]).transpose(0, 1)

        return observations, actions, locations


def collate_walk_samples(samples: List[WalkSample], *, return_locations: bool) -> WalkBatch:
    """Collate unbatched walks into a time-major TEM batch.

    Each sample contains:
        - observations: [T, n_observations]
        - actions: [T]
        - locations: [T]

    Returns:
        - observations: [T, B, n_observations]
        - actions: [T, B]
        - locations: [T, B] (or zeros if return_locations=False)
    """

    observations = torch.stack([s[0] for s in samples]).transpose(0, 1)
    actions = torch.stack([s[1] for s in samples]).transpose(0, 1)
    locations = torch.stack([s[2] for s in samples]).transpose(0, 1)

    if not return_locations:
        locations = torch.zeros_like(actions)

    return observations, actions, locations


class WalkDatasetParams(Protocol):
    """Minimal parameter interface required by `WalkDataset`.

    This is intentionally defined as a Protocol so that configs (e.g.
    `DataModuleConfig`) can be passed directly, as long as they provide these
    attributes.

    Attributes:
        policy: Policy configuration used to sample actions.
        sequence_length: Number of timesteps per generated walk.
        seed: Base seed for deterministic item generation.
    """

    policy: PolicyConfig
    environment: EnvironmentConfig
    sequence_length: int
    seed: Optional[int]


class WalkDataset(Dataset[WalkSample]):
    """Map-style dataset that generates single walks on demand.

    Each dataset item is an *unbatched* walk (a `WalkSample`). Batching is
    performed by the DataLoader via `collate_walk_samples`.
    """

    def __init__(self, n_items: int, env: Environment, policy_gen: PolicyGenerator, walk_gen: WalkGenerator, params: WalkDatasetParams):
        """Initialize the dataset.

        Args:
            n_items: Total number of items the dataset will expose.
            env: Runtime environment used for walk generation.
            policy_gen: Policy generator bound to the environment.
            walk_gen: Walk generator bound to the environment.
            params: Configuration-like object providing `policy`,
                `sequence_length`, and an optional `seed`.
        """
        super().__init__()
        self._n_items = int(n_items)
        self._env = env
        self._policy_gen = policy_gen
        self._walk_gen = walk_gen
        self._policy_cfg = params.policy
        self._env_cfg = params.environment
        self._sequence_length = int(params.sequence_length)
        self._seed = params.seed

    def __len__(self) -> int:  # type: ignore[override]
        return self._n_items

    def __getitem__(self, index: int) -> WalkSample:
        """Generate a single walk for the given index.

        Args:
            index: Dataset index.

        Returns:
            A tuple `(observations, actions, locations)` with shapes:
            - observations: (T, n_x)
            - actions: (T,)
            - locations: (T,)
        """
        base_seed = 0 if self._seed is None else int(self._seed)
        seed = base_seed + int(index)

        # Ensure reproducible generation per item (including across workers).
        np.random.seed(seed)
        torch.manual_seed(seed)

        walk = self._generate_walk(self._sequence_length)
        return walk.observations, walk.actions, walk.locations

    def _generate_walk(self, t_steps: int) -> Walk:
        policy_cfg = self._policy_cfg

        if isinstance(policy_cfg, ShinyPolicyConfig):
            builder = ShinyEnvironmentBuilder(self._env, self._env_cfg, policy_cfg, self._policy_gen)
            shiny_locations = builder.place_shiny_objects()
            builder.mark_environment(shiny_locations)
            shiny_policies = builder.generate_shiny_policies(shiny_locations)
            return self._walk_gen.generate_shiny_walk(
                walk_length=t_steps,
                shiny_locations=shiny_locations,
                shiny_policies=shiny_policies,
                returns=self._env_cfg.shiny_returns,
            )

        policy = self._create_location_policy(self._policy_gen, policy_cfg)
        return self._walk_gen.generate_walk(walk_length=t_steps, policy=policy)

    def _create_location_policy(self, policy_gen: PolicyGenerator, policy_cfg: PolicyConfig) -> List[Location]:
        if isinstance(policy_cfg, RandomPolicyConfig):
            return policy_gen.random_policy()

        if isinstance(policy_cfg, DistancePolicyConfig):
            goals = self._sample_goals(policy_cfg.goal_mode, policy_cfg.n_goals)
            return policy_gen.distance_policy(goal_locations=goals, beta=policy_cfg.beta)

        if isinstance(policy_cfg, QLearningPolicyConfig):
            goals = self._sample_goals(policy_cfg.goal_mode, policy_cfg.n_goals)
            return policy_gen.q_learning_policy(
                goal_locations=goals,
                gamma=policy_cfg.gamma,
                beta=policy_cfg.beta,
                n_iterations=policy_cfg.n_iterations,
            )

        if isinstance(policy_cfg, MixedPolicyConfig):
            idx = int(np.random.choice(len(policy_cfg.policies), p=np.array(policy_cfg.weights)))
            return self._create_location_policy(policy_gen, policy_cfg.policies[idx])

        raise NotImplementedError(f"Unsupported policy type: {getattr(policy_cfg, 'type', type(policy_cfg))}")

    def _sample_goals(self, goal_mode: str, n_goals: int) -> List[int]:
        if goal_mode == "fixed":
            return list(range(n_goals))
        return np.random.choice(self._env.n_locations, size=n_goals, replace=False).tolist()
