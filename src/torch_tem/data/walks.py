"""Walk sequence generation and batching."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.config import EnvironmentConfig
from torch_tem.data.environment import Environment, Location

from ..types import Vector


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

    def __init__(self, environment: Environment, repeat_bias: float = 2.0, env_config: EnvironmentConfig | None = None):
        """Initialize walk generator.

        Args:
            environment: Environment to generate walks in
            repeat_bias: Multiplicative bias for repeating previous action. If
                ``env_config`` is provided its ``explore_bias`` field is used
                as the default value when ``repeat_bias`` is left at the
                constructor default.
            env_config: Optional ``EnvironmentConfig`` driving exploration
                behaviour.
        """
        self.env = environment
        if env_config is not None and repeat_bias == 2.0:
            self.repeat_bias = env_config.explore_bias
        else:
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

            # Apply repeat bias to previous action if applicable
            if prev_action_id is not None and current_location_id != location_ids[-1]:
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
            observations: [batch, walk_length, n_observations]
            actions: [batch, walk_length]
            locations: [batch, walk_length]
        """
        observations = torch.stack([walk.observations for walk in walks])
        actions = torch.stack([walk.actions for walk in walks])
        locations = torch.stack([walk.locations for walk in walks])

        return observations, actions, locations
