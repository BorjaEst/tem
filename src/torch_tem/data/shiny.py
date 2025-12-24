"""Shiny object placement and policy generation.

This module intentionally does *not* define a separate ShinyConfig model.
Shiny-related scalar parameters live in :class:`torch_tem.config.EnvironmentConfig`.
"""

from typing import List, Optional

import numpy as np

from torch_tem.config import EnvironmentConfig, ShinyPolicyConfig
from torch_tem.data.environment import Environment, EnvLocation
from torch_tem.data.policies import PolicyGenerator


class ShinyEnvironmentBuilder:
    """Augments environments with shiny object placement.

    Handles shiny object placement with minimum separation constraints,
    observation deduplication, and shiny-directed policy generation.
    """

    def __init__(
        self,
        environment: Environment,
        env_config: EnvironmentConfig,
        policy_config: ShinyPolicyConfig,
        policy_generator: PolicyGenerator,
        rng: Optional[np.random.Generator] = None,
    ):
        """Initialize shiny environment builder.

        Args:
            environment: Runtime environment to augment.
            env_config: Environment configuration (source of shiny parameters).
            policy_config: Shiny policy configuration (currently only used for
                placement constraints such as minimum separation).
            policy_generator: Policy generator for shiny-directed policies.
        """
        self.env = environment
        self._env_cfg = env_config
        self._policy_cfg = policy_config
        self.policy_gen = policy_generator
        self._rng = rng or np.random.default_rng()

    def place_shiny_objects(self) -> List[int]:
        """Place shiny objects with minimum separation constraint.

        Uses graph distances to ensure shiny objects are sufficiently spread out.
        Sampling is rejection-based: repeatedly sample locations until separation
        constraint is satisfied.

        Returns:
            List[int]: Location IDs designated as shiny

        Raises:
            RuntimeError: If unable to place all shiny objects after max attempts
        """
        n_shiny = int(self._env_cfg.shiny_n)
        if n_shiny <= 0:
            raise ValueError("shiny_n must be > 0 when using ShinyPolicyConfig")

        dist_matrix = self.env.shortest_paths()
        max_distance = np.max(dist_matrix)
        min_distance_threshold = max_distance * float(self._policy_cfg.min_separation)

        shiny_locations = []
        max_attempts = 1000

        for _ in range(n_shiny):
            attempts = 0
            while attempts < max_attempts:
                # Sample candidate location
                candidate = int(self._rng.integers(self.env.n_locations))

                # Check separation from existing shiny locations
                if not shiny_locations:
                    # First shiny object, no separation check needed
                    shiny_locations.append(candidate)
                    break

                # Check if candidate is far enough from all existing shiny locations
                too_close = any(dist_matrix[candidate, existing] < min_distance_threshold for existing in shiny_locations)

                if not too_close:
                    shiny_locations.append(candidate)
                    break

                attempts += 1

            if attempts >= max_attempts:
                raise RuntimeError(
                    f"Could not place {n_shiny} shiny objects with "
                    f"min_separation={self._policy_cfg.min_separation} after {max_attempts} attempts. "
                    f"Try reducing n or min_separation."
                )

        return shiny_locations

    def mark_environment(self, shiny_locations: List[int]) -> Environment:
        """Update environment with shiny markers and deduplicate observations.

        Marks shiny locations and ensures shiny observations don't appear at
        non-shiny locations (makes shiny objects distinctive).

        Args:
            shiny_locations: List of location IDs to mark as shiny

        Returns:
            Environment: Updated environment with shiny markers
        """
        # Get shiny object observations
        shiny_objects = [self.env.locations[loc_id].observation for loc_id in shiny_locations]

        # Get non-shiny observations
        non_shiny_objects = [obs for obs in range(self.env.n_observations) if obs not in shiny_objects]

        if not non_shiny_objects:
            raise ValueError("All observations are used by shiny objects. " "Increase n_observations or reduce number of shiny objects.")

        # Update locations
        new_locations = []
        for loc_id, location in enumerate(self.env.locations):
            is_shiny = loc_id in shiny_locations

            # If non-shiny location has shiny observation, replace it
            if not is_shiny and location.observation in shiny_objects:
                new_observation = int(self._rng.choice(non_shiny_objects))
                new_location = location.model_copy(update={"observation": new_observation, "shiny": False})
            else:
                new_location = location.model_copy(update={"shiny": is_shiny})

            new_locations.append(new_location)

        # Update environment
        self.env.locations = new_locations
        return self.env

    def generate_shiny_policies(self, shiny_locations: List[int]) -> List[List[EnvLocation]]:
        """Generate goal-directed policy for each shiny object.

        Creates one distance-based policy per shiny location for goal switching.

        Args:
            shiny_locations: List of shiny location IDs

        Returns:
            List[List[EnvLocation]]: One policy per shiny object
        """
        beta = float(self._env_cfg.shiny_beta)
        return [self.policy_gen.distance_policy(goal_locations=shiny_loc, beta=beta) for shiny_loc in shiny_locations]
