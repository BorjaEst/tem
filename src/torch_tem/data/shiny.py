"""Shiny object configuration and placement."""

import copy
from typing import List

import numpy as np
from pydantic import BaseModel, Field, model_validator

from torch_tem.config import EnvironmentConfig
from torch_tem.data.environment import Environment, Location
from torch_tem.data.policies import PolicyGenerator


class ShinyConfig(BaseModel):
    """Configuration for shiny object environments.

    Shiny objects are special reward locations that drive goal-directed
    behavior with automatic goal switching during walks.
    """

    n: int = Field(gt=0, description="Number of shiny objects")
    returns: int = Field(gt=0, description="Steps to linger at shiny object before switching")
    gamma: float = Field(default=0.9, ge=0.0, le=1.0, description="Discount factor for Q-learning")
    beta: float = Field(default=1.0, gt=0.0, description="Softmax temperature")
    min_separation: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum distance ratio between shiny objects")

    @model_validator(mode="after")
    def validate_feasibility(self) -> "ShinyConfig":
        """Check if configuration is internally consistent."""
        if self.n > 1 and self.min_separation > 0.9:
            raise ValueError("Cannot place multiple shiny objects with min_separation > 0.9")
        return self

    @classmethod
    def from_environment_config(cls, env_config: EnvironmentConfig, min_separation: float = 0.3) -> "ShinyConfig":
        """Create shiny configuration from an ``EnvironmentConfig``.

        Args:
            env_config: Global environment configuration.
            min_separation: Minimum distance ratio between shiny objects.

        Returns:
            ShinyConfig: Constructed shiny configuration.
        """
        return cls(
            n=env_config.shiny_n,
            returns=env_config.shiny_returns,
            gamma=env_config.shiny_gamma,
            beta=env_config.shiny_beta,
            min_separation=min_separation,
        )


class ShinyEnvironmentBuilder:
    """Augments environments with shiny object placement.

    Handles shiny object placement with minimum separation constraints,
    observation deduplication, and shiny-directed policy generation.
    """

    def __init__(self, environment: Environment, shiny_config: ShinyConfig, policy_generator: PolicyGenerator):
        """Initialize shiny environment builder.

        Args:
            environment: Base environment to augment
            shiny_config: Shiny object configuration
            policy_generator: Policy generator for shiny-directed policies
        """
        self.env = environment
        self.config = shiny_config
        self.policy_gen = policy_generator

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
        dist_matrix = self.env.shortest_paths()
        max_distance = np.max(dist_matrix)
        min_distance_threshold = max_distance * self.config.min_separation

        shiny_locations = []
        max_attempts = 1000

        for _ in range(self.config.n):
            attempts = 0
            while attempts < max_attempts:
                # Sample candidate location
                candidate = np.random.randint(self.env.n_locations)

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
                    f"Could not place {self.config.n} shiny objects with "
                    f"min_separation={self.config.min_separation} after {max_attempts} attempts. "
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
                new_observation = np.random.choice(non_shiny_objects)
                new_location = location.model_copy(update={"observation": new_observation, "shiny": False})
            else:
                new_location = location.model_copy(update={"shiny": is_shiny})

            new_locations.append(new_location)

        # Update environment
        self.env.locations = new_locations
        return self.env

    def generate_shiny_policies(self, shiny_locations: List[int]) -> List[List[Location]]:
        """Generate goal-directed policy for each shiny object.

        Creates one distance-based policy per shiny location for goal switching.

        Args:
            shiny_locations: List of shiny location IDs

        Returns:
            List[List[Location]]: One policy per shiny object
        """
        return [self.policy_gen.distance_policy(goal_locations=shiny_loc, beta=self.config.beta) for shiny_loc in shiny_locations]
