"""PyTorch Lightning DataModule for TEM training."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple, Literal, Union

import lightning as L
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset

from torch_tem.config import EnvironmentConfig
from torch_tem.data.environment import Environment, Location
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyConfig, ShinyEnvironmentBuilder
from torch_tem.data.walks import Walk, WalkGenerator
from torch_tem.types import Vector


class TEMDataParams(Protocol):
    """Protocol defining parameters for TEM DataModule configuration."""
    
    env: Environment
    batch_size: int
    walk_length: int
    env_config: Optional[EnvironmentConfig]
    shiny_config: Optional[ShinyConfig]
    policy_type: Literal["random", "distance", "q_learning"]
    repeat_bias: float
    num_workers: int
    pin_memory: bool


@dataclass
class TEMDataParamsImpl:
    """Concrete implementation of TEMDataParams protocol.
    
    This class can be instantiated directly or created from a DataModuleConfig.
    """
    
    env: Environment
    batch_size: int
    walk_length: int
    env_config: Optional[EnvironmentConfig] = None
    shiny_config: Optional[ShinyConfig] = None
    policy_type: Literal["random", "distance", "q_learning"] = "random"
    repeat_bias: float = 2.0
    num_workers: int = 0
    pin_memory: bool = False

    @classmethod
    def from_config(
        cls,
        env: Environment,
        datamodule_config,
        env_config: Optional[EnvironmentConfig] = None,
        shiny_config: Optional[ShinyConfig] = None,
    ) -> "TEMDataParamsImpl":
        """Create from DataModuleConfig.
        
        Args:
            env: Environment instance
            datamodule_config: DataModuleConfig instance
            env_config: Optional environment configuration
            shiny_config: Optional shiny configuration
            
        Returns:
            TEMDataParamsImpl instance
        """
        return cls(
            env=env,
            batch_size=datamodule_config.batch_size,
            walk_length=datamodule_config.walk_length,
            env_config=env_config,
            shiny_config=shiny_config,
            policy_type=datamodule_config.policy_type,
            repeat_bias=datamodule_config.repeat_bias,
            num_workers=datamodule_config.num_workers,
            pin_memory=datamodule_config.pin_memory,
        )


class InfiniteWalkDataset(IterableDataset):
    """Infinite dataset that generates walks on-demand.
    
    Each iteration generates a fresh batch of walks from the environment,
    supporting TEM's infinite training paradigm.
    """

    def __init__(
        self,
        walk_generator: WalkGenerator,
        policy: List[Location],
        batch_size: int,
        walk_length: int,
        shiny_locations: Optional[List[int]] = None,
        shiny_policies: Optional[List[List[Location]]] = None,
        shiny_returns: int = 5,
    ):
        """Initialize infinite walk dataset.
        
        Args:
            walk_generator: WalkGenerator instance
            policy: Action policy for each location
            batch_size: Walks per batch
            walk_length: Steps per walk
            shiny_locations: Optional shiny object locations for goal-directed walks
            shiny_policies: Optional policies for shiny object navigation
            shiny_returns: Steps to linger at shiny objects before switching goals
        """
        super().__init__()
        self.walk_gen = walk_generator
        self.policy = policy
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.shiny_locations = shiny_locations
        self.shiny_policies = shiny_policies
        self.shiny_returns = shiny_returns

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        """Generate infinite batches of walks."""
        while True:
            # Generate walks based on mode
            if self.shiny_locations is not None and self.shiny_policies is not None:
                walks = self.walk_gen.generate_shiny_walks(
                    n_walks=self.batch_size,
                    walk_length=self.walk_length,
                    shiny_locations=self.shiny_locations,
                    shiny_policies=self.shiny_policies,
                    returns=self.shiny_returns,
                )
            else:
                walks = self.walk_gen.generate_walks(
                    n_walks=self.batch_size,
                    walk_length=self.walk_length,
                    policy=self.policy,
                )

            # Batch and yield
            yield self.walk_gen.batch_walks(walks)


class TEMDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for TEM training with infinite walk generation.
    
    Unlike traditional DataModules with fixed train/val/test splits, TEMDataModule
    generates walks on-demand infinitely, supporting TEM's training paradigm where
    each batch is a fresh sample from the environment.
    
    Supports:
    - Random exploration with configurable repeat bias
    - Multiple policy types (random, distance-based, Q-learning)
    - Shiny object goal-directed navigation
    - Curriculum learning through dynamic walk lengths
    """

    def __init__(self, params: TEMDataParams):
        """Initialize TEM DataModule.
        
        Args:
            params: TEMDataParams protocol with all configuration parameters
        """
        super().__init__()
        self.params = params

        # Initialize generators
        self.walk_gen = WalkGenerator(
            params.env, 
            repeat_bias=params.repeat_bias, 
            env_config=params.env_config
        )
        self.policy_gen = PolicyGenerator(params.env)

        # Generate policy based on type
        if params.policy_type == "random":
            self.policy = self.policy_gen.random_policy()
        elif params.policy_type == "distance":
            # Distance-based policy requires a goal location
            # Use center of environment as default goal
            goal_location = params.env.n_locations // 2
            self.policy = self.policy_gen.distance_policy(goal_location)
        elif params.policy_type == "q_learning":
            # Q-learning policy requires reward locations
            # Use shiny locations if available, otherwise use random location
            if params.shiny_config is not None:
                reward_locations = params.shiny_config.shiny_locations
            else:
                reward_locations = [params.env.n_locations // 2]
            self.policy = self.policy_gen.q_learning_policy(reward_locations)
        else:
            raise ValueError(f"Unknown policy type: {params.policy_type}")

        # Setup shiny configuration if provided
        self.shiny_locations = None
        self.shiny_policies = None
        if params.shiny_config is not None:
            self.shiny_locations = params.shiny_config.shiny_locations
            # Generate goal-directed policies for each shiny object
            self.shiny_policies = [
                self.policy_gen.distance_policy(shiny_loc) 
                for shiny_loc in params.shiny_config.shiny_locations
            ]

    def generate_batch(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate single batch of walks.
        
        Returns:
            observations: [walk_length, batch_size, n_observations]
            actions: [walk_length, batch_size]
            locations: [walk_length, batch_size]
        """
        if self.shiny_locations is not None and self.shiny_policies is not None:
            walks = self.walk_gen.generate_shiny_walks(
                n_walks=self.params.batch_size,
                walk_length=self.params.walk_length,
                shiny_locations=self.shiny_locations,
                shiny_policies=self.shiny_policies,
                returns=self.params.shiny_config.shiny_returns if self.params.shiny_config else 5,
            )
        else:
            walks = self.walk_gen.generate_walks(
                n_walks=self.params.batch_size,
                walk_length=self.params.walk_length,
                policy=self.policy,
            )

        return self.walk_gen.batch_walks(walks)

    def train_dataloader(self) -> DataLoader:
        """Create infinite training dataloader.
        
        Returns:
            DataLoader that generates walks infinitely
        """
        dataset = InfiniteWalkDataset(
            walk_generator=self.walk_gen,
            policy=self.policy,
            batch_size=self.params.batch_size,
            walk_length=self.params.walk_length,
            shiny_locations=self.shiny_locations,
            shiny_policies=self.shiny_policies,
            shiny_returns=self.params.shiny_config.shiny_returns if self.params.shiny_config else 5,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=None,  # Batching handled by dataset
            num_workers=self.params.num_workers,
            pin_memory=self.params.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader.
        
        For reproducible validation, you may want to set a fixed random seed
        before calling this method.
        
        Returns:
            DataLoader for validation (also infinite)
        """
        return self.train_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Create test dataloader.
        
        Returns:
            DataLoader for testing (also infinite)
        """
        return self.train_dataloader()

    def update_walk_length(self, new_length: int):
        """Update walk length for curriculum learning.
        
        Args:
            new_length: New walk length for future batches
        """
        self.params.walk_length = new_length

    def update_policy(self, new_policy_type: Literal["random", "distance", "q_learning"], **kwargs):
        """Update policy during training.
        
        Args:
            new_policy_type: New policy type
            **kwargs: Additional arguments for policy generation (e.g., goal_location for distance)
        """
        self.params.policy_type = new_policy_type

        if new_policy_type == "random":
            self.policy = self.policy_gen.random_policy()
        elif new_policy_type == "distance":
            goal_location = kwargs.get("goal_location", self.params.env.n_locations // 2)
            self.policy = self.policy_gen.distance_policy(goal_location)
        elif new_policy_type == "q_learning":
            reward_locations = kwargs.get("reward_locations", [self.params.env.n_locations // 2])
            self.policy = self.policy_gen.q_learning_policy(reward_locations)
        else:
            raise ValueError(f"Unknown policy type: {new_policy_type}")
