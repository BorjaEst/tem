"""PyTorch Lightning DataModule for TEM training."""

from typing import Callable, Dict, List, Optional, Tuple

import lightning as L
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from torch_tem.config import EnvironmentConfig
from torch_tem.data.environment import Environment, EnvironmentParams, Location
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyConfig, ShinyEnvironmentBuilder
from torch_tem.data.walks import Walk, WalkGenerator


class InfiniteWalkDataset(IterableDataset):
    """Infinite stream of walks for training.

    Generates walks on-the-fly without pre-caching, enabling infinite
    training data with curriculum learning support.
    """

    def __init__(self, datamodule: "TEMDataModule"):
        """Initialize infinite walk dataset.

        Args:
            datamodule: Parent datamodule providing generation methods
        """
        self.dm = datamodule
        self.epoch = 0

    def __iter__(self):
        """Yield walks indefinitely."""
        while True:
            # Generate single walk
            if self.dm.shiny_config is None:
                walk = self.dm.walk_gen.generate_walk(self.dm.walk_length, self.dm._get_current_policy(self.epoch))
            else:
                walk = self.dm.walk_gen.generate_shiny_walk(self.dm.walk_length, self.dm.shiny_locations, self.dm.shiny_policies, self.dm.shiny_config.returns)
            yield walk


class TEMDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for TEM training.

    Generates infinite stream of walks without pre-caching.
    Supports curriculum learning via epoch-dependent policy mixing.
    Integrates environment, policy, and walk generation.
    """

    def __init__(
        self,
        env: Environment,
        batch_size: int,
        walk_length: int,
        shiny_config: Optional[ShinyConfig] = None,
        randomize_observations: bool = False,
        repeat_bias: float = 2.0,
        env_config: Optional[EnvironmentConfig] = None,
        curriculum_schedule: Optional[Callable[[int], Dict]] = None,
        num_workers: int = 0,
    ):
        """Initialize TEM DataModule.

        Args:
            env: Pre-built ``Environment`` instance
            batch_size: Walks per batch
            walk_length: Steps per walk
            shiny_config: Optional shiny object configuration
            randomize_observations: Shuffle observation assignments
            repeat_bias: Action repeat bias for straight-line movement. If
                ``env_config`` is provided and ``repeat_bias`` is left at its
                default, ``env_config.explore_bias`` is used instead.
            env_config: Optional ``EnvironmentConfig`` controlling exploration
                and shiny behaviour.
            curriculum_schedule: Optional epoch -> policy_params mapping
            num_workers: Number of dataloader workers
        """
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.walk_length = walk_length
        self.repeat_bias = repeat_bias
        self.curriculum_schedule = curriculum_schedule
        self.num_workers = num_workers
        self.env_config = env_config

        # Validate environment
        self.env.validate()

        # Build policy generator
        self.policy_gen = PolicyGenerator(self.env)

        # Setup shiny if requested
        self.shiny_config = shiny_config
        self.shiny_locations = None
        self.shiny_policies = None

        if shiny_config is not None:
            builder = ShinyEnvironmentBuilder(self.env, shiny_config, self.policy_gen)
            self.shiny_locations = builder.place_shiny_objects()
            self.env = builder.mark_environment(self.shiny_locations)
            self.shiny_policies = builder.generate_shiny_policies(self.shiny_locations)

        # Walk generator
        if self.env_config is not None and self.repeat_bias == 2.0:
            self.walk_gen = WalkGenerator(self.env, env_config=self.env_config)
        else:
            self.walk_gen = WalkGenerator(self.env, repeat_bias=self.repeat_bias)

    def setup(self, stage: Optional[str] = None):
        """Prepare data for training/validation/testing.

        Args:
            stage: 'fit', 'validate', 'test', or 'predict'
        """
        # No setup needed for on-the-fly generation
        pass

    def train_dataloader(self) -> DataLoader:
        """Infinite walk generation for training.

        Returns:
            DataLoader: Infinite walk dataloader
        """
        dataset = InfiniteWalkDataset(self)
        return DataLoader(dataset, batch_size=self.batch_size, collate_fn=self._collate_walks, num_workers=self.num_workers)

    def val_dataloader(self) -> DataLoader:
        """Fixed validation set for consistent metrics.

        Generates a fixed set of walks for validation to ensure
        consistent metric tracking across epochs.

        Returns:
            DataLoader: Validation dataloader
        """
        # Generate fixed validation walks
        n_val_walks = self.batch_size * 10  # 10 batches for validation

        if self.shiny_config is None:
            val_walks = self.walk_gen.generate_walks(n_val_walks, self.walk_length, None)
        else:
            val_walks = self.walk_gen.generate_shiny_walks(n_val_walks, self.walk_length, self.shiny_locations, self.shiny_policies, self.shiny_config.returns)

        return DataLoader(val_walks, batch_size=self.batch_size, collate_fn=self._collate_walks, num_workers=0)

    def _collate_walks(self, walks: List[Walk]) -> Tuple[Tensor, Tensor, Tensor]:
        """Collate walks into batched tensors.

        Args:
            walks: List of Walk objects

        Returns:
            observations: [batch, walk_length, n_observations]
            actions: [batch, walk_length]
            locations: [batch, walk_length]
        """
        return self.walk_gen.batch_walks(walks)

    def _get_current_policy(self, epoch: int) -> Optional[List[Location]]:
        """Get policy for current epoch based on curriculum schedule."""
        if self.curriculum_schedule is None:
            return None

        schedule = self.curriculum_schedule(epoch)
        policy_specs = schedule.get("policies", [])
        weights = schedule.get("weights", [])

        if not policy_specs:
            return None

        # Convert policy names to actual policy objects
        policies = []
        for spec in policy_specs:
            if isinstance(spec, str):
                # Generate policy from name
                if spec == "random":
                    policies.append(self.policy_gen.random_policy())
                elif spec == "distance":
                    policies.append(self.policy_gen.distance_policy(goal_locations=self.goal_locations, beta=self.beta))
                elif spec == "q_learning":
                    policies.append(self.policy_gen.q_learning_policy(goal_locations=self.goal_locations, gamma=self.gamma, beta=self.beta, n_iterations=100))
                else:
                    raise ValueError(f"Unknown policy type: {spec}")
            else:
                # Already a policy object
                policies.append(spec)

        if len(policies) == 1:
            return policies[0]

        return self.policy_gen.mix_policies(policies, weights)

    def generate_batch(self, epoch: int = 0) -> Tuple[Tensor, Tensor, Tensor]:
        """Generate single batch of walks for current epoch.

        Useful for manual training loops outside Lightning.

        Args:
            epoch: Current epoch number for curriculum scheduling

        Returns:
            observations: [batch, walk_length, n_observations]
            actions: [batch, walk_length]
            locations: [batch, walk_length]
        """
        policy = self._get_current_policy(epoch)

        if self.shiny_config is None:
            walks = self.walk_gen.generate_walks(self.batch_size, self.walk_length, policy)
        else:
            walks = self.walk_gen.generate_shiny_walks(self.batch_size, self.walk_length, self.shiny_locations, self.shiny_policies, self.shiny_config.returns)

        return self.walk_gen.batch_walks(walks)
