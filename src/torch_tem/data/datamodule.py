"""PyTorch Lightning DataModule for TEM training data generation.

This module provides DataModule and TEMDataset for streaming on-the-fly
batch generation during TEM training.

Architecture Note
-----------------
Settings composition:
    - DataConfig composes low-level '*Settings' from settings.py
    - Prevents duplication of parameters like walk curriculum bounds
    - Instantiated in run.py from individual settings components
"""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, IterableDataset

from torch_tem import data, settings
from torch_tem.data.env_validation import validate_envs_against_contract


class DataConfig(BaseModel):
    """Composite configuration for TEM data generation (Lightning datamodule + dataset).

    This Config class composes low-level '*Settings' from settings.py to provide
    complete configuration for DataModule and TEMDataset. It aggregates settings
    for environment generation, rollout chunking, evaluation protocols, exploration
    behavior, shiny environment sampling, and walk length curriculum.

    Architecture:
        - Composes settings.EnvironmentSettings, settings.RolloutSettings, etc.
        - Used by DataModule and TEMDataset
        - Instantiated from RunArguments in run.py (prevents parameter duplication)

    Note:
        Walk curriculum settings (walk) are shared with TrainerConfig as a single
        runtime curriculum: the trainer schedules the current walk length, and the
        data pipeline consumes it when generating walks.
        Space contract (space) is shared across the entire run to ensure consistency.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    space: settings.SpaceContractSettings = Field(
        default_factory=settings.SpaceContractSettings,
        description="Space contract: observation and action space dimensions.",
    )
    env: settings.EnvironmentSettings = Field(
        default_factory=settings.EnvironmentSettings,
        description="Environment generation settings.",
    )
    iterator: settings.RolloutStreamSettings = Field(
        default_factory=settings.RolloutStreamSettings,
        description="Iterator protocol settings (rollout chunking + eval protocol).",
    )
    policy: settings.EnvSamplingSettings = Field(
        default_factory=settings.EnvSamplingSettings,
        description="Data generation policies (exploration + shiny).",
    )
    walk: settings.CurriculumSettings = Field(
        default_factory=settings.CurriculumSettings,
        description="Walk length curriculum settings.",
    )


class DataModule(pl.LightningDataModule):
    """Lightning DataModule for TEM training."""

    def __init__(self, data_settings: DataConfig):
        super().__init__()
        self.data_settings = data_settings
        self.dataset: Optional[TEMDataset] = None
        self.val_dataset: Optional[TEMDataset] = None
        self.test_dataset: Optional[TEMDataset] = None
        self._validated = False  # Track whether env validation has been performed

    def setup(self, stage: str = None):
        """Setup is called on every process in DDP.

        Create the dataset here (not in train_dataloader) to ensure it's only
        created once per process, even if train_dataloader is called multiple times.
        """
        # Defense-in-depth: validate env files against contract (idempotent)
        # This ensures alternate entrypoints that bypass run.py validation still fail fast
        if not self._validated:
            validate_envs_against_contract(self.data_settings.env.envs, self.data_settings.space)
            self._validated = True

        if stage in (None, "fit") and self.dataset is None:
            self.dataset = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
            )

        # Create validation dataset (finite, deterministic)
        if stage in (None, "fit", "validate") and self.data_settings.iterator.eval.enable_validation and self.val_dataset is None:
            self.val_dataset = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
                max_batches=self.data_settings.iterator.eval.val_batches,
                seed=self.data_settings.iterator.eval.val_seed,
            )

        # Create test dataset (finite, deterministic)
        if stage in (None, "test") and self.data_settings.iterator.eval.enable_test and self.test_dataset is None:
            self.test_dataset = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
                max_batches=self.data_settings.iterator.eval.test_batches,
                seed=self.data_settings.iterator.eval.test_seed,
            )

    def train_dataloader(self):
        """Return training dataloader.

        Returns a DataLoader wrapping the iterable dataset.
        num_workers=0 because TEMDataset holds stateful environment objects
        that are not safe to pickle across worker processes.
        """
        # Ensure dataset is created (in case setup wasn't called)
        if self.dataset is None:
            self.setup("fit")

        # Return DataLoader with batch_size=None (dataset yields pre-batched data)
        return DataLoader(self.dataset, batch_size=None, num_workers=0)

    def val_dataloader(self):
        """Return validation dataloader (finite, deterministic).

        Returns empty list if validation is disabled (Lightning requires iterable, not None).
        """
        if not self.data_settings.iterator.eval.enable_validation:
            return []

        return DataLoader(self.val_dataset, batch_size=None, num_workers=0)

    def test_dataloader(self):
        """Return test dataloader (finite, deterministic).

        Returns empty list if test is disabled (Lightning requires iterable, not None).
        """
        if not self.data_settings.iterator.eval.enable_test:
            return []

        return DataLoader(self.test_dataset, batch_size=None, num_workers=0)

    def set_walk_length_center(self, value: float):
        """Control surface: set walk length center (called by trainer during training)."""
        if self.dataset is not None:
            self.dataset.walk_length_center = value

    def sample_batch(self, split: str):
        """Utility to sample a single batch from the specified split.

        Args:
            split: One of "train", "validate", or "test".

        Returns:
            A single batch from the specified split.
        """
        if split == "train":
            return next(iter(self.dataset))
        elif split == "validate":
            return next(iter(self.val_dataset))
        elif split == "test":
            return next(iter(self.test_dataset))
        raise ValueError(f"Invalid split '{split}'; must be one of 'train', 'validate', or 'test'.")


class TEMDataset(IterableDataset):
    """Iterable dataset that generates TEM batches on-the-fly.

    Args:
        data_settings: Data generation settings.
        walk_it_min: Minimum walk length multiplier.
        walk_it_max: Maximum walk length multiplier.
        walk_it_window: Window for walk length sampling.
        max_batches: If set, dataset yields exactly this many batches then stops (for val/test).
        seed: If set, use this seed for deterministic generation (for val/test).
    """

    def __init__(
        self,
        data_settings: DataConfig,
        walk_it_min: int,
        walk_it_max: int,
        walk_it_window: float,
        max_batches: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.data_settings = data_settings
        self.env_paths = [str(p) for p in data_settings.env.envs]

        # Walk curriculum bounds (data-owned, training-controlled via set_walk_length_center)
        self.walk_it_min = walk_it_min
        self.walk_it_max = walk_it_max
        self.walk_it_window = walk_it_window
        self._walk_length_center = walk_it_max  # Default to max

        # Finite iteration control (for val/test)
        self.max_batches = max_batches
        self.seed = seed

        # Local RNG for deterministic val/test (avoids polluting global random state)
        # Always use Generator for consistent API (integers, random, choice)
        if self.seed is not None:
            self.np_rng = np.random.default_rng(self.seed)  # Seeded for val/test
        else:
            self.np_rng = np.random.default_rng()  # Unseeded for training

        # Torch generator only needed if seed is set
        self.torch_generator = torch.Generator().manual_seed(self.seed) if self.seed is not None else None

        # Initialize environments and walks
        self.environments, self.walks, self.visited = self._setup_environments()

    def _setup_environments(self):
        """Initialize training environments, walks, and visit tracking."""
        # Use local RNG for deterministic val/test, global for training
        rng = self.np_rng

        environments = [
            data.World(
                graph,
                randomise_observations=self.data_settings.env.randomise_observations,
                shiny=self._maybe_build_shiny_config(rng),
            )
            for graph in rng.choice(self.env_paths, self.data_settings.iterator.rollout.batch_size)
        ]

        visited = [[False for _ in range(env.n_locations)] for env in environments]

        walks = [
            env.generate_walks(
                self.data_settings.iterator.rollout.n_rollout * rng.integers(self.walk_it_min, self.walk_it_max),
                1,
            )[0]
            for env in environments
        ]

        return environments, walks, visited

    @property
    def walk_length_center(self) -> float:
        """Current center of walk length sampling window (updated by trainer)."""
        return self._walk_length_center

    @walk_length_center.setter
    def walk_length_center(self, value: float):
        """Update walk length center (called by LightningModule during training)."""
        self._walk_length_center = value

    def __iter__(self):
        """Generate batches (infinite for training, finite for val/test)."""
        if self.max_batches is None:
            # Infinite iteration (training)
            while True:
                yield self._generate_batch()
        else:
            # Finite iteration (validation/test)
            for _ in range(self.max_batches):
                yield self._generate_batch()

    def _generate_batch(self):
        """Generate a single batch (chunk) of data."""
        walk_length_center = int(self._walk_length_center)
        low = max(1, int(walk_length_center - self.walk_it_window * 0.5))
        high = max(low + 1, int(walk_length_center + self.walk_it_window * 0.5))

        # Use local RNG for deterministic val/test, global for training
        rng = self.np_rng

        # Build batch chunk
        chunk: list[list[list[Any]]] = []
        for env_i, walk in enumerate(self.walks):
            if len(walk) < self.data_settings.iterator.rollout.n_rollout:
                # Generate new environment and walk
                self.environments[env_i] = data.World(
                    self.env_paths[rng.integers(len(self.env_paths))],
                    randomise_observations=self.data_settings.env.randomise_observations,
                    shiny=self._maybe_build_shiny_config(rng),
                )
                self.visited[env_i] = [False for _ in range(self.environments[env_i].n_locations)]
                walk = self.environments[env_i].generate_walks(
                    self.data_settings.iterator.rollout.n_rollout * rng.integers(low, high),
                    1,
                )[0]
                self.walks[env_i] = walk

            for step in range(self.data_settings.iterator.rollout.n_rollout):
                if len(chunk) < self.data_settings.iterator.rollout.n_rollout:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)

        # Stack observations
        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0)

        return chunk, self.visited

    def _maybe_build_shiny_config(self, rng: np.random.Generator) -> Optional[dict[str, Any]]:
        """Build the shiny config dict consumed by `data.World`.

        Parameters:
            rng: Local RNG (seeded for val/test, unseeded for training).

        Returns:
            Dict with keys expected by `data.World(shiny=...)`, or None if shiny
            sampling does not trigger for this batch/environment.
        """
        shiny_settings = self.data_settings.policy.shiny
        if rng.random() >= shiny_settings.shiny_rate:
            return None

        return {
            "gamma": shiny_settings.shiny_gamma,
            "beta": shiny_settings.shiny_beta,
            "n": shiny_settings.shiny_n,
            "returns": shiny_settings.shiny_returns,
        }
