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

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypeAlias, Union

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, IterableDataset

from torch_tem import settings
from torch_tem.data.env_validation import validate_envs_against_contract
from torch_tem.data.world import World, WorldStep

SplitName: TypeAlias = Literal["train", "validate", "test"]


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
        self.datasets: Dict[SplitName, TEMDataset] = {}
        self._validated = False  # Track whether env validation has been performed

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup is called on every process in DDP.

        Create the dataset here (not in train_dataloader) to ensure it's only
        created once per process, even if train_dataloader is called multiple times.
        """
        # Defense-in-depth: validate env files against contract (idempotent)
        # This ensures alternate entrypoints that bypass run.py validation still fail fast
        if not self._validated:
            validate_envs_against_contract(self.data_settings.env.envs, self.data_settings.space)
            self._validated = True

        if stage in (None, "fit") and self.get_dataset("train") is None:
            self.datasets["train"] = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
            )

        # Create validation dataset (finite, deterministic)
        if stage in (None, "fit", "validate") and self.data_settings.iterator.eval.enable_validation and self.get_dataset("validate") is None:
            self.datasets["validate"] = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
                max_batches=self.data_settings.iterator.eval.val_steps,
                seed=self.data_settings.iterator.eval.val_seed,
            )

        # Create test dataset (finite, deterministic)
        if stage in (None, "test") and self.data_settings.iterator.eval.enable_test and self.get_dataset("test") is None:
            self.datasets["test"] = TEMDataset(
                self.data_settings,
                walk_it_min=self.data_settings.walk.walk_it_min,
                walk_it_max=self.data_settings.walk.walk_it_max,
                walk_it_window=self.data_settings.walk.walk_it_window,
                max_batches=self.data_settings.iterator.eval.test_steps,
                seed=self.data_settings.iterator.eval.test_seed,
            )

    @property
    def dataset(self) -> Optional["TEMDataset"]:
        """Compatibility accessor for the training dataset."""
        return self.get_dataset("train")

    @property
    def val_dataset(self) -> Optional["TEMDataset"]:
        """Compatibility accessor for the validation dataset."""
        return self.get_dataset("validate")

    @property
    def test_dataset(self) -> Optional["TEMDataset"]:
        """Compatibility accessor for the test dataset."""
        return self.get_dataset("test")

    def get_dataset(self, split: SplitName) -> Optional["TEMDataset"]:
        """Return the dataset for the requested split, if available.

        Args:
            split: One of "train", "validate", or "test".

        Returns:
            The dataset for the split, or None if it has not been created.
        """
        return self.datasets.get(split)

    def _ensure_dataset(self, split: SplitName) -> "TEMDataset":
        """Ensure the dataset for the requested split is initialized.

        Args:
            split: One of "train", "validate", or "test".

        Returns:
            The initialized dataset.

        Raises:
            ValueError: If the split is invalid or disabled.
        """
        if split == "train":
            if self.get_dataset("train") is None:
                self.setup("fit")
        elif split == "validate":
            if self.get_dataset("validate") is None:
                self.setup("validate")
        elif split == "test":
            if self.get_dataset("test") is None:
                self.setup("test")
        else:
            raise ValueError(f"Invalid split '{split}'; must be one of 'train', 'validate', or 'test'.")

        dataset = self.get_dataset(split)
        if dataset is None:
            raise ValueError(f"Dataset for split '{split}' is not available. " "Check that the split is enabled in settings.")
        return dataset

    def dataloader(self, split: SplitName) -> Union[DataLoader, list[DataLoader]]:
        """Return dataloader for the specified split.

        Args:
            split: One of "train", "validate", or "test".

        Returns:
            DataLoader wrapping the appropriate dataset.
        """
        if split == "train":
            return self.train_dataloader()
        if split == "validate":
            return self.val_dataloader()
        if split == "test":
            return self.test_dataloader()
        raise ValueError(f"Invalid split '{split}'; must be one of 'train', 'validate', or 'test'.")

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader.

        Returns a DataLoader wrapping the iterable dataset.
        num_workers=0 because TEMDataset holds stateful environment objects
        that are not safe to pickle across worker processes.
        """
        # Ensure dataset is created (in case setup wasn't called)
        dataset = self._ensure_dataset("train")

        # Return DataLoader with batch_size=None (dataset yields pre-batched data)
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def val_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        """Return validation dataloader (finite, deterministic).

        Returns empty list if validation is disabled (Lightning requires iterable, not None).
        """
        if not self.data_settings.iterator.eval.enable_validation:
            return []

        dataset = self._ensure_dataset("validate")
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def test_dataloader(self) -> Union[DataLoader, list[DataLoader]]:
        """Return test dataloader (finite, deterministic).

        Returns empty list if test is disabled (Lightning requires iterable, not None).
        """
        if not self.data_settings.iterator.eval.enable_test:
            return []

        dataset = self._ensure_dataset("test")
        return DataLoader(dataset, batch_size=None, num_workers=0)

    def set_walk_length_center(self, value: float) -> None:
        """Control surface: set walk length center (called by trainer).

        Args:
            value: New walk length center for all initialized datasets.
        """
        for dataset in self.datasets.values():
            dataset.walk_length_center = value

    def sample_batch(self, split: SplitName) -> Any:
        """Utility to sample a single batch from the specified split.

        Args:
            split: One of "train", "validate", or "test".

        Returns:
            A single batch from the specified split.
        """
        dataset = self._ensure_dataset(split)
        return next(iter(dataset))

    def sample_episode(self, split: SplitName, n_batches: int, reset_before: bool = False) -> tuple[list[Any], list[list[bool]], list[World]]:
        """Sample consecutive batches and stitch them into a continuous episode.

        Args:
            split: One of "train", "validate", or "test".
            n_batches: Number of consecutive batches to stitch.
            reset_before: Whether to reset deterministic splits before sampling.

        Returns:
            Tuple of (episode_walk, visited_snapshot, environments_snapshot).

        Raises:
            ValueError: If n_batches is invalid or the split is exhausted.
        """
        if n_batches <= 0:
            raise ValueError("n_batches must be positive")

        if reset_before:
            self.reset_split(split)

        dataset = self._ensure_dataset(split)
        iterator = iter(dataset)

        episode_walk: list[Any] = []
        visited_snapshot = [list(mask) for mask in dataset.visited]
        environments_snapshot = list(dataset.environments)

        try:
            for _ in range(n_batches):
                chunk, _ = next(iterator)
                episode_walk.extend(chunk)
        except StopIteration as exc:
            raise ValueError(f"Split '{split}' exhausted before collecting {n_batches} batches. " "Reduce n_batches or increase eval steps.") from exc

        return episode_walk, visited_snapshot, environments_snapshot

    def reset_split(self, split: SplitName) -> None:
        """Reset a deterministic dataset split to its initial state.

        Args:
            split: One of "validate" or "test". "train" is ignored.

        Raises:
            ValueError: If the split name is invalid.
        """
        if split == "train":
            return
        if split not in ("validate", "test"):
            raise ValueError(f"Invalid split '{split}'; must be one of 'train', 'validate', or 'test'.")
        dataset = self.get_dataset(split)
        if dataset is None:
            return
        dataset.reset()


@dataclass
class DataStep:
    environments: List[World]
    world_step: WorldStep
    visited: Optional[List[List[bool]]]


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

    def _is_seeded_eval(self) -> bool:
        """Return True when dataset is a finite, seeded evaluation stream."""
        return self.seed is not None and self.max_batches is not None

    def _eval_min_walk_length(self) -> int:
        """Minimum walk length required for a seeded evaluation stream."""
        if not self._is_seeded_eval():
            return 0
        return self.max_batches * self.data_settings.iterator.rollout.n_rollout

    def reset(self) -> None:
        """Reset deterministic datasets to their initial seeded state.

        For training datasets (unseeded), this is a no-op to avoid disrupting
        training dynamics.
        """
        if self.seed is None:
            return

        self.np_rng = np.random.default_rng(self.seed)
        self.torch_generator = torch.Generator().manual_seed(self.seed)
        self.environments, self.walks, self.visited = self._setup_environments()

    def _setup_environments(self):
        """Initialize training environments, walks, and visit tracking."""
        # Use local RNG for deterministic val/test, global for training
        rng = self.np_rng

        environments = [
            World(graph, randomise_observations=self.data_settings.env.randomise_observations, shiny=self._maybe_build_shiny_config(rng), rng=rng)
            for graph in rng.choice(self.env_paths, self.data_settings.iterator.rollout.batch_size)
        ]
        walks = [env.generate_walks(self._initial_walk_length(rng), 1)[0] for env in environments]
        visited = [[False for _ in range(env.n_locations)] for env in environments]

        return environments, walks, visited

    def _initial_walk_length(self, rng: np.random.Generator) -> int:
        """Compute initial walk length for a new environment slot."""
        base_length = self.data_settings.iterator.rollout.n_rollout * rng.integers(self.walk_it_min, self.walk_it_max)
        min_length = self._eval_min_walk_length()
        return max(base_length, min_length)

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
                if self._is_seeded_eval():
                    self._raise_eval_depletion_error(env_i, len(walk))
                # Generate new environment and walk (training only)
                self.environments[env_i] = World(
                    self.env_paths[rng.integers(len(self.env_paths))],
                    randomise_observations=self.data_settings.env.randomise_observations,
                    shiny=self._maybe_build_shiny_config(rng),
                    rng=rng,
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

    def _raise_eval_depletion_error(self, env_i: int, remaining: int) -> None:
        """Raise a configuration error when a seeded eval walk is depleted."""
        n_rollout = self.data_settings.iterator.rollout.n_rollout
        min_length = self._eval_min_walk_length()
        raise ValueError(
            "Seeded evaluation stream depleted a walk before completing the finite stream. "
            f"env_index={env_i}, remaining_steps={remaining}, n_rollout={n_rollout}, "
            f"required_min_walk_length={min_length}. "
            "Increase walk length (walk_it_min/walk_it_max) or reduce val_steps/test_steps or n_rollout."
        )

    def _maybe_build_shiny_config(self, rng: np.random.Generator) -> Optional[dict[str, Any]]:
        """Build the shiny config dict consumed by `World`.

        Parameters:
            rng: Local RNG (seeded for val/test, unseeded for training).

        Returns:
            Dict with keys expected by `World(shiny=...)`, or None if shiny
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
