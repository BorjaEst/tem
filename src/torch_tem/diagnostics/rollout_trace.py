"""Plot-ready dataclasses for TEM rollout traces.

Defines lightweight, CPU-based data containers that represent model outputs
and environment states over time. These dataclasses isolate figure rendering
from simulation logic and GPU memory.

Key Principle:
    Figures consume these dataclasses, NOT Model or live RolloutStream objects.
    All tensors are detached and moved to CPU before storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch
from numpy.typing import NDArray


@dataclass
class TEMRolloutTrace:
    """Time-series trace of TEM outputs over a rollout segment.

    This dataclass contains plot-ready arrays for visualization. All data
    is CPU-based (NumPy arrays or Python lists) to prevent GPU memory pressure.

    Attributes:
        # Temporal dimension
        n_steps: Number of timesteps in this trace.

        # Sensory reconstruction (timestep, batch, obs_dim)
        o_predicted: Predicted observations (probabilities) from p_inf.
        o_true: Ground truth observations.

        # Location predictions (timestep, batch, location_dim)
        p_inf: Inferred grounded location beliefs.
        p_gen: Generated grounded location beliefs.
        g_inf: Inferred abstract location codes.
        g_gen: Generated abstract location codes.

        # Optional signals
        p_xi: Sensory-cued grounded beliefs (None if use_x_cued_recall=False).

        # Labels (ground truth)
        location_ids: True location IDs at each timestep (timestep, batch).
        actions: Actions taken at each timestep (timestep, batch).

        # Metadata
        batch_size: Number of parallel environments.
        env_names: Optional names/identifiers for each environment in batch.
    """

    n_steps: int
    batch_size: int

    # Sensory reconstructions
    o_predicted: NDArray[np.float32]  # (T, B, n_obs)
    o_true: NDArray[np.float32]  # (T, B, n_obs)

    # Location representations (vary by model architecture)
    p_inf: List[NDArray[np.float32]]  # [(T, B, n_place[f]) for f in frequencies]
    p_gen: List[NDArray[np.float32]]  # [(T, B, n_place[f]) for f in frequencies]
    g_inf: List[NDArray[np.float32]]  # [(T, B, n_grid[f]) for f in frequencies]
    g_gen: List[NDArray[np.float32]]  # [(T, B, n_grid[f]) for f in frequencies]

    # Ground truth labels
    location_ids: NDArray[np.int32]  # (T, B)
    actions: NDArray[np.int32]  # (T, B)

    # Optional signals (must come after non-default fields)
    p_xi: Optional[List[NDArray[np.float32]]] = None  # [(T, B, n_place[f]) for f]

    # Metadata
    env_names: Optional[List[str]] = None

    @property
    def n_frequencies(self) -> int:
        """Number of frequency modules (inferred from p_inf length)."""
        return len(self.p_inf)

    def select_env(self, env_idx: int) -> TEMRolloutTrace:
        """Extract trace for a single environment from the batch.

        Args:
            env_idx: Index of the environment to extract (0 <= env_idx < batch_size).

        Returns:
            New TEMRolloutTrace with batch_size=1 containing only env_idx.
        """
        if env_idx >= self.batch_size:
            raise IndexError(f"env_idx {env_idx} out of range for batch_size {self.batch_size}")

        return TEMRolloutTrace(
            n_steps=self.n_steps,
            batch_size=1,
            o_predicted=self.o_predicted[:, env_idx : env_idx + 1, :],
            o_true=self.o_true[:, env_idx : env_idx + 1, :],
            p_inf=[p[:, env_idx : env_idx + 1, :] for p in self.p_inf],
            p_gen=[p[:, env_idx : env_idx + 1, :] for p in self.p_gen],
            g_inf=[g[:, env_idx : env_idx + 1, :] for g in self.g_inf],
            g_gen=[g[:, env_idx : env_idx + 1, :] for g in self.g_gen],
            p_xi=[p[:, env_idx : env_idx + 1, :] for p in self.p_xi] if self.p_xi is not None else None,
            location_ids=self.location_ids[:, env_idx : env_idx + 1],
            actions=self.actions[:, env_idx : env_idx + 1],
            env_names=[self.env_names[env_idx]] if self.env_names is not None else None,
        )

    def downsample_time(self, stride: int) -> TEMRolloutTrace:
        """Downsample the trace along the time dimension.

        Args:
            stride: Keep every stride-th timestep.

        Returns:
            New TEMRolloutTrace with n_steps = ceil(original_n_steps / stride).
        """
        return TEMRolloutTrace(
            n_steps=int(np.ceil(self.n_steps / stride)),
            batch_size=self.batch_size,
            o_predicted=self.o_predicted[::stride, :, :],
            o_true=self.o_true[::stride, :, :],
            p_inf=[p[::stride, :, :] for p in self.p_inf],
            p_gen=[p[::stride, :, :] for p in self.p_gen],
            g_inf=[g[::stride, :, :] for g in self.g_inf],
            g_gen=[g[::stride, :, :] for g in self.g_gen],
            p_xi=[p[::stride, :, :] for p in self.p_xi] if self.p_xi is not None else None,
            location_ids=self.location_ids[::stride, :],
            actions=self.actions[::stride, :],
            env_names=self.env_names,
        )


@dataclass
class EnvMapData:
    """Environment geometry for map rendering.

    Minimal environment representation for plotting. Contains only the
    spatial layout and per-location scalars needed for visualization.

    Attributes:
        n_locations: Number of locations in the environment.
        positions: 2D positions of locations [(x, y), ...] normalized to [0, 1].
        values: Optional scalar values to color locations (e.g., visit counts, place cell rates).
        shiny_locations: Indices of shiny locations (empty if no shiny objects).
    """

    n_locations: int
    positions: NDArray[np.float32]  # (n_locations, 2) - (x, y) coordinates
    values: Optional[NDArray[np.float32]] = None  # (n_locations,) - scalar per location
    shiny_locations: Sequence[int] = ()

    @classmethod
    def from_world(cls, world, values: Optional[NDArray[np.float32]] = None) -> EnvMapData:
        """Create EnvMapData from a torch_tem.data.World object.

        Args:
            world: World instance with .locations attribute.
            values: Optional per-location scalar values for coloring.

        Returns:
            EnvMapData instance ready for plotting.
        """
        positions = np.array([[loc["o"], loc["y"]] for loc in world.locations], dtype=np.float32)
        shiny = [i for i, loc in enumerate(world.locations) if loc.get("shiny", False)]

        return cls(
            n_locations=len(world.locations),
            positions=positions,
            values=values,
            shiny_locations=shiny,
        )
