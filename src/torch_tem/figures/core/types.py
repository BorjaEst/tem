"""Core type definitions for tracing and figure generation.

Defines Protocol-based interfaces for trace extraction and plotting.
These protocols enable capability-based polymorphism without rigid inheritance.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, Protocol, TypeVar


class PlotTrace(Protocol):
    """Protocol for plot-ready trace data.

    Defines the minimal capabilities required for trace objects to be
    consumed by figure modules. Concrete traces implement this protocol
    by providing the required attributes and methods.

    Attributes:
        batch_size: Number of parallel environments/samples in the trace.
        n_steps: Number of timesteps recorded in the trace.
        meta: Optional metadata dictionary (e.g., run info, split name, step count).
    """

    @property
    def batch_size(self) -> int:
        """Number of parallel environments in the trace."""
        ...

    @property
    def n_steps(self) -> int:
        """Number of timesteps in the trace."""
        ...

    @property
    def meta(self) -> Mapping[str, Any]:
        """Metadata dictionary for the trace (e.g., run info, split)."""
        ...

    def select_env(self, env_idx: int) -> PlotTrace:
        """Extract trace for a single environment.

        Args:
            env_idx: Index of environment to extract (0 <= env_idx < batch_size).

        Returns:
            New trace with batch_size=1 containing only env_idx.
        """
        ...

    def downsample_time(self, stride: int) -> PlotTrace:
        """Downsample trace along time dimension.

        Args:
            stride: Keep every stride-th timestep.

        Returns:
            New trace with n_steps = ceil(original_n_steps / stride).
        """
        ...


EventT = TypeVar("EventT")
TraceT = TypeVar("TraceT", bound=PlotTrace)
