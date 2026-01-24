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


class TraceExtractor(Protocol[EventT, TraceT]):
    """Protocol for extracting plot-ready traces from event streams.

    A trace extractor is a stateful builder that observes events one-by-one
    and produces a complete trace when finalized.

    Type Parameters:
        EventT: Type of events to observe (e.g., RolloutEvent).
        TraceT: Type of trace to produce (must implement PlotTrace).
    """

    def observe(self, event: EventT) -> None:
        """Process one event and accumulate state.

        Args:
            event: Event to observe and incorporate into the trace.
        """
        ...

    def finalize(self) -> TraceT:
        """Finalize extraction and return the complete trace.

        Returns:
            Complete trace object ready for plotting.

        Raises:
            ValueError: If no events were observed (unless extractor supports empty traces).
        """
        ...
