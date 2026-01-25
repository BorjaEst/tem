"""Utilities for converting TEM walk batch formats.

TEM datasets produce batches in a *time-major* format:

  chunk[t] = [locations_t, observations_t, actions_t]

Where:
  - locations_t is a list[dict] of length B
  - observations_t is a Tensor of shape (B, ...)
  - actions_t is a list[int] of length B

Most figure traces operate in an *env-major* format:

  walks[env][t] = [location_dict, observation_tensor, action_int]
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable


def time_major_to_env_major(chunk: Sequence[Sequence[Any]], *, validate: bool = True) -> list[list[list[Any]]]:
    """Convert a time-major chunk to env-major sequences.

    A *time-major* chunk is a sequence of timesteps. Each timestep is a sequence
    of components. Each component must be batch-indexable along its first
    dimension, and all components must share the same batch size.

    Args:
        chunk: Sequence of timesteps. Each timestep is a sequence of components.
        validate: If True, validate consistent component counts and batch sizes.

    Returns:
        Env-major structure: list length B; each element is a list length T; each
        step is a list of the timestep components indexed at that env.
    """
    if len(chunk) == 0:
        return []

    first_step = chunk[0]
    if len(first_step) == 0:
        raise ValueError("Expected each timestep to contain at least one component")

    n_components = len(first_step)
    batch_size = _batch_size_of(first_step[0])

    if validate:
        for comp_i, comp in enumerate(first_step):
            comp_b = _batch_size_of(comp)
            if comp_b != batch_size:
                raise ValueError(f"Inconsistent batch size in first step: comp[0]={batch_size} but comp[{comp_i}]={comp_b}")

    env_major: list[list[list[Any]]] = [[] for _ in range(batch_size)]

    for t, step in enumerate(chunk):
        if validate and len(step) != n_components:
            raise ValueError(f"Inconsistent component count at t={t}: expected {n_components}, got {len(step)}")

        if validate:
            for comp_i, comp in enumerate(step):
                comp_b = _batch_size_of(comp)
                if comp_b != batch_size:
                    raise ValueError(f"Inconsistent batch size at t={t}: expected {batch_size}, got {comp_b} for component {comp_i}")

        for env_i in range(batch_size):
            env_major[env_i].append([comp[env_i] for comp in step])

    return env_major


def time_major_to_env_major_walks(
    chunk: Sequence[Sequence[Any]],
    *,
    validate: bool = True,
) -> list[list[list[Any]]]:
    """Convert a time-major TEM batch chunk to env-major walks.

    This preserves the legacy function name used by figures.
    """
    return time_major_to_env_major(chunk, validate=validate)


def time_major_location_ids(
    chunk: Sequence[Sequence[Any]],
    *,
    location_component_idx: int = 0,
    key: str = "id",
    getter: Callable[[Any], int] | None = None,
    validate: bool = True,
) -> list[list[int]]:
    """Extract per-env per-step location IDs from a time-major chunk.

    Defaults to reading the first component of each timestep (locations).

    - If `getter` is provided: uses `getter(location) -> int`.
    - Else if `location` is a mapping and contains `key` (default: "id"): uses that value.
    - Else: attempts to coerce the location itself into an `int` (supports scalar tensors).
    """
    env_major = time_major_to_env_major(chunk, validate=validate)

    def extract(loc: Any) -> int:
        if getter is not None:
            return int(getter(loc))
        if isinstance(loc, Mapping) and key in loc:
            return _coerce_int(loc[key])
        return _coerce_int(loc)

    return [[extract(step[location_component_idx]) for step in env_walk] for env_walk in env_major]


def downsample_env_major(sequences: list[list[Any]], stride: int) -> list[list[Any]]:
    """Downsample env-major sequences with a stride (1 keeps all)."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1 (got {stride})")
    if stride == 1:
        return sequences
    return [[seq[i] for i in range(0, len(seq), stride)] for seq in sequences]


__all__ = ["time_major_to_env_major", "time_major_to_env_major_walks", "time_major_location_ids", "downsample_env_major"]


def _batch_size_of(component: Any) -> int:
    """Infer batch size from the first dimension of a batch-indexable component."""
    if component is None:
        raise TypeError("Cannot infer batch size from None component")

    shape = getattr(component, "shape", None)
    if shape is not None:
        try:
            return int(shape[0])
        except Exception:
            pass

    try:
        return len(component)  # type: ignore[arg-type]
    except TypeError as e:
        raise TypeError(f"Component of type {type(component).__name__} is not batch-sized") from e


def _coerce_int(x: Any) -> int:
    """Coerce scalar-like values to `int` (supports 0-d tensors via `.item()`)."""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    item = getattr(x, "item", None)
    if callable(item):
        try:
            return int(item())
        except Exception:
            pass
    return int(x)
