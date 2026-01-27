"""TraceTree access helpers for figure modules."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from torch_tem.diagnostics.traces import TraceNode, TraceTree


def get_batch_size(trace: TraceTree) -> int:
    """Return the batch size for the trace."""
    return int(trace.batch_size or 0)


def get_length(trace: TraceTree) -> int:
    """Return the number of time steps in the trace."""
    return int(trace.length)


def get_meta(trace: TraceTree) -> dict[str, Any]:
    """Return root metadata dictionary, if available."""
    meta = trace.root.meta_static.get("meta")
    return dict(meta) if isinstance(meta, dict) else {}


def get_environments(trace: TraceTree) -> list[Any]:
    """Return environments stored in the trace metadata."""
    envs = trace.root.meta_static.get("environments")
    return list(envs) if envs is not None else []


def get_visited(trace: TraceTree) -> Any:
    """Return visited masks stored in the trace metadata."""
    return trace.root.meta_static.get("visited")


def get_world(trace: TraceTree, env_idx: int) -> Any:
    """Return the World object for a selected environment index."""
    envs = get_environments(trace)
    if not envs:
        raise ValueError("TraceTree has no environments")
    if not (0 <= env_idx < len(envs)):
        raise IndexError(f"env_idx {env_idx} out of range [0, {len(envs)})")
    return envs[env_idx]


def get_location_ids(trace: TraceTree) -> np.ndarray:
    """Return the location id array with shape (T, B)."""
    return get_dense(trace, "world_step/location_ids")


def get_action_ids(trace: TraceTree) -> np.ndarray:
    """Return the action id array with shape (T, B)."""
    return get_dense(trace, "world_step/action_ids")


def get_observations(trace: TraceTree) -> np.ndarray:
    """Return the observation array with shape (T, B, n_o)."""
    return get_dense(trace, "world_step/observation")


def get_location_ids_for_env(trace: TraceTree, env_idx: int) -> list[int]:
    """Return per-step location ids for a selected environment."""
    location_ids = get_location_ids(trace)
    if location_ids.size == 0:
        return []
    return [int(v) for v in location_ids[:, env_idx]]


def get_action_ids_for_env(trace: TraceTree, env_idx: int) -> list[int]:
    """Return per-step action ids for a selected environment."""
    action_ids = get_action_ids(trace)
    if action_ids.size == 0:
        return []
    return [int(v) for v in action_ids[:, env_idx]]


def get_multiscale(
    trace: TraceTree,
    base_path: str,
    freq_idx: int,
) -> np.ndarray:
    """Return a multi-scale dense array for a single frequency.

    Args:
        trace: TraceTree holding rollout data.
        base_path: Path to the multiscale node (e.g. "output/inference/g_inf").
        freq_idx: Frequency index to select.

    Returns:
        NumPy array of shape (T, B, C) for the selected frequency.
    """
    node = get_node(trace, base_path)
    child = node.children.get(str(freq_idx))
    if child is None:
        raise IndexError(f"freq_idx {freq_idx} not available at path '{base_path}'")
    if "value" not in child.data:
        raise ValueError(f"No dense value at '{base_path}/{freq_idx}'")
    return child.data["value"]


def get_n_freq(trace: TraceTree, base_path: str) -> int:
    """Return number of frequency modules at a multiscale path."""
    node = get_node(trace, base_path)
    return len(_sorted_child_indices(node.children.keys()))


def validate_env_idx(trace: TraceTree, env_idx: int) -> int:
    """Validate and return environment index."""
    batch_size = get_batch_size(trace)
    if not (0 <= env_idx < batch_size):
        raise IndexError(f"env_idx {env_idx} out of range [0, {batch_size})")
    return env_idx


def validate_freq_idx(trace: TraceTree, base_path: str, freq_idx: int) -> int:
    """Validate and return frequency index."""
    n_freq = get_n_freq(trace, base_path)
    if not (0 <= freq_idx < n_freq):
        raise IndexError(f"freq_idx {freq_idx} out of range [0, {n_freq})")
    return freq_idx


def get_dense(trace: TraceTree, path: str) -> np.ndarray:
    """Return dense data stored at a node path.

    Args:
        trace: TraceTree instance.
        path: Path with a data key (e.g. "world_step/location_ids").

    Returns:
        Dense NumPy array stored at the requested key.
    """
    node_path, key = _split_path(path)
    node = get_node(trace, node_path)
    if key not in node.data:
        raise ValueError(f"Dense key '{key}' missing at '{node_path}'")
    return node.data[key]


def get_node(trace: TraceTree, path: str) -> TraceNode:
    """Return the TraceNode for a slash-delimited path."""
    current = trace.root
    for part in _iter_path_parts(path):
        if part not in current.children:
            raise ValueError(f"Trace path '{path}' is missing '{part}'")
        current = current.children[part]
    return current


def _split_path(path: str) -> tuple[str, str]:
    """Split a path into node path and data key."""
    parts = list(_iter_path_parts(path))
    if len(parts) < 2:
        raise ValueError("Path must include a node and data key")
    node_path = "/".join(parts[:-1])
    return node_path, parts[-1]


def _iter_path_parts(path: str) -> Iterable[str]:
    """Yield path parts for non-empty segments."""
    for part in path.split("/"):
        if part:
            yield part


def _sorted_child_indices(keys: Iterable[str]) -> list[int]:
    """Return numeric child indices sorted ascending."""
    indices = []
    for key in keys:
        try:
            indices.append(int(key))
        except ValueError:
            continue
    return sorted(indices)
