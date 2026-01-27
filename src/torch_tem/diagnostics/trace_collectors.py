"""Trace collectors for TEM rollouts.

These helpers build TraceTree instances from model rollouts and provide
utilities for downsampling and concatenation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from torch_tem.diagnostics.traces import Event, MetaUpdate, TraceConfig, TraceNode, TraceTree
from torch_tem.model import Model, RolloutStep, RolloutStream, TEMOutput, TEMState


@dataclass(frozen=True)
class TraceWorldStep:
    """World-step fields captured for TraceTree ingestion.

    Attributes:
        observation: Batched observation tensor for the step.
        action_ids: Batched action ids with -1 for missing actions.
        location_ids: Batched location ids as integers.
    """

    observation: torch.Tensor
    action_ids: torch.Tensor
    location_ids: torch.Tensor


@dataclass(frozen=True)
class TraceStep:
    """TraceTree-friendly step container.

    Attributes:
        world_step: World-level inputs and metadata for the step.
        output: TEM model outputs for the step.
        state: TEM recurrent state for the step.
    """

    world_step: TraceWorldStep
    output: TEMOutput
    state: TEMState


def collect_rollout_trace_tree(
    batch: Any,
    environments: Sequence[Any],
    model: Model,
    stop: Optional[int],
    *,
    meta: Optional[dict[str, Any]] = None,
    config: Optional[TraceConfig] = None,
) -> TraceTree:
    """Collect a TraceTree from a rollout batch.

    Args:
        batch: Tuple of (walk, visited) produced by the dataloader.
        environments: Environments aligned to the rollout batch.
        model: TEM model used to generate outputs and state.
        stop: Optional maximum number of rollout steps to record.
        meta: Optional metadata dictionary to store at the trace root.
        config: Optional TraceConfig for strictness and behavior.

    Returns:
        TraceTree containing the collected rollout data.
    """
    if type(batch) not in {tuple, list}:
        raise ValueError(f"Expected batch=(walk, visited); got {type(batch).__name__}")
    if len(batch) != 2:
        raise ValueError(f"Expected batch of length 2; got {len(batch)}")

    walk, visited = batch
    trace = TraceTree(config=config or TraceConfig())

    for idx, step in enumerate(RolloutStream(model, walk)):
        if stop is not None and idx >= stop:
            break
        trace.append(_to_trace_step(step))

    trace.finalize()
    trace.root.meta_static["environments"] = list(environments)
    trace.root.meta_static["visited"] = visited
    if meta is not None:
        trace.root.meta_static["meta"] = dict(meta)
    return trace


def downsample_trace(trace: TraceTree, stride: int) -> TraceTree:
    """Downsample a trace along the time axis.

    Args:
        trace: TraceTree to downsample.
        stride: Sampling stride (1 keeps all steps).

    Returns:
        A new TraceTree containing downsampled data.
    """
    if stride <= 1:
        return trace

    downsampled = TraceTree(config=trace.config)
    downsampled.root = _downsample_node(trace.root, stride)
    downsampled.length = _downsample_length(trace.length, stride)
    downsampled.batch_size = trace.batch_size
    return downsampled


def concat_traces(traces: Sequence[TraceTree]) -> TraceTree:
    """Concatenate multiple traces along time.

    Args:
        traces: Sequence of TraceTree instances.

    Returns:
        Concatenated TraceTree with time-aligned metadata.
    """
    if not traces:
        raise ValueError("No traces provided for concatenation")

    base = traces[0]
    _validate_concat_compatibility(traces)

    offsets = _time_offsets(traces)
    merged = TraceTree(config=base.config)
    merged.root = _concat_nodes([t.root for t in traces], offsets)
    merged.length = sum(t.length for t in traces)
    merged.batch_size = base.batch_size
    return merged


def _to_trace_step(step: RolloutStep) -> TraceStep:
    """Convert a RolloutStep into a TraceStep.

    Args:
        step: RolloutStep from RolloutStream.

    Returns:
        TraceStep with numeric world-step fields.
    """
    action_ids = _coerce_action_ids(step.world_step.action)
    location_ids = _coerce_location_ids(step.world_step.locations)
    world_step = TraceWorldStep(
        observation=step.world_step.observation,
        action_ids=action_ids,
        location_ids=location_ids,
    )
    return TraceStep(world_step=world_step, output=step.output, state=step.state)


def _coerce_action_ids(actions: Iterable[Any]) -> torch.Tensor:
    """Convert action list into a numeric tensor.

    Args:
        actions: Iterable of action ids or None.

    Returns:
        Tensor of shape (B,) with -1 for missing actions.
    """
    action_ids = [(-1 if action is None else int(action)) for action in actions]
    return torch.tensor(action_ids, dtype=torch.long)


def _coerce_location_ids(locations: Iterable[Any]) -> torch.Tensor:
    """Convert location payloads into a numeric tensor.

    Args:
        locations: Iterable of location payloads (dict, tensor, int).

    Returns:
        Tensor of shape (B,) containing integer location ids.
    """
    loc_ids = [_coerce_location_id(location) for location in locations]
    return torch.tensor(loc_ids, dtype=torch.long)


def _coerce_location_id(location: Any) -> int:
    """Convert a location payload into an integer id."""
    if isinstance(location, dict) and "id" in location:
        return int(location["id"])
    if hasattr(location, "item"):
        return int(location.item())
    return int(location)


def _downsample_node(node: TraceNode, stride: int) -> TraceNode:
    """Downsample a TraceNode recursively."""
    down = TraceNode()
    down.data = {key: value[::stride] for key, value in node.data.items()}
    down.meta_static = dict(node.meta_static)
    down.meta_sparse = _downsample_meta_sparse(node.meta_sparse, stride)
    down.events = _downsample_events(node.events, stride)
    for name, child in node.children.items():
        down.children[name] = _downsample_node(child, stride)
    return down


def _downsample_meta_sparse(
    meta_sparse: dict[str, list[MetaUpdate]],
    stride: int,
) -> dict[str, list[MetaUpdate]]:
    """Downsample sparse metadata updates by stride."""
    downsampled: dict[str, list[MetaUpdate]] = {}
    for key, updates in meta_sparse.items():
        kept = [MetaUpdate(t=upd.t // stride, value=upd.value) for upd in updates if upd.t % stride == 0]
        if kept:
            downsampled[key] = kept
    return downsampled


def _downsample_events(
    events: dict[str, list[Event]],
    stride: int,
) -> dict[str, list[Event]]:
    """Downsample events by stride."""
    downsampled: dict[str, list[Event]] = {}
    for key, entries in events.items():
        kept = [Event(t=evt.t // stride, payload=evt.payload) for evt in entries if evt.t % stride == 0]
        if kept:
            downsampled[key] = kept
    return downsampled


def _downsample_length(length: int, stride: int) -> int:
    """Compute downsampled length for a stride."""
    if length <= 0:
        return 0
    return int(np.ceil(length / stride))


def _validate_concat_compatibility(traces: Sequence[TraceTree]) -> None:
    """Validate traces before concatenation."""
    base = traces[0]
    base_batch = base.batch_size
    base_envs = base.root.meta_static.get("environments")

    for idx, trace in enumerate(traces[1:], start=1):
        if trace.batch_size != base_batch:
            raise ValueError("Concatenated traces must share batch size")
        envs = trace.root.meta_static.get("environments")
        if base_envs is None or envs is None:
            continue
        if len(envs) != len(base_envs):
            raise ValueError("Concatenated traces must share environment count")
        for env_idx, (left, right) in enumerate(zip(base_envs, envs)):
            if left is not right:
                raise ValueError("Environment mismatch at index " f"{env_idx} for trace {idx}")


def _time_offsets(traces: Sequence[TraceTree]) -> list[int]:
    """Compute cumulative time offsets for concatenation."""
    offsets = [0]
    for trace in traces[:-1]:
        offsets.append(offsets[-1] + trace.length)
    return offsets


def _concat_nodes(nodes: Sequence[TraceNode], offsets: Sequence[int]) -> TraceNode:
    """Concatenate TraceNode data with time offsets."""
    base = nodes[0]
    merged = TraceNode()
    merged.meta_static = dict(base.meta_static)
    merged.data = _concat_data(nodes)
    merged.meta_sparse = _merge_meta_sparse(nodes, offsets)
    merged.events = _merge_events(nodes, offsets)

    child_names = set(base.children.keys())
    for name in child_names:
        if any(name not in node.children for node in nodes):
            raise ValueError(f"Child node '{name}' missing in some traces")
        merged.children[name] = _concat_nodes(
            [node.children[name] for node in nodes],
            offsets,
        )
    return merged


def _concat_data(nodes: Sequence[TraceNode]) -> dict[str, np.ndarray]:
    """Concatenate dense data arrays across nodes."""
    merged: dict[str, np.ndarray] = {}
    base = nodes[0]
    for key in base.data:
        if any(key not in node.data for node in nodes):
            raise ValueError(f"Data key '{key}' missing in some traces")
        merged[key] = np.concatenate([node.data[key] for node in nodes], axis=0)
    return merged


def _merge_meta_sparse(
    nodes: Sequence[TraceNode],
    offsets: Sequence[int],
) -> dict[str, list[MetaUpdate]]:
    """Merge sparse metadata updates with offsets."""
    merged: dict[str, list[MetaUpdate]] = {}
    for node, offset in zip(nodes, offsets):
        for key, updates in node.meta_sparse.items():
            merged.setdefault(key, [])
            merged[key].extend([MetaUpdate(t=upd.t + offset, value=upd.value) for upd in updates])
    return merged


def _merge_events(
    nodes: Sequence[TraceNode],
    offsets: Sequence[int],
) -> dict[str, list[Event]]:
    """Merge events with offsets."""
    merged: dict[str, list[Event]] = {}
    for node, offset in zip(nodes, offsets):
        for key, entries in node.events.items():
            merged.setdefault(key, [])
            merged[key].extend([Event(t=evt.t + offset, payload=evt.payload) for evt in entries])
    return merged
