"""Trace Tree implementation for batched simulation rollouts.

This module provides a clean, minimal implementation of the Trace Tree
Pattern with explicit handling of dense data, sparse metadata, and events.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from numbers import Number
from typing import Any, Iterable, Iterator, List, Optional

import numpy as np


@dataclass(frozen=True)
class MetaUpdate:
    """Time-stamped metadata update."""

    t: int
    value: Any


@dataclass(frozen=True)
class Event:
    """Time-stamped event payload."""

    t: int
    payload: Any


@dataclass
class TraceConfig:
    """Configuration for trace construction and validation.

    Attributes:
        strict: Whether to raise errors on invariant violations.
        allow_events: Whether to record events for irregular data.
        allow_meta_sparse: Whether to promote changing metadata to sparse.
        global_paths: Paths treated as global (no batch axis enforced).
        rebase_time_on_slice: Whether to rebase time to zero on slices.
    """

    strict: bool = True
    allow_events: bool = True
    allow_meta_sparse: bool = True
    global_paths: set[str] = field(default_factory=set)
    rebase_time_on_slice: bool = True


@dataclass
class TraceNode:
    """Node in a trace tree with dense data, metadata, and events."""

    data: dict[str, np.ndarray] = field(default_factory=dict)
    meta_static: dict[str, Any] = field(default_factory=dict)
    meta_sparse: dict[str, list[MetaUpdate]] = field(default_factory=dict)
    events: dict[str, list[Event]] = field(default_factory=dict)
    children: dict[str, "TraceNode"] = field(default_factory=dict)
    _buffers: dict[str, list[np.ndarray]] = field(default_factory=dict, repr=False)

    def child(self, name: str) -> "TraceNode":
        """Return (or create) a child node by name."""

        if name not in self.children:
            self.children[name] = TraceNode()
        return self.children[name]

    def record_event(self, key: str, t: int, payload: Any) -> None:
        """Record a time-stamped event payload."""

        self.events.setdefault(key, []).append(Event(t=t, payload=payload))

    def record_meta(self, key: str, t: int, value: Any, *, allow_sparse: bool) -> None:
        """Record metadata, promoting to sparse updates if the value changes."""

        if key in self.meta_sparse:
            updates = self.meta_sparse[key]
            if updates[-1].value != value:
                updates.append(MetaUpdate(t=t, value=value))
            return

        if key in self.meta_static:
            if self.meta_static[key] != value:
                if allow_sparse:
                    old = self.meta_static.pop(key)
                    self.meta_sparse[key] = [MetaUpdate(t=0, value=old), MetaUpdate(t=t, value=value)]
                else:
                    self.record_event(key, t, value)
            return

        self.meta_static[key] = value

    def append_data(self, key: str, value: np.ndarray) -> None:
        """Append a dense value to the buffer for a given key."""

        self._buffers.setdefault(key, []).append(value)

    def finalize(self, *, length: int, strict: bool) -> None:
        """Stack buffered data into dense arrays and recurse into children."""

        for key, values in self._buffers.items():
            if strict and len(values) != length:
                raise ValueError("Incomplete data buffer for key " f"{key!r}: expected {length}, got {len(values)}")
            self.data[key] = np.stack(values, axis=0)
        self._buffers.clear()

        for child in self.children.values():
            child.finalize(length=length, strict=strict)

    def slice_time(self, t0: int, t1: int, *, rebase: bool) -> "TraceNode":
        """Return a time-sliced copy of this node."""

        sliced = TraceNode()

        for key, arr in self.data.items():
            sliced.data[key] = arr[t0:t1]

        sliced.meta_static = dict(self.meta_static)
        sliced.meta_sparse = _filter_meta_sparse(self.meta_sparse, t0, t1, rebase)
        sliced.events = _filter_events(self.events, t0, t1, rebase)

        for name, child in self.children.items():
            sliced.children[name] = child.slice_time(t0, t1, rebase=rebase)

        return sliced


@dataclass
class TraceTree:
    """Trace tree builder and container for a rollout."""

    config: TraceConfig = field(default_factory=TraceConfig)
    root: TraceNode = field(default_factory=TraceNode)
    batch_size: Optional[int] = None
    length: int = 0

    def append(self, state: Any) -> None:
        """Append a state snapshot into the trace tree."""

        t = self.length
        _append_state(node=self.root, state=state, t=t, path=(), tree=self)
        self.length += 1

    def finalize(self) -> None:
        """Finalize the trace by stacking buffered data into arrays."""

        self.root.finalize(length=self.length, strict=self.config.strict)

    def slice_time(self, t0: int, t1: int) -> "TraceTree":
        """Return a time-sliced copy of the trace."""

        t0 = max(0, t0)
        t1 = min(self.length, t1)
        sliced = TraceTree(config=self.config)
        sliced.root = self.root.slice_time(t0, t1, rebase=self.config.rebase_time_on_slice)
        sliced.length = max(0, t1 - t0)
        sliced.batch_size = self.batch_size
        return sliced


def _append_state(*, node: TraceNode, state: Any, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a state tree into the trace tree (recursive)."""

    if _is_dataclass_instance(state):
        for field_info in fields(state):
            name = field_info.name
            value = getattr(state, name)
            _append_value(node=node, name=name, value=value, t=t, path=path, tree=tree)
        return

    raise TypeError("State must be a dataclass instance. " f"Got {type(state)!r} at path {"/".join(path) or "<root>"}.")


def _append_value(*, node: TraceNode, name: str, value: Any, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a single field value into the trace tree."""

    current_path = path + (name,)

    if _is_dataclass_instance(value):
        child = node.child(name)
        _append_state(node=child, state=value, t=t, path=current_path, tree=tree)
        return

    if _is_list_of_numeric_arrays(value):
        container = node.child(name)
        container.meta_static.setdefault("length", len(value))
        for idx, elem in enumerate(value):
            idx_node = container.child(str(idx))
            arr = _to_numeric_array(elem)
            _append_dense(node=idx_node, key="value", arr=arr, t=t, path=current_path + (str(idx), "value"), tree=tree)
        return

    arr = _to_numeric_array(value)
    if arr is not None:
        _append_dense(node=node, key=name, arr=arr, t=t, path=current_path + (name,), tree=tree)
        return

    node.record_meta(key=name, t=t, value=value, allow_sparse=tree.config.allow_meta_sparse)


def _append_dense(*, node: TraceNode, key: str, arr: np.ndarray, t: int, path: tuple[str, ...], tree: TraceTree) -> None:
    """Append a dense numeric array with validation."""

    if arr.dtype == object:
        _handle_irregular(node=node, key=key, t=t, value=arr, tree=tree)
        return

    _validate_batch(arr, path=path, tree=tree)
    _validate_shape(node, key, arr, path=path, tree=tree)
    node.append_data(key, arr)


def _validate_batch(arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree) -> None:
    """Validate or set the batch size for batched arrays."""

    if arr.ndim == 0:
        return

    full_path = "/".join(path)
    if full_path in tree.config.global_paths:
        return

    if tree.batch_size is None:
        tree.batch_size = int(arr.shape[0])
        return

    if int(arr.shape[0]) != tree.batch_size:
        _raise_or_event(tree, ValueError("Batch size mismatch at " f"{full_path}: expected {tree.batch_size}, got {arr.shape[0]}"))


def _validate_shape(node: TraceNode, key: str, arr: np.ndarray, *, path: tuple[str, ...], tree: TraceTree) -> None:
    """Validate shape and dtype stability for a dense key."""

    if key not in node._buffers or not node._buffers[key]:
        return

    prev = node._buffers[key][-1]
    if prev.shape != arr.shape or prev.dtype != arr.dtype:
        full_path = "/".join(path)
        _raise_or_event(tree, ValueError("Shape/dtype mismatch at " f"{full_path}: expected {prev.shape}/{prev.dtype}, " f"got {arr.shape}/{arr.dtype}"))


def _handle_irregular(*, node: TraceNode, key: str, t: int, value: Any, tree: TraceTree) -> None:
    """Handle irregular data by recording events or raising errors."""

    if tree.config.allow_events:
        node.record_event(key, t, value)
        return

    raise ValueError(f"Irregular value for key {key!r} at t={t}")


def _raise_or_event(tree: TraceTree, exc: Exception) -> None:
    """Raise in strict mode or swallow in lenient mode."""

    if tree.config.strict:
        raise exc


def _to_numeric_array(value: Any) -> Optional[np.ndarray]:
    """Convert a value to a numeric NumPy array if possible."""

    if value is None:
        return None

    if isinstance(value, Number):
        return np.asarray(value)

    if isinstance(value, np.ndarray):
        return value

    if hasattr(value, "detach") and callable(value.detach):
        try:
            return value.detach().cpu().numpy()
        except Exception:
            return None

    try:
        arr = np.asarray(value)
    except Exception:
        return None

    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None

    return arr


def _is_list_of_numeric_arrays(value: Any) -> bool:
    """Return True if value is a non-empty list/tuple of numeric arrays."""

    if not isinstance(value, (list, tuple)):
        return False
    if not value:
        return False
    return all(_to_numeric_array(item) is not None for item in value)


def _is_dataclass_instance(value: Any) -> bool:
    """Return True for dataclass instances (not classes)."""

    return is_dataclass(value) and not isinstance(value, type)


def _filter_events(events: dict[str, list[Event]], t0: int, t1: int, rebase: bool) -> dict[str, list[Event]]:
    """Filter events by time range and optionally rebase timestamps."""

    filtered: dict[str, list[Event]] = {}
    for key, entries in events.items():
        subset = [e for e in entries if t0 <= e.t < t1]
        if rebase:
            subset = [Event(t=e.t - t0, payload=e.payload) for e in subset]
        if subset:
            filtered[key] = subset
    return filtered


def _filter_meta_sparse(updates: dict[str, list[MetaUpdate]], t0: int, t1: int, rebase: bool) -> dict[str, list[MetaUpdate]]:
    """Filter sparse metadata updates by time range."""

    filtered: dict[str, list[MetaUpdate]] = {}
    for key, entries in updates.items():
        subset = [u for u in entries if t0 <= u.t < t1]
        if rebase:
            subset = [MetaUpdate(t=u.t - t0, value=u.value) for u in subset]
        if subset:
            filtered[key] = subset
    return filtered


def iter_nodes(root: TraceNode) -> Iterator[TraceNode]:
    """Depth-first iteration over trace nodes."""

    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(list(node.children.values())))
