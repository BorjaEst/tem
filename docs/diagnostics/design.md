# Trace Tree Convention (Concise)

This document defines the **Trace Tree convention** for logging and analyzing batched simulation rollouts using the Trace Tree Pattern.

## Purpose

- Provide a stable mapping from nested per-timestep **State** objects to time-indexed **Trace** objects.
- Enforce consistent tensor/array shapes with a constant batch size $B$ for the duration of a rollout.
- Provide explicit escape hatches for non-dense signals via **events** and **sparse metadata**.

## Definitions

- **State**: nested dataclass tree for a single timestep $t$.
- **Trace**: time-indexed recording across timesteps $t \in [0, T)$ mirroring the State structure.
- **Node path**: `/`-separated names identifying a node in the trace tree (e.g., `world/agent/sensors`).
- **Batch size ($B$)**: number of parallel environments; constant within a rollout.
- **Time length ($T$)**: number of recorded timesteps.

## Trace Node Structure

Each trace node is a container with four channels plus children:

- **`data`**: dense numeric arrays stored as NumPy ndarrays shaped `(T, ...)`.
- **`meta_static`**: time-invariant metadata for the node.
- **`meta_sparse`**: infrequently changing metadata stored as time-stamped updates.
- **`events`**: time-stamped records for non-dense or categorical time-varying data.
- **`children`**: mapping from child name to child trace node.

Conceptual schema:

```text
TraceNode
  data: Dict[str, ndarray]          # dense numeric arrays, first axis is time
  meta_static: Dict[str, Any]       # constant for the rollout (or chunk)
  meta_sparse: Dict[str, List[MetaUpdate]]
  events: Dict[str, List[Event]]
  children: Dict[str, TraceNode]

MetaUpdate = (t: int, value: Any)
Event      = (t: int, payload: Any)
```

## Shape Convention (Time + Batch)

The trace system logs **batched** signals with consistent conventions:

- Any per-environment value at timestep $t$ has leading shape `(B, ...)`.
- Any traced dense signal is stacked along the time axis to `(T, B, ...)`.
- Truly global (non-batched) scalar signals may be stored as `(T,)` when they are not per-environment.

Examples:

```text
State scalar per env:      (B,)           -> Trace: (T, B)
State vector per env:      (B, D)         -> Trace: (T, B, D)
State matrix per env:      (B, D1, D2)    -> Trace: (T, B, D1, D2)
State global scalar:       () or (1,)     -> Trace: (T,)          (recommended)
```

## Mapping Rules: State → Trace

For each timestep, the collector transforms a State tree into a Trace tree.

### Dataclass / nested objects

- Dataclass-valued fields become **child nodes**.
- The child node name matches the field name.

### Dense numeric fields

- Numeric/array-like fields that have a stable shape across time are stored in `data`.
- Each dense field is stored as a NumPy ndarray with first axis time.

### Metadata fields

- Non-numeric, time-invariant values are stored in `meta_static`.
- Non-numeric values that change rarely are stored in `meta_sparse` as updates.

### Events

- Non-numeric values that change frequently or irregularly (e.g., strings, enums) SHOULD be stored in `events`.
- Ragged numeric values (variable length) SHOULD be stored in `events` unless a padding strategy is explicitly configured.

## List-of-Tensors Convention (Option A: Indexed Children)

Some State fields are lists of arrays with different trailing shapes, e.g. a multiscale representation:

```text
cells: List[x_i] where x_i has shape (B, units[i]) and i in [0, n)
```

This is represented as a container node with indexed children:

```text
parent.children["cells"]
  .meta_static["length"] = n
  .children["0"].data["value"] -> (T, B, units[0])
  .children["1"].data["value"] -> (T, B, units[1])
  ...
  .children[str(n-1)].data["value"] -> (T, B, units[n-1])
```

Notes:

- Indices are encoded as **strings** (`"0"`, `"1"`, ...) for stable path semantics and serialization.
- Child iteration MUST be in numeric index order, not lexicographic order.
- The container node MAY store helpful descriptors such as `meta_static["units_by_index"]` or `meta_static["shapes_by_index"]`.

## Validation and Failure Modes

The collector validates invariants at append-time (preferred) or finalize-time (acceptable for performance):

- **Batch invariant**: for batched signals, the leading dimension is always `B`.
- **Dense invariant**: for each `data[key]`, the trailing shape and dtype are stable across time.
- **List invariant**: for list fields, list length is stable across time, and each index has stable trailing shape.

If invariants are violated:

- In **strict mode**, raise an error containing `(node_path, field, timestep, expected_shape, observed_shape)`.
- In **lenient mode**, store the offending value as an `events` entry for that node/field and continue.

## Data Flow (Append + Finalize)

```text
for t in 0..T-1:
  state_t = step()
  append(trace_root, state_t)

finalize(trace_root):
  for each node in tree:
    for each key in buffered data:
      node.data[key] = np.stack(buffer[key], axis=0)   # (T, ...)
```

## Tradeoffs (Why These Conventions)

- **NumPy dense arrays**: enable fast slicing `(t0:t1)` and vectorized analysis; require stable shapes.
- **Indexed children for lists**: preserves hierarchy without forcing ragged object arrays.
- **`events` + `meta_sparse`**: prevents the system from collapsing when encountering categorical time series or ragged numeric outputs.
