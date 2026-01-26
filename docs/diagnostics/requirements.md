# Trace Tree Convention Requirements (Concise)

## Definitions

- **State**: nested dataclass tree at timestep $t$.
- **Trace**: time-indexed recording of many States over $T$ timesteps.
- **Trace node**: a node in the trace tree with `data`, `meta_static`, `meta_sparse`, `events`, and `children`.
- **Batch size ($B$)**: number of parallel environments in a rollout; constant over time.

## Requirements (EARS Notation)

### Structure mirroring

- WHEN a State is appended at timestep $t$, THE SYSTEM SHALL ensure the trace tree contains a corresponding node for every nested dataclass field in the State.
- WHEN a State field is a nested dataclass, THE SYSTEM SHALL map it to a trace child node with the same field name.

### Dense numeric data

- WHEN a State field value is numeric or array-like and has stable shape and dtype across time, THE SYSTEM SHALL record it as dense `data` and produce a NumPy ndarray whose first axis is time.
- WHEN a dense `data` field is per-environment, THE SYSTEM SHALL store it with shape `(T, B, ...)`.
- WHEN a dense `data` field is global (not per-environment), THE SYSTEM SHALL store it with shape `(T, ...)` without introducing an artificial batch axis.

### Batch invariants

- WHEN recording a rollout, THE SYSTEM SHALL treat batch size `B` as constant for the rollout.
- IF a batched field’s leading dimension differs from `B` at any timestep, THEN THE SYSTEM SHALL either raise a structured error (strict mode) or record the offending value as an event (lenient mode).

### Metadata and classification

- WHEN a State field value is non-numeric and time-invariant, THE SYSTEM SHALL store it in `meta_static`.
- WHEN a State field value is non-numeric and changes infrequently, THE SYSTEM SHALL store it in `meta_sparse` as time-stamped updates.
- WHEN a State field value is time-varying and not representable as a dense numeric array (e.g., strings, enums), THE SYSTEM SHALL store it in `events`.
- WHEN classification is ambiguous, THE SYSTEM SHALL provide an explicit override mechanism to force a field into `data`, `meta_static`, `meta_sparse`, or `events`.

### List-of-tensors (indexed children)

- WHEN a State field is a list of numeric arrays whose elements have different trailing shapes, THE SYSTEM SHALL represent that field as a container node with indexed children.
- WHEN representing list elements as indexed children, THE SYSTEM SHALL use stringified indices (`"0"`, `"1"`, ...) as child keys.
- WHEN iterating indexed children, THE SYSTEM SHALL use numeric index order.
- IF the list length changes across time, THEN THE SYSTEM SHALL either raise a structured error (strict mode) or store the list as an `events` stream for the affected timesteps (lenient mode).

### Events

- WHEN an event is recorded at timestep $t$, THE SYSTEM SHALL store it as an `(t, payload)` tuple in a named event stream at the appropriate node.
- WHEN slicing a trace in time from `[t0:t1)`, THE SYSTEM SHALL filter events to the selected time range.

### Sparse metadata

- WHEN `meta_sparse` updates are stored, THE SYSTEM SHALL preserve update order and timestamps.
- WHEN a user queries the value of a sparse metadata key at time $t$, THE SYSTEM SHALL resolve the value as the most recent update with timestamp `<= t`.

### Errors and diagnostics

- IF the shape or dtype of a dense `data` field changes across time, THEN THE SYSTEM SHALL emit a structured error including node path, field name, timestep, expected shape/dtype, and observed shape/dtype.

## Acceptance Criteria

- A batched scalar field recorded for $T$ steps produces an array shaped `(T, B)`.
- A batched vector field recorded for $T$ steps produces an array shaped `(T, B, D)`.
- A list-of-tensors field with `n` indices produces `n` child nodes under the container node, each with `data["value"]` shaped `(T, B, units[i])`.
- A time-varying string field is stored in `events` and is correctly filtered by time slicing.
- A sparse metadata key updated at timesteps `t=0` and `t=10` resolves correctly for `t<10` and `t>=10`.
