## TEM Figures Submodule — Design

## Summary

This design preserves the existing registry-driven figure system while enabling
paper-style, square-grid spatial rate maps (e.g., “grid cell” style maps) in the
overview figure.

Key properties:

- Figures remain registry-driven via `FigureSpec` and executed as
  `spec.plot(trace, ctx)`.
- Figure modules remain stateless (pure `plot(trace, ctx)`).
- A new combined trace type, `RolloutTrace` (Option C), provides the minimal
  bundle needed for spatial rate maps: world geometry + per-step position IDs +
  model rollout outputs aligned in time.

## Goals and Non-goals

Goals:

- Support both training-time (Lightning callback) and offline scripts using the
  same registry and figure modules.
- Enable spatial 2D rate maps on square-grid environments using existing map
  primitives.
- Keep trace contracts explicit and testable (capability-based `PlotTrace`).
- Keep walk trajectory plotting deterministic by default.

Non-goals:

- Interactive dashboards.
- Replacing matplotlib sinks/style system.
- Adding heavy plotting dependencies beyond matplotlib.

## Existing Components (baseline)

- Registry + context: [src/torch_tem/figures/core/registry.py](src/torch_tem/figures/core/registry.py)
- Trace protocol: [src/torch_tem/figures/core/types.py](src/torch_tem/figures/core/types.py)
- Data trace: [src/torch_tem/figures/core/data_trace.py](src/torch_tem/figures/core/data_trace.py)
- Built-in registration: [src/torch_tem/figures/register.py](src/torch_tem/figures/register.py)
- Training callback entrypoint: [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)
- Model trace source: [src/torch_tem/diagnostics/traces.py](src/torch_tem/diagnostics/traces.py)
- Environment primitives (maps/walk overlays): [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py)
- Overview figure (current): [src/torch_tem/figures/modules/overview.py](src/torch_tem/figures/modules/overview.py)

## Definitions

- **FigureSpec**: Registry entry describing a figure (name, plot callable,
  accepted trace type, etc.).
- **FigureContext**: Runtime selection and styling (env index, frequency index,
  step/split metadata).
- **PlotTrace**: Protocol providing `batch_size`, `n_steps`, `meta`,
  `select_env(...)`, `downsample_time(...)`.
- **World geometry**: `World.locations[*]` with at least `o` (x) and `y` (y)
  coordinates (normalized to `[0, 1]`), plus connectivity and optional `shiny`.
- **Walk step**: The data pipeline uses the step structure documented in
  `DataTrace`: `[location_dict, observation_tensor, action_int]` where
  `location_dict["id"]` is the visited location ID.

## Architecture

### Registry Execution Model

- The registry remains the single source of truth for discoverable figure names.
- A figure module is a stateless function:
  - `plot(trace: PlotTrace, ctx: FigureContext) -> matplotlib.figure.Figure`
- Trace construction and selection occurs outside figure modules
  (callback/offline collectors).

### Runtime Context

`FigureContext` supplies:

- `env_idx`: selects a single environment from a batch trace.
- `freq_idx`: selects a scale/module from multiscale codes.
- `figsize`, `style`, and optional metadata: `global_step`, `split_name`.

## Trace Types and Contracts

All trace inputs to figure modules must implement the `PlotTrace` protocol
(from [src/torch_tem/figures/core/types.py](src/torch_tem/figures/core/types.py)).

### ModelTrace

Source: [src/torch_tem/diagnostics/traces.py](src/torch_tem/diagnostics/traces.py)

- Stores per-step model outputs/states from a rollout.
- Not sufficient (by design) for spatial maps because it does not guarantee a
  plot-friendly per-step location ID sequence aligned with the plotted tensors.

### DataTrace

Source: [src/torch_tem/figures/core/data_trace.py](src/torch_tem/figures/core/data_trace.py)

- Intended for environment/walk/split figures that do not require model outputs.
- Contains:
  - `worlds: list[World]` (batch of environments)
  - `walks: list[list[step]]` (batch of walks)
  - `visited: list[list[bool]] | None`
  - `meta: dict[str, Any]`

Contract notes:

- `n_steps` is derived from `walks[0]` in the current implementation and assumes
  consistent step counts across environments in the trace.

### RolloutTrace (Option C)

Purpose:

- Enables figures that require both model outputs and position alignment:
  spatial rate maps, joint diagnostic panels, etc.

Normative (required) fields:

- `worlds: list[World]`
- `walks: list[list[step]]`
- `location_ids: list[list[int]]`
  - Derived from `walks[env][t][0]["id"]`
- `model: ModelTrace`
- `meta: dict[str, Any]`

Normative invariants:

- `batch_size == len(worlds) == len(walks) == len(location_ids)`
- For each environment `e`:
  - `len(location_ids[e]) == len(walks[e]) == n_steps`
- Time alignment:
  - For each plotted model sequence `a_t` used in a figure,
    the time dimension MUST align with the `location_ids` time dimension after
    applying the same downsampling and max-step truncation rules.

Downsampling semantics (normative):

- `RolloutTrace.downsample_time(stride)` MUST keep indices
  `0, stride, 2*stride, ...` consistently across:
  - `walks`
  - `location_ids`
  - all per-step series inside `model` that are used for plotting

## Rate Map Computation

This section defines the computation that overview-style rate maps use.

Inputs:

- A per-step activity tensor for one scale, one env:
  - Example: `a[t, c]` for a chosen `freq_idx`
- Per-step visited locations:
  - `loc[t] = location_ids[t]` where `loc[t] in [0, n_locations)`

Output:

- A per-location scalar vector `m[loc_id]` (length `n_locations`) where:
  - `m[ℓ] = mean({ a[t, c] | loc[t] == ℓ })`

Edge cases (normative):

- If a location `ℓ` is never visited, `m[ℓ]` MUST be represented as missing
  (e.g., `NaN`) so rendering can show it as blank/unfilled.
- If a location is visited once, the mean is that single sample.
- If an activity value is `NaN` at some timestep, it SHOULD be excluded from
  the mean for that location (or documented if a different policy is used).

## Rate Map Rendering

Rendering uses existing map primitives:

- Use `plot_map(environment, values, shape="square")` from
  [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py)
  to draw square-tile maps.
- The “square” shape is a visual convention; coordinates still come from
  `World.locations[*]["o"]` and `["y"]`.

## Callback Integration (Online)

The Lightning callback is responsible for building traces from the training
batch and dispatching to registry specs.

Normative steps:

1. Obtain `walk` (and optionally `visited`) from the current batch.
2. Run a rollout (bounded by `max_rollout_steps`) to produce `ModelTrace`.
3. Build `DataTrace` for figures that accept it.
4. Build `RolloutTrace` for figures that accept it.
5. Dispatch figures by trace compatibility.

Dispatch rule (normative):

- The callback SHOULD use `isinstance(trace, spec.accepts)` to decide which
  trace to pass to a figure.

Implementation gap (current code):

- The current callback uses `spec.accepts == ModelTrace` and a `__name__`
  comparison for `DataTrace`. This should be replaced by `isinstance` to match
  this design and to support `RolloutTrace` cleanly.

## Offline Integration

Offline scripts should not depend on internal dataset fields beyond what the
collector APIs promise.

Existing collector:

- `collect_data_trace(datamodule, split) -> DataTrace` in
  [src/torch_tem/figures/core/data_trace.py](src/torch_tem/figures/core/data_trace.py)

Implementation gap (current code):

- The current `collect_data_trace` samples a batch but returns `dataset.walks`
  rather than the sampled `walk`. The collector should be clarified/fixed to
  either:
  - return the sampled batch trace, or
  - be renamed/documented as “dataset snapshot trace”.

Planned collector:

- `collect_rollout_trace(model, datamodule, split, *, max_steps, downsample_stride)
-> RolloutTrace`
  - samples a batch via `datamodule.sample_batch(split)`
  - runs `RolloutStream(model, walk, initial=None)`
  - builds `ModelTrace` using the same truncation/downsampling rules
  - derives aligned `location_ids`

## Determinism Strategy

- Walk trajectory visuals must be deterministic by default.
- Prefer local RNG injection (`numpy.random.Generator`) instead of using global
  `np.random`.

Note:

- `walk.trajectories` currently implements deterministic jitter internally; keep
  that behavior and do not regress it to global RNG usage.

## Error Handling Matrix

- Unknown figure name: `REGISTRY.validate(...)` raises `ValueError` listing
  available names.
- Incompatible trace type:
  - training callback: warn and skip
  - offline direct usage: raise `TypeError`
- Invalid `env_idx`: raise `IndexError` with valid range.
- Alignment failures (time length mismatch between model series and location IDs):
  raise `ValueError` describing expected and actual lengths.

## Testing Plan

- Unit tests for `DataTrace` protocol behavior (`select_env`, `downsample_time`).
- Unit tests for `RolloutTrace` protocol behavior and invariants.
- Unit tests for rate map aggregation (known walk + known activations yields
  known per-location means).
- Smoke test:
  - training callback generates `overview` plus at least one data figure from the
    same registry.

## Decision Record (compressed)

Decision: Add `RolloutTrace` to combine model + environment data |
Rationale: enables spatial rate maps without stateful figures or metadata hacks |
Impact: introduces a third trace type + collector/constructor path |
Review: revisit if rollout/batch contracts change.
