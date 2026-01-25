# TEM Figures Submodule — Design

## Summary

This design extends the existing `torch_tem.figures` infrastructure to support:

- domain-namespaced figure modules (e.g., `figures.environment.layout.plot(...)`)
- registry-driven figure generation for both model traces and data/environment traces
- deterministic walk trajectory plotting by default

The design is intentionally minimal: it keeps the existing `FigureSpec` and `REGISTRY` APIs and extends the callback to build the right trace objects.

## Existing Components (baseline)

- Registry types: [src/torch_tem/figures/core/registry.py](src/torch_tem/figures/core/registry.py)
- Trace protocol: [src/torch_tem/figures/core/types.py](src/torch_tem/figures/core/types.py)
- Style: [src/torch_tem/figures/style.py](src/torch_tem/figures/style.py)
- Sinks: [src/torch_tem/figures/sinks.py](src/torch_tem/figures/sinks.py)
- Environment drawing primitives: [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py)
- Training callback: [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)
- Example model figure: [src/torch_tem/figures/modules/overview.py](src/torch_tem/figures/modules/overview.py)

## Package Layout

Add domain subpackages under `torch_tem.figures`:

- `torch_tem/figures/__init__.py`
  - re-export domains: `environment`, `walk`, `split`, `overview`
  - re-export `style`, `sinks` (optional)

- `torch_tem/figures/modules/environment/`
  - `layout.py` → `plot(trace: DataTrace, ctx: FigureContext) -> Figure`

- `torch_tem/figures/modules/walk/`
  - `trajectories.py` → `plot(trace: DataTrace, ctx: FigureContext) -> Figure`
  - `statistics.py` → `plot(trace: DataTrace, ctx: FigureContext) -> Figure`

- `torch_tem/figures/modules/split/`
  - `statistics.py` → `plot(trace: DataTrace, ctx: FigureContext) -> Figure`

Model figures remain where they are (`torch_tem/figures/modules/*`) in v1.

## Core Design: Two Trace Families

### 1) ModelTrace

Used by existing training diagnostics. Produced in the callback via `ModelTrace.from_rollout(...)`.

### 2) DataTrace (new)

Used for environment/walk/split figures.

#### Data model

`DataTrace` is a lightweight, plot-oriented wrapper around a batch of environments and walks.

Proposed fields:

- `worlds: list[torch_tem.data.world.World]`
- `walks: list[list[list[Any]]]` (current walk step shape: `[location_dict, observation_tensor, action_int]`)
- `visited: list[list[bool]] | None`
- `meta: dict[str, Any]`

#### Protocol compliance

`DataTrace` SHALL implement the `PlotTrace` protocol:

- `batch_size: int` → number of environments
- `n_steps: int` → number of timesteps in the selected walk
- `select_env(env_idx)` → returns a single-env `DataTrace`
- `downsample_time(stride)` → downsample walk steps deterministically

This enables:

- uniform registry execution (`spec.plot(trace, ctx)`)
- type-based dispatch in the callback (`isinstance(trace, spec.accepts)`)

## Determinism Strategy

Current primitives (notably walk plotting) introduce randomness via jitter.

Design requirement: deterministic by default.

Approach:

- Add a deterministic RNG parameterization to the trajectory plotting path.
- Prefer injecting a local RNG (NumPy `Generator`) into the plotting function rather than using global `np.random`.
- Provide an explicit config flag (e.g., `deterministic: bool = True`) and optional `seed`.

Implementation detail:

- Either update [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py) to accept `rng`/`seed`, or keep primitives unchanged and implement deterministic jitter in the higher-level `walk.trajectories` figure.

## Styling Strategy

Use the existing `StyleConfig.apply_context()` from [src/torch_tem/figures/style.py](src/torch_tem/figures/style.py).

Rule:

- Figure modules should wrap plotting in a context manager:
  - `with (ctx.style or DEFAULT_STYLE).apply_context(): ...`

This prevents global matplotlib state leakage.

## Registry Naming and Tags

Stable registry names mirror domain paths:

- `overview` (existing)
- `environment.layout`
- `walk.trajectories`
- `walk.statistics`
- `split.statistics`

Tags:

- Model figures: `{"model", "rollout"}`
- Data figures: `{"data", "debug"}`

## Callback Integration

The existing callback currently builds only `ModelTrace`.

To support data figures in v1 without changing `FigureSpec`:

1. build `ModelTrace` as today
2. build `DataTrace` from the training batch and datamodule environments
3. for each requested figure name:
   - if `spec.accepts` matches `ModelTrace`, pass `ModelTrace`
   - if it matches `DataTrace`, pass `DataTrace`
   - otherwise skip with a warning

This keeps `FigureSpec` stable and avoids introducing a second registry.

## Offline Integration

Offline scripts should not access internal dataset state directly.

Provide a “collector” utility:

- `collect_data_trace(datamodule, split) -> DataTrace`
  which:
- samples a batch using `sample_batch(split)`
- retrieves the corresponding `World` objects for that split
- assembles a `DataTrace`

This standardizes access patterns and makes it easier to test.

## Error Handling Matrix

- Unknown figure name: `REGISTRY.validate(...)` raises `ValueError` with available names.
- Incompatible trace type:
  - training callback: warn and skip
  - offline direct usage: raise `TypeError` with expected type
- Invalid `env_idx`: raise `IndexError` with valid range.
- Missing optional deps for sinks (e.g., PIL): raise `ImportError` with install hint.

## Testing Plan

- Unit tests for `DataTrace` protocol behavior (`select_env`, `downsample_time`).
- Unit tests that each new figure module returns a `matplotlib.figure.Figure`.
- Determinism test: generate `walk.trajectories` twice with same seed/config and compare rendered pixel arrays (PNG buffer) or compare plotted coordinates.
- Smoke test: run the offline example path to generate figures to disk.

## Decision Record (compressed)

Decision: Use a `DataTrace` implementing `PlotTrace` | Rationale: enables registry + callback parity for data figures | Impact: adds a small new trace type and collector utility | Review: revisit if data pipeline batch contract is refactored.
