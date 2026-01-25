## TEM Figures Submodule — Requirements

## Objective

Provide a cohesive `torch_tem.figures` submodule that supports:

1. registry-driven figure generation during training (Lightning callback), and
2. offline/debug visualization of environments, walks, dataset splits, and model
   rollouts.

The system MUST preserve the existing figure infrastructure:

- registry/specs: [src/torch_tem/figures/core/registry.py](src/torch_tem/figures/core/registry.py)
- trace protocol: [src/torch_tem/figures/core/types.py](src/torch_tem/figures/core/types.py)
- sinks: [src/torch_tem/figures/sinks.py](src/torch_tem/figures/sinks.py)
- style: [src/torch_tem/figures/style.py](src/torch_tem/figures/style.py)
- training integration: [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)

## Actors

- Researcher (offline): runs scripts such as
  [examples/data_datamodule.py](examples/data_datamodule.py)
- Trainer (online): uses [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)

## Definitions

- **PlotTrace**: Protocol defining `batch_size`, `n_steps`, `meta`,
  `select_env(env_idx)`, `downsample_time(stride)`.
- **ModelTrace**: Plot-ready model rollout trace (per-step outputs/states).
- **DataTrace**: Plot-ready environment/walk trace.
- **RolloutTrace**: Plot-ready combined trace (Option C) that aligns model
  outputs with per-step visited location IDs.
- **Location ID**: Integer index into `World.locations`, accessed from the data
  pipeline via `location_dict["id"]`.
- **Rate map**: Per-location aggregation of activity values by visited location.

## Decisions (locked)

- **DEC-001**: The figure registry remains the single source of truth for
  discoverable figure names.
- **DEC-002**: Figure implementations in `torch_tem.figures.modules` remain
  stateless `plot(trace, ctx)` functions.
- **DEC-003**: Walk trajectory plotting is deterministic by default.
- **DEC-004**: Option C is adopted: introduce `RolloutTrace` to support combined
  model+position figures (square-grid rate maps).

## Requirements (EARS)

### Imports and Namespacing

- **REQ-IMP-001**:
  WHEN a user imports `from torch_tem import figures`,
  THE SYSTEM SHALL expose domains such that `figures.environment`,
  `figures.walk`, and `figures.split` are importable.

### Registry and Discovery

- **REQ-REG-001**:
  WHEN built-in figures are loaded,
  THE SYSTEM SHALL register all built-in figure specs into the global `REGISTRY`
  with stable names.

- **REQ-REG-002**:
  WHEN a user configures the training callback with `figures=[...]`,
  THE SYSTEM SHALL validate requested names and fail fast with an actionable
  error if any are unknown.

### Figure Execution

- **REQ-EXE-001**:
  WHEN a registered figure is executed,
  THE SYSTEM SHALL call its `plot(trace, ctx)` callable and return a
  `matplotlib.figure.Figure`.

- **REQ-EXE-002**:
  WHEN a figure receives an incompatible trace type,
  THE SYSTEM SHALL skip generation during training (with a warning) OR raise
  `TypeError` in offline usage.

### Trace Types

- **REQ-TRC-001**:
  WHEN generating model-only figures,
  THE SYSTEM SHALL support `ModelTrace` inputs from
  [src/torch_tem/diagnostics/traces.py](src/torch_tem/diagnostics/traces.py).

- **REQ-TRC-002**:
  WHEN generating environment/walk figures,
  THE SYSTEM SHALL support `DataTrace` inputs from
  [src/torch_tem/figures/core/data_trace.py](src/torch_tem/figures/core/data_trace.py)
  that:
  - include `World` geometry
  - include walk sequences
  - implement `PlotTrace`

- **REQ-TRC-003**:
  WHEN generating combined model+position figures,
  THE SYSTEM SHALL support `RolloutTrace` inputs that:
  - include `World` geometry
  - include walk sequences
  - include per-step `location_id` sequences derived from the walk
  - include model rollout outputs aligned to the same timesteps
  - implement `PlotTrace`

- **REQ-TRC-004 (Alignment)**:
  WHEN `RolloutTrace.downsample_time(stride)` is called,
  THE SYSTEM SHALL downsample `walks`, `location_ids`, and any plotted model
  time series using the same kept indices `0, stride, 2*stride, ...`.

### Overview Rate Maps

- **REQ-OVR-001**:
  WHEN generating the `overview` figure with a `RolloutTrace`,
  THE SYSTEM SHALL render at least one square-grid spatial rate map for a
  selected frequency scale `ctx.freq_idx`.

- **REQ-OVR-002**:
  IF a location is never visited in the displayed rollout,
  THEN THE SYSTEM SHALL render that location as missing/blank (or a consistent
  sentinel) rather than fabricating values.

- **REQ-OVR-003 (Transition)**:
  WHEN generating the `overview` figure with only a `ModelTrace`,
  THE SYSTEM SHOULD provide a sensible fallback visualization (e.g., time-series
  heatmaps) rather than failing, until `RolloutTrace` is fully integrated
  everywhere.

### Styling

- **REQ-STY-001**:
  WHEN generating a figure,
  THE SYSTEM SHALL apply `StyleConfig` without permanently mutating global
  matplotlib state.

### Outputs

- **REQ-OUT-001**:
  WHEN configured to save artifacts,
  THE SYSTEM SHALL save PDF and/or PNG outputs using sinks.

- **REQ-OUT-002**:
  WHEN configured to log previews,
  THE SYSTEM SHALL log TensorBoard images via sinks.

### Determinism

- **REQ-DET-001**:
  WHEN plotting walk trajectories,
  THE SYSTEM SHALL be deterministic by default.

- **REQ-DET-002**:
  IF a user requests non-determinism,
  THEN THE SYSTEM SHALL provide an explicit configuration switch to enable
  randomness.

## Non-Functional Requirements

- **NFR-001**: Figure modules remain small and composable (“one module = one figure”).
- **NFR-002**: No new heavy plotting dependencies (matplotlib is sufficient).
- **NFR-003**: Testability with fast unit tests.

## Acceptance Criteria

- **AC-001 (Offline)**:
  Given a configured datamodule, When running
  [examples/data_datamodule.py](examples/data_datamodule.py),
  Then `environment.layout` and `walk.trajectories` are generated without
  placeholders and without exceptions.

- **AC-002 (Online)**:
  Given training is running, When the Lightning callback triggers,
  Then it can generate `overview` and at least one data figure from the same
  registry using the current batch available at `on_train_batch_start`.

- **AC-003 (Rate maps)**:
  Given a `RolloutTrace`, When `overview` is executed,
  Then at least one square-grid rate-map panel is rendered using world geometry
  and per-step `location_id` alignment.

- **AC-004 (Determinism)**:
  Given the same seed/config, When generating `walk.trajectories` repeatedly,
  Then visuals are identical within rasterization tolerances.
