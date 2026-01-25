# TEM Figures Submodule — Requirements

## Objective

Provide a cohesive `torch_tem.figures` submodule that supports:

1. registry-driven figure generation during training (Lightning callback), and
2. offline/debug visualization of environments, policies, walks, and dataset splits.

The design MUST preserve the existing figure infrastructure:

- registry/specs in [src/torch_tem/figures/core/registry.py](src/torch_tem/figures/core/registry.py)
- sinks in [src/torch_tem/figures/sinks.py](src/torch_tem/figures/sinks.py)
- style in [src/torch_tem/figures/style.py](src/torch_tem/figures/style.py)
- environment drawing primitives in [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py)
- training integration in [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)

## Actors

- Researcher (offline): runs scripts such as [examples/data_datamodule.py](examples/data_datamodule.py)
- Trainer (online): uses [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py)

## Decisions (locked)

- Data/environment figures SHALL be callable through the registry and usable from the Lightning callback in v1.
- Walk trajectory plotting SHALL be deterministic by default.
- Data/environment figure inputs SHALL use a structured trace object (a “data trace”) rather than relying on ad-hoc access to internal dataset fields.

## Functional Requirements (EARS)

### Imports and Namespacing

- WHEN a user imports `from torch_tem import figures`, THE SYSTEM SHALL expose importable domains such that `figures.environment`, `figures.walk`, and `figures.split` are available.

### Registry and Discovery

- WHEN built-in figures are loaded, THE SYSTEM SHALL register all built-in figure specs into the global `REGISTRY` with stable names.
- WHEN a user configures the training callback with `figures=[...]`, THE SYSTEM SHALL validate requested names and fail fast with an actionable error if any are unknown.

### Figure Generation

- WHEN a registered figure is executed, THE SYSTEM SHALL call its `plot(trace, ctx)` callable and return a `matplotlib.figure.Figure`.
- WHEN a figure receives an incompatible trace type, THE SYSTEM SHALL skip generation (training) or raise a clear `TypeError` (offline usage), depending on entrypoint.

### Trace Types

- WHEN generating model figures in training, THE SYSTEM SHALL support `ModelTrace` inputs.
- WHEN generating environment/walk figures, THE SYSTEM SHALL support a dedicated `DataTrace` type that:
  - contains the environment(s)
  - contains the walk sequence(s)
  - implements the `PlotTrace` protocol (batch selection and time downsampling)

### Styling

- WHEN generating a figure, THE SYSTEM SHALL apply `StyleConfig` without permanently mutating global matplotlib state.

### Outputs

- WHEN configured to save artifacts, THE SYSTEM SHALL save PDF and/or PNG outputs using sinks.
- WHEN configured to log previews, THE SYSTEM SHALL log TensorBoard images via sinks.

### Determinism

- WHEN plotting walk trajectories, THE SYSTEM SHALL be deterministic by default.
- IF a user requests non-determinism, THEN THE SYSTEM SHALL provide an explicit configuration switch to enable randomness.

## Non-Functional Requirements

- THE SYSTEM SHALL keep figure modules small and composable (“one module = one figure”).
- THE SYSTEM SHALL preserve backward compatibility with legacy plotting primitives re-exported elsewhere.
- THE SYSTEM SHALL avoid introducing new heavy plotting dependencies (matplotlib is sufficient).
- THE SYSTEM SHALL be testable with fast unit tests.

## Acceptance Criteria

- Offline: [examples/data_datamodule.py](examples/data_datamodule.py) can generate at least:
  - `environment.layout`
  - `walk.trajectories`
  - `walk.statistics`
  - `split.statistics` (if implemented)
    without placeholders (`...`) and without exceptions.
- Online: the Lightning callback can generate both `overview` and at least one data figure from the same registry using the batch available at `on_train_batch_start`.
- Determinism: repeated runs with the same seed/config produce identical trajectory visuals (within rasterization tolerances).
