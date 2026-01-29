# Figures module requirements

This document defines functional and non-functional requirements for the figure
generation stack in `torch_tem.figures`.

The figures stack is responsible for:

- Providing reusable panel-level plot functions.
- Providing figure templates (orchestrators) that define layouts and shared
  rendering logic.
- Providing resolved figure modules that expose a uniform public API:
  `plot(trace: TraceTree, ctx: FigureContext) -> matplotlib.figure.Figure`.
- Registering figures under names for callback-driven generation.
- Supporting persistence and logging sinks (PDF, TensorBoard).

## Definitions

- **TraceTree**: Hierarchical rollout trace container
  (`torch_tem.diagnostics.traces.TraceTree`).
- **FigureContext**: Rendering context passed to figure plot functions
  (`torch_tem.figures.registry.FigureContext`).
- **Template**: Abstract base class that defines a layout and panel interface.
- **Resolved figure**: Public module that implements a template and exports a
  `plot(trace, ctx)` function.
- **Panel plot**: Reusable plotting helper that renders into a Matplotlib Axes.
- **Registry**: Name → figure specification mapping used for runtime dispatch.
- **Sink**: Output handler for saving/logging figures.

## Requirements (EARS notation)

### Public API and imports

- **REQ-001 (Stable plot signature)**: WHEN a resolved figure is exposed,
  THE SYSTEM SHALL provide a module-level function
  `plot(trace: TraceTree, ctx: FigureContext) -> matplotlib.figure.Figure`.

- **REQ-002 (Import ergonomics)**: WHEN a resolved figure named `X` exists,
  THE SYSTEM SHALL support `from torch_tem.figures import X` and
  `X.plot(trace, ctx)`.

- **REQ-003 (Public plots API)**: WHEN a panel plot is intended for reuse,
  THE SYSTEM SHALL expose it under `torch_tem.figures.plots`.

### Template orchestration

- **REQ-010 (Template lifecycle)**: WHEN a template instance is rendered via
  `plot()`, THE SYSTEM SHALL:
  1.  Create a Matplotlib Figure and named Axes according to a declarative
      layout.
  2.  Call each panel fill method in the template layout.
  3.  Run post-processing.
  4.  Apply configured shared colorbars.

- **REQ-011 (Panel method contract)**: WHEN a template declares a panel name,
  THE SYSTEM SHALL require the concrete implementation to provide a
  corresponding method (`fill_<panel>(ax)` or `<panel>(ax)`).

- **REQ-012 (Layout safety)**: IF a declared layout overlaps panels, contains
  invalid positions, or uses non-positive spans, THEN THE SYSTEM SHALL fail fast
  with an actionable error.

### Shared colorbars and scaling

- **REQ-020 (Colorbar mappable registration)**: WHEN a panel produces a
  color-mapped artist, THE SYSTEM SHALL support registering a mappable for
  shared colorbar generation.

- **REQ-021 (Colorbar groups)**: WHEN a template defines a colorbar group,
  THE SYSTEM SHALL generate one colorbar spanning the configured group panels.

- **REQ-022 (Deterministic group source)**: WHEN a colorbar group declares a
  source panel, THE SYSTEM SHALL use the source panel’s mappable for the group.

### Trace and context usage

- **REQ-030 (Context-driven selection)**: WHEN `FigureContext.env_idx` or
  `FigureContext.freq_idx` are provided, THE SYSTEM SHALL use them to select
  which environment and frequency module slices are rendered.

- **REQ-031 (Index validation)**: IF `env_idx` or `freq_idx` is out of bounds
  for the requested trace signals, THEN THE SYSTEM SHALL raise a clear error or
  validate/clamp via shared helper functions with documented behavior.

- **REQ-032 (Graceful empty/NaN handling)**: WHEN required trace arrays are
  empty or contain only non-finite values, THE SYSTEM SHALL render a valid
  figure without raising due to scaling edge cases.

### Registry and callback integration

- **REQ-040 (Idempotent built-in registration)**: WHEN built-in figure
  registration is invoked multiple times, THE SYSTEM SHALL not duplicate
  registrations and SHALL remain safe to call from callbacks.

- **REQ-041 (Registry validation)**: WHEN a client provides a list of figure
  names, THE SYSTEM SHALL validate all names are registered and raise a clear
  error listing any unknown names.

- **REQ-042 (Figure spec contract)**: WHEN a figure spec is registered,
  THE SYSTEM SHALL store at minimum: name, plot callable, default filename, and
  optional tags/description.

### Persistence and logging

- **REQ-050 (PDF output path rule)**: WHEN saving a figure to disk,
  THE SYSTEM SHALL save under `<base_dir>/figures/<name>(-step=<step>).pdf`.

- **REQ-051 (TensorBoard logging)**: WHEN TensorBoard logging is enabled and a
  compatible logger is provided, THE SYSTEM SHALL log the figure to a tag under
  `figures/<default_filename>`.

### Non-functional requirements

- **REQ-060 (Training robustness)**: IF any figure generation step raises an
  exception inside the training callback, THEN THE SYSTEM SHALL catch and
  report the error without terminating the training process.

- **REQ-061 (Resource hygiene)**: WHEN figures are generated repeatedly in a
  loop, THE SYSTEM SHALL allow callers to close figures after saving/logging to
  prevent memory leaks.

## Acceptance criteria

- **AC-001**: Given the built-in `overview` figure is registered, when a client
  calls `REGISTRY.get("overview").plot(trace, ctx)`, then a Matplotlib Figure is
  returned.

- **AC-002**: Given a resolved figure `overview`, when executing
  `from torch_tem.figures import overview`, then `overview.plot(trace, ctx)` is
  callable and returns a Matplotlib Figure.

- **AC-003**: Given a template defining a shared colorbar group across panels,
  when the figure is rendered, then exactly one shared colorbar is created for
  that group.

- **AC-004**: Given a trace with an empty trajectory or non-finite arrays,
  when a figure is rendered, then the figure renders without raising due to
  normalization or scaling errors.
