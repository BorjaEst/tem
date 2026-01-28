# Figures module requirements

This document defines functional and non-functional requirements for the
`torch_tem.figures` package architecture.

The figures stack is intentionally split into two layers:

- **Plots (axis-level primitives)**: small, reusable drawing functions that

  render onto a provided Matplotlib `Axes`.

- **Figures (figure-level orchestration)**: higher-level figure constructors

  that create the `Figure`, manage subplot layout, apply style/templates, and
  compose multiple primitives into consistent multi-panel outputs.

This separation follows the Matplotlib Figure–Axes model and the common
“primitives + orchestration” pattern used by Matplotlib/Seaborn/Plotly/Altair.

## Definitions

- **Axes**: A Matplotlib `Axes` instance; the target surface on which artists

  are drawn.

- **Figure**: A Matplotlib `Figure` instance; the container that owns axes,

  layout, and export configuration.

- **Plot primitive**: A pure-ish function that draws onto a provided `Axes` and

  returns artist handles or structured results.

- **Figure module**: A function/module that creates a `Figure` and one or more

  `Axes` (subplots), then orchestrates calls to plot primitives.

- **Theme/style**: Centralized configuration applied to Matplotlib (typically

  via `rcParams` or a context manager) to enforce consistent visuals.

- **Template**: A reusable layout/styling recipe for a family of figures.
- **Composition**: Combining multiple primitives across multiple axes into a

  single coherent figure (shared guides, shared scales, consistent spacing).

- **Sink**: An output adapter that saves or logs a generated figure (PNG/PDF,

  TensorBoard, etc.).

- **Registry**: A stable mapping from figure names to figure specifications for

  configuration-driven selection and validation.

## Requirements (EARS notation)

### Plot primitives (axis-level)

- **REQ-001 (Axes-first contract)**: WHEN a plot primitive is called, THE

  SYSTEM SHALL accept a caller-provided `Axes` as the drawing target.

- **REQ-002 (No figure creation)**: WHEN a plot primitive is called, THE SYSTEM

  SHALL NOT create a Matplotlib `Figure` or allocate subplots.

- **REQ-003 (No layout orchestration)**: WHEN a plot primitive is called, THE

  SYSTEM SHALL NOT call layout orchestration operations (e.g., creating
  `GridSpec`, `tight_layout`, `constrained_layout`) as part of the primitive.

- **REQ-004 (No global style side effects)**: WHEN a plot primitive is called,

  THE SYSTEM SHALL NOT mutate global Matplotlib state (e.g., `mpl.rcParams`) and
  SHALL NOT depend on implicit global current axes (e.g., `plt.gca()`).

- **REQ-005 (Deterministic output)**: WHEN a plot primitive is called with the

  same inputs and a freshly configured `Axes`, THE SYSTEM SHALL produce the same
  set of artists and properties (modulo Matplotlib version rendering
  differences).

- **REQ-006 (Return handles/results)**: WHEN a plot primitive creates artists,

  THE SYSTEM SHALL return artist handles or a structured result sufficient for
  figure-level composition (e.g., legend/colorbar construction).

### Figure orchestration (figure-level)

- **REQ-010 (Figure ownership)**: WHEN a figure module is called, THE SYSTEM

  SHALL create and own the Matplotlib `Figure` and all required `Axes` objects.

- **REQ-011 (Layout responsibility)**: WHEN a figure module is called, THE

  SYSTEM SHALL manage subplot layout (rows/cols, `GridSpec`, spacing,
  aspect/limits policy, alignment of colorbars/legends).

- **REQ-012 (Composition responsibility)**: WHEN composing multiple primitives,

  THE SYSTEM SHALL centralize guide placement (legends/colorbars) and other
  cross-axes coordination in the figure module (not in primitives).

- **REQ-013 (Styling entrypoint)**: WHEN a figure module is called with a style

  configuration, THE SYSTEM SHALL apply style consistently across all axes in
  the figure via a centralized mechanism.

- **REQ-014 (Context contract)**: WHEN a figure module is called with a

  `FigureContext` (or equivalent), THE SYSTEM SHALL interpret it consistently
  (e.g., `figsize`, `env_idx`, `freq_idx`, `split_name`, `global_step`).

### Style and themes

- **REQ-020 (Scoped style application)**: WHEN a style is applied for figure

  creation, THE SYSTEM SHALL provide a scoped mechanism (e.g., context manager)
  so that global Matplotlib configuration can be restored after figure
  generation.

- **REQ-021 (Single source of truth)**: WHEN style defaults are needed, THE

  SYSTEM SHALL read them from a single centralized configuration object (e.g.,
  `StyleConfig`) rather than duplicating defaults across figure modules.

### Registry and discoverability

- **REQ-030 (Stable names)**: WHEN a figure is registered, THE SYSTEM SHALL

  assign it a stable, unique name used for configuration-driven selection.

- **REQ-031 (Validation)**: WHEN a client requests figures by name, THE SYSTEM

  SHALL validate names and provide an actionable error listing available names
  for unknown entries.

- **REQ-032 (Deterministic listing)**: WHEN listing registered figures, THE

  SYSTEM SHALL return results in deterministic order.

### Sinks (save/log)

- **REQ-040 (Backend-safe output)**: WHEN saving or logging figures, THE SYSTEM

  SHALL support headless execution (non-interactive Matplotlib backend) without
  requiring a GUI.

- **REQ-041 (Resource ownership)**: WHEN a sink completes, THE SYSTEM SHALL NOT

  leak figure resources; it SHALL either close figures or document ownership
  rules clearly.

### Testability

- **REQ-050 (Primitive unit tests)**: WHEN testing plot primitives, THE SYSTEM

  SHALL allow tests to run without file I/O and without requiring full figure
  layouts (i.e., primitives must be testable via a provided `Axes`).

- **REQ-051 (Figure integration tests)**: WHEN testing figure modules, THE

  SYSTEM SHALL allow assertions on layout invariants (axes count, grid shape,
  shared guides) without depending on pixel-perfect image comparisons.

## Constraints

- **CON-001 (Rendering backend)**: The primary rendering backend SHALL be

  Matplotlib.

- **CON-002 (Headless compatibility)**: The system SHOULD work under headless

  environments (e.g., CI) using a non-interactive Matplotlib backend.

- **CON-003 (No implicit global state)**: Plot primitives MUST avoid reliance on

  global current figure/axes state to reduce coupling and improve composability.

## Acceptance criteria

- **AC-001**: Given a plot primitive and a caller-created `Axes`, when the

  primitive is called, then no new `Figure` is created and no global rcParams are
  mutated.

- **AC-002**: Given a figure module, when it is called with a `FigureContext`

  specifying `figsize`, then the returned `Figure` has that size.

- **AC-003**: Given a composed multi-panel figure, when it is generated, then

  legends/colorbars are placed consistently by the figure module (not duplicated
  by primitives).

- **AC-004**: Given an unknown figure name, when registry validation is

  performed, then the raised error includes the unknown names and a sorted list
  of available figure names.

- **AC-005**: Given a style configuration applied via a context mechanism, when

  figure generation exits the context, then Matplotlib rcParams are restored to
  their prior values.
