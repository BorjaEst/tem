# Figures package requirements

This document defines functional and non-functional requirements for the
`torch_tem.figures` package.

The goals are:

- Provide built-in, discoverable figures for evaluating TEM behavior.
- Provide stable interfaces so new figures can be added consistently.
- Ensure figures work for batched traces (multiple environments) and
  multi-frequency (multi-scale) model outputs.

## Definitions

- **Figure**: A `matplotlib.figure.Figure` object created by a plotting function.
- **Trace**: A `TraceBase` subclass that stores time-indexed model/world outputs.
- **WorldTrace**: A trace containing world steps (locations, observations, actions)
  plus batch-level `environments` and optional `visited` masks.
- **RolloutTrace**: A trace containing `WorldTrace` plus model `output` and
  internal `state` traces.
- **Figure module**: A file implementing a `plot(trace, ctx) -> Figure` function.
- **Registry**: Name-based mapping from stable figure names to `FigureSpec`.
- **Context (`FigureContext`)**: Runtime selection and styling options passed to
  plotting functions.
- **Primitives**: Low-level plotting helpers for environment-centric rendering
  (maps, action arrows, walk overlays).

## Requirements (EARS notation)

- **REQ-001 (Discoverability)**: WHEN a figure name is requested, THE SYSTEM
  SHALL resolve it via the global `REGISTRY` or raise a helpful error listing
  available names.
- **REQ-002 (Stable identifiers)**: WHEN a built-in figure is registered, THE
  SYSTEM SHALL use a stable, namespaced identifier (e.g., `overview.rate_maps`).
- **REQ-003 (Explicit registration)**: WHEN the figures package is used, THE
  SYSTEM SHALL allow registration of built-in figures by calling
  `register_builtin_figures()`.
- **REQ-004 (Idempotent registration)**: WHEN `register_builtin_figures()` is
  invoked multiple times, THE SYSTEM SHALL not create duplicate registrations
  or change existing registrations.
- **REQ-005 (Typed dispatch contract)**: WHEN a figure is registered, THE SYSTEM
  SHALL declare the accepted trace type via `FigureSpec.accepts` and SHALL
  reject non-`TraceBase` types.
- **REQ-006 (Single entrypoint contract)**: WHEN a figure module is implemented,
  THE SYSTEM SHALL expose a `plot(trace, ctx) -> Figure` function.
- **REQ-007 (Batch selection)**: WHEN a trace contains batch dimension $B>1$,
  THE SYSTEM SHALL allow selecting a single environment by `ctx.env_idx`.
- **REQ-008 (Multi-scale selection)**: WHEN model codes contain multiple scales
  (MultiScaleCode), THE SYSTEM SHALL allow selecting a single scale by
  `ctx.freq_idx`.
- **REQ-009 (Empty traces)**: IF the input trace has zero time steps, THEN THE
  SYSTEM SHALL return a valid `Figure` (e.g., with an informative “empty” title)
  OR raise a clear, actionable error.
- **REQ-010 (Index validation)**: IF `ctx.env_idx` or `ctx.freq_idx` are out of
  range for the provided trace, THEN THE SYSTEM SHALL raise `IndexError` with
  the valid range included.

## Requirements: plotting primitives

- **REQ-020 (Environment map rendering)**: WHEN `plot_map(environment, values)`
  is called, THE SYSTEM SHALL render one marker per location using
  normalized coordinates and hide axes.
- **REQ-021 (NaN handling)**: WHEN per-location `values` contain NaNs,
  THE SYSTEM SHALL render those locations in a visually distinguishable way
  without raising an exception.
- **REQ-022 (Action visualization)**: WHEN action arrows are requested,
  THE SYSTEM SHALL render transitions for actions with positive probability.
- **REQ-023 (Shiny highlighting)**: WHEN a location is marked as shiny,
  THE SYSTEM SHALL visually highlight it (e.g., red outline) in map plots.

## Requirements: persistence and logging

- **REQ-030 (Canonical artifacts)**: WHEN a figure is saved to disk,
  THE SYSTEM SHALL support vector output as PDF.
- **REQ-031 (Preview artifacts)**: WHEN preview images are needed,
  THE SYSTEM SHALL support raster output as PNG.
- **REQ-032 (TensorBoard logging)**: WHEN a logger with a SummaryWriter-like
  `experiment` is provided, THE SYSTEM SHALL log a rasterized figure image under
  a caller-specified tag.
- **REQ-033 (Deterministic output paths)**: WHEN an output path is generated,
  THE SYSTEM SHALL follow a deterministic naming convention and include optional
  step/version fields.

## Constraints

- **CON-001 (No global side-effects by default)**: Plot functions SHOULD NOT
  permanently mutate global matplotlib configuration. If styling is applied,
  it SHOULD be scoped via a context manager (e.g., `StyleConfig.apply_context()`).
- **CON-002 (Trace immutability)**: Plot functions SHALL NOT mutate traces.
- **CON-003 (Headless environments)**: Figure generation SHALL be compatible
  with non-interactive backends (CI/HPC runs).
- **CON-004 (Minimal coupling)**: Figure modules SHOULD depend only on trace
  public attributes and figure primitives, not on training loops.

## Guidelines

- **GUD-001 (Namespacing)**: Use `<domain>.<figure>` naming (e.g., `walk.statistics`).
- **GUD-002 (Error messages)**: Raise errors that include the invalid value and
  valid range, plus the figure name if available.
- **GUD-003 (Determinism)**: Stochastic elements (e.g., jitter in trajectories)
  SHOULD be controllable by seed arguments or context.
- **GUD-004 (Layout)**: Figures SHOULD call `tight_layout()` (or equivalent)
  before returning.
- **GUD-005 (Metadata)**: If `ctx.split_name` or `ctx.global_step` are provided,
  figures SHOULD include them in titles or annotations.

## Acceptance criteria

- **AC-001**: Given built-in registration has been called, When `REGISTRY.get()`
  is called with a valid name, Then it returns a `FigureSpec` with a callable
  `plot` and a `default_filename` derived from the name.
- **AC-002**: Given built-in registration has been called, When `REGISTRY.get()`
  is called with an invalid name, Then it raises `KeyError` listing available
  names.
- **AC-003**: Given a batched `RolloutTrace` with batch size $B$, When a plot
  function is called with `ctx.env_idx` outside `[0, B)`, Then it raises
  `IndexError` containing `[0, B)`.
- **AC-004**: Given a `WorldTrace` with environment locations, When
  `environment.layout` is plotted, Then a `Figure` is produced with a title
  containing `n_locations`.
- **AC-005**: Given per-location values containing NaN, When `plot_map()` is
  called, Then it completes without exception and produces visible patches.

## Test automation strategy

- **Unit tests**:
  - Validate `FigureSpec` type enforcement and `default_filename` generation.
  - Validate `FigureRegistry.get/list/validate` behavior.
  - Validate `WorldTrace.location_ids` shape and content.
- **Smoke tests** (matplotlib headless):
  - Create minimal synthetic `WorldTrace` and `RolloutTrace` and ensure each
    built-in figure returns a `Figure`.
  - Save to PDF/PNG and verify files exist and are non-empty.
