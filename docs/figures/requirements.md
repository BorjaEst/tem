# Figures module requirements

This document defines functional and non-functional requirements for the
`torch_tem.figures` submodule.

The figures submodule is responsible for:

- Defining a stable, discoverable interface for figure generation via a
  registry.
- Rendering matplotlib `Figure` objects from rollout `TraceTree` objects.
- Persisting figures as on-disk artifacts (PDF/PNG) and logging rasterized
  previews to TensorBoard.

Figures are generated from traces produced by `torch_tem.diagnostics`.

## Definitions

- **TraceTree**: Hierarchical time-indexed recording of rollout state and
  outputs (see `torch_tem.diagnostics.traces`).
- **Figure module**: A Python module that implements `plot(trace, ctx) -> Figure`.
- **FigureContext**: Runtime context for figure generation (e.g. `env_idx`,
  `freq_idx`, `global_step`, `split_name`).
- **FigureSpec**: Registry entry that binds a stable name to a plot function and
  metadata.
- **Registry name**: Stable identifier used in configuration and logs, using
  dotted namespaces (e.g. `spatial.structure`).
- **Artifact**: Saved figure file on disk (canonical: PDF).
- **Preview**: Rasterized image logged to TensorBoard.

## Requirements (EARS notation)

### Figure module interface

- **REQ-001 (Plot signature)**: WHEN a figure module is registered,
  THE SYSTEM SHALL provide a `plot(trace: TraceTree, ctx: FigureContext) -> Figure`
  callable.

- **REQ-002 (No persistence side effects)**: WHEN `plot(...)` is called,
  THE SYSTEM SHALL return a matplotlib `Figure` and SHALL NOT write files to
  disk or log to TensorBoard.

- **REQ-003 (Empty trace handling)**: WHEN `plot(...)` is called with
  `trace.length == 0`, THE SYSTEM SHALL return a valid `Figure` containing a
  clear “no data” indicator rather than failing.

- **REQ-004 (Index validation)**: WHEN `plot(...)` uses `ctx.env_idx` or
  `ctx.freq_idx`, THE SYSTEM SHALL validate index ranges against the trace and
  SHALL raise a clear error (or render a clear “missing/out of range” panel).

- **REQ-005 (Deterministic rendering)**: WHEN `plot(trace, ctx)` is called with
  the same inputs, THE SYSTEM SHALL produce the same visual output modulo
  matplotlib backend differences.

### Registry

- **REQ-010 (Stable name lookup)**: WHEN a client requests a figure by name,
  THE SYSTEM SHALL resolve it via a central registry mapping names to
  `FigureSpec`.

- **REQ-011 (Idempotent registration)**: WHEN registering a `FigureSpec` whose
  name already exists, THE SYSTEM SHALL treat the operation as a no-op.

- **REQ-012 (Validation)**: WHEN configuration provides a list of figure names,
  THE SYSTEM SHALL validate that every name is registered and SHALL fail fast
  with an error that lists unknown names and available names.

- **REQ-013 (List ordering)**: WHEN listing registry entries,
  THE SYSTEM SHALL return specs ordered alphabetically by name.

### Built-in figures

- **REQ-020 (Built-in registration entrypoint)**: WHEN
  `torch_tem.figures.register.register_builtin_figures()` is called,
  THE SYSTEM SHALL register all built-in figures.

- **REQ-021 (Built-in minimum set)**: WHEN built-in figures are registered,
  THE SYSTEM SHALL include at minimum the following names:
  - `overview`
  - `spatial.structure`

### Styling

- **REQ-030 (Central style config)**: WHEN figure styling is required,
  THE SYSTEM SHALL support a centralized style configuration that can be applied
  globally or as a context manager.

- **REQ-031 (Style isolation)**: WHEN a style context manager is used,
  THE SYSTEM SHALL restore previous matplotlib rcParams on exit.

### Persistence and logging (sinks)

- **REQ-040 (PDF as canonical artifact)**: WHEN saving figures to disk,
  THE SYSTEM SHALL support PDF output and SHALL create parent directories as
  needed.

- **REQ-041 (PNG output)**: WHEN saving figures as raster images,
  THE SYSTEM SHALL support PNG output and SHALL create parent directories as
  needed.

- **REQ-042 (Deterministic paths)**: WHEN constructing output paths,
  THE SYSTEM SHALL generate deterministic filenames based on
  `(base_dir, figure_name, step?, version?, extension)`.

- **REQ-043 (TensorBoard logging)**: WHEN logging a figure preview,
  THE SYSTEM SHALL rasterize the figure to an image and log it via a TensorBoard
  writer-compatible interface.

- **REQ-044 (Resource cleanup)**: WHEN logging or saving figures in long-running
  processes, THE SYSTEM SHALL provide a way to close figures to avoid memory
  leaks.

### Training integration (callback behavior)

Note: The Lightning callback is implemented in `torch_tem.callbacks.figures`.
These requirements define the expected interaction between training and
`torch_tem.figures`.

- **REQ-050 (Rank-0 safety)**: WHEN figures are generated during distributed
  training, THE SYSTEM SHALL generate artifacts only on rank 0.

- **REQ-051 (Batch capture)**: WHEN capturing evaluation batches for figures,
  THE SYSTEM SHALL snapshot any mutable components (e.g. visited masks) so that
  later mutations do not affect generated figures.

- **REQ-052 (Episode stitching)**: WHEN figures require temporal context longer
  than a single rollout chunk, THE SYSTEM SHALL support stitching consecutive
  batches into a single episode before trace collection.

- **REQ-053 (Non-fatal failures)**: IF a figure fails to generate due to missing
  trace paths or plotting errors, THEN THE SYSTEM SHALL surface the error for
  debugging and SHOULD continue training.

## Constraints

- **CON-001 (Matplotlib dependency)**: Figure generation uses matplotlib and
  SHALL remain compatible with headless execution (e.g. CI / servers).

- **CON-002 (TraceTree contract)**: Figure modules SHALL treat `TraceTree` as
  the single input source of rollout data, and SHOULD use
  `torch_tem.diagnostics.trace_access` helpers for robustness.

- **CON-003 (Separation of concerns)**: Plot construction (`plot(...)`) SHALL be
  decoupled from persistence/logging (sinks).

## Acceptance criteria

- **AC-001**: Given the process imports `torch_tem.figures.register` and calls
  `register_builtin_figures()`, when the registry is listed, then entries
  include `overview` and `spatial.structure`.

- **AC-002**: Given a configuration containing an unknown figure name, when the
  registry validates the list, then validation fails with an error that includes
  both the unknown name and the available names.

- **AC-003**: Given a base output directory and a step, when
  `make_figure_path(base_dir, "overview", step=10, extension="pdf")` is called,
  then the returned path is under `<base_dir>/figures/` and includes
  `overview_step10.pdf`.

- **AC-004**: Given a valid matplotlib `Figure`, when `save_pdf(fig, path)` is
  called with a non-existent parent directory, then the directory is created and
  the file is written.

- **AC-005**: Given `trace.length == 0`, when a built-in figure’s `plot(...)` is
  called, then a `Figure` is returned and no exception is raised.
