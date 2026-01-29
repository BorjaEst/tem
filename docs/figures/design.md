# Figures module design

This document describes the design of `torch_tem.figures`.

The module implements a small plotting framework with:

- **Templates** (orchestrators) that define a layout and rendering lifecycle.
- **Resolved figures** that extract data from a `TraceTree` and fill template
  panels.
- **Panel plots** as reusable drawing utilities.
- **A registry** for name-based dispatch and callback integration.
- **Sinks** to persist figures (PDF and TensorBoard).

## Goals

- Provide a uniform figure entry point:
  `plot(trace: TraceTree, ctx: FigureContext) -> matplotlib.figure.Figure`.
- Keep figure orchestration/layout logic centralized in templates.
- Keep reusable plotting primitives centralized under `torch_tem.figures.plots`.
- Support training-time dispatch through a registry (name → FigureSpec).
- Support persistence/logging via small, testable sink functions.

## Non-goals

- The figures module does not define trace semantics or trace collection.
- The figures module does not schedule figure generation; it is invoked by the
  training callback.
- The figures module does not define the data module or environment APIs.

## Package structure and responsibilities

- `torch_tem.figures.figures`
  - Template and orchestration logic.
  - Abstract base classes that define layouts and required panel methods.
  - Shared lifecycle: create layout → fill panels → post-process → colorbars.

- `torch_tem.figures.plots`
  - Reusable panel-level plotting functions.
  - These functions accept an Axes and the minimal data needed to render a
    panel and should return an Axes (or optionally a mappable).

- `torch_tem.figures.modules`
  - Resolved figures intended for direct import and interactive use.
  - Each resolved figure exports a module-level `plot(trace, ctx)` function.
  - Resolved figures typically encapsulate state in a class that inherits from
    a template and implements the template’s panel methods.

- `torch_tem.figures.registry` / `torch_tem.figures.register`
  - Registry data structures and built-in registration helper.

- `torch_tem.figures.sinks`
  - PDF saving and TensorBoard logging helpers.

## Public API surface

### Resolved figure modules

Each resolved figure module provides:

- `plot(trace: TraceTree, ctx: FigureContext) -> Figure`

This is the primary public API used by:

- Interactive analysis: `from torch_tem.figures import overview; overview.plot(trace, ctx)`
- Training callback dispatch through the registry.

### Registry-based dispatch

Built-in figures are registered as `FigureSpec` entries.
The training callback resolves figures by name and calls `spec.plot(trace, ctx)`.

Design intent:

- The callback does not need to import individual figure modules directly.
- Figure selection is controlled by configuration via names and tags.

### Panel plots

Plot helpers in `torch_tem.figures.plots` provide reusable building blocks.
They should be treated as stable APIs once exported via `plots.__init__`.

## End-to-end flow

1. Training/evaluation code produces a `TraceTree` (rollout trace).
2. A `FigureContext` is created (env index, frequency index, step metadata).
3. A `FigureSpec` is resolved from the registry by name.
4. The callback calls `spec.plot(trace, ctx)` to obtain a Matplotlib Figure.
5. Sinks persist the figure (PDF and/or TensorBoard).
6. The callback closes the figure to avoid leaks.

## Template Method pattern

### Template lifecycle

The base template owns the algorithm:

- Create figure + axes from a declarative layout (`LAYOUT`).
- Invoke panel methods to populate each axes.
- Apply global styling from context where present.
- Apply shared colorbars (`COLORBAR_GROUPS`).

Concrete templates:

- Define `LAYOUT` mapping panel names to grid positions and spans.
- Define `COLORBAR_GROUPS` to declare shared colorbars.
- Define an abstract interface for required panel methods.

Resolved figures:

- Validate `env_idx` / `freq_idx` and extract the relevant trace slices.
- Precompute shared ranges and normalizations.
- Implement the template’s panel methods by delegating to plot helpers.

### Panel naming convention

Panels are addressed by name. For a panel name `ratemap_a`, the template fills
it by calling either:

- `fill_ratemap_a(ax)` if present, otherwise
- `ratemap_a(ax)`

This supports both “fill\_\*” naming and direct panel-method naming.

## Shared colorbar strategy

### Mechanism

Panels may return a Matplotlib mappable, or store one on an Axes attribute.
Templates then use mappables to generate shared colorbars for a group.

### Determinism

To keep shared colorbars deterministic across runs and avoid choosing an
arbitrary axes’ artist, each group may declare a `source` panel. The group’s
colorbar is then derived from that source panel’s mappable.

## FigureContext and styling

`FigureContext` is kept intentionally small and is safe to extend.

Current templates optionally read additional style attributes from `ctx` using
`getattr` (for example, `style`, `color_cycle`, and `tick_fontsize`).

Design rule:

- Optional styling fields are duck-typed: templates should treat missing fields
  as “use Matplotlib defaults”.

## Extension guide: adding a new figure

To add a figure `my_figure`:

1. Add reusable panel plot helpers under `torch_tem.figures.plots` if needed.
2. Define or reuse a template under `torch_tem.figures.figures.templates`.
3. Implement a resolved figure under `torch_tem.figures.modules.my_figure`
   exporting `plot(trace, ctx)` and the template’s panel methods.
4. Export `my_figure` from `torch_tem.figures` to support
   `from torch_tem.figures import my_figure`.
5. Register the figure in the built-in registry bootstrap.

## Error handling and robustness

- Indexing into the trace should be validated (env/frequency selection).
- Panel plots should handle empty arrays and NaN-only arrays gracefully.
- Callback integration should remain robust by catching exceptions and closing
  figures after persistence.

## Test automation strategy (recommended)

- Unit tests for plot helpers: they run on synthetic inputs and return Axes.
- Smoke tests for each resolved figure: `plot(trace, ctx)` returns a Figure for
  a minimal trace fixture.
- Registry tests: built-in registration is idempotent and validation errors are
  actionable.
