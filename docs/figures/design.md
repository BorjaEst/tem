# Figures module design

This document describes the design of the `torch_tem.figures` package.

The design follows a layered visualization architecture:

1. **Plot primitives (axis-level)**: reusable drawing functions that operate on
   a caller-provided Matplotlib `Axes`.
2. **Figure modules (figure-level)**: orchestration functions that create the
   `Figure`, manage subplot layout, apply consistent style, and compose multiple
   primitives into a coherent multi-panel output.

This mirrors the separation commonly used by Matplotlib (Figure vs Axes) and by
libraries with Grammar-of-Graphics lineage (layered marks + higher-level
composition).

## Goals

- Provide consistent, reusable axis-level primitives for common visual elements
  (maps, trajectories, heatmaps, time series, insets).
- Provide figure-level orchestration for multi-panel layouts with consistent
  spacing, guide placement (legend/colorbar), and styling.
- Enable configuration-driven figure selection using stable names and a
  registry.
- Ensure headless execution is supported for CI and remote environments.
- Improve testability by enabling unit tests for primitives and integration
  tests for figure layouts.

## Non-goals

- Perfect pixel-identical rendering across Matplotlib versions/backends.
- A general-purpose charting API independent of TEM domain concepts.
- A full Grammar-of-Graphics implementation (encodings/scales as first-class
  objects). The design borrows the layering principle but stays Matplotlib-first.

## Architecture overview

### Package structure (conceptual)

```text
torch_tem.figures/
	plots/          # axis-level primitives: draw(ax, ...)
	figures/        # figure-level constructors/templates
	palettes/       # color palettes, colormap helpers
	style.py        # StyleConfig and style application
	registry.py     # FigureSpec, FigureContext, registry
	sinks.py        # save/log adapters (PNG/PDF/TensorBoard)
	modules/        # domain figure modules that return a Figure
	utils/          # shared helpers (formatting, annotation, etc.)
```

Notes:

- `plots/` contains functions that draw on a provided `Axes`.
- `modules/` contains callable figure definitions that create and return a
  Matplotlib `Figure` (often multi-panel).
- `registry.py` enables discoverable name-based generation.
- `sinks.py` handles I/O responsibilities.

### Dependency direction

Enforce a one-way dependency graph:

```text
modules/figures -> plots
modules/figures -> style/palettes/utils
modules/figures -> registry (for typing / context)

plots -> (palettes/utils)    # allowed
plots -X-> modules/figures   # forbidden
plots -X-> sinks/registry    # avoid coupling
```

This prevents circular imports and preserves reusability of primitives.

## Data flow

The typical flow for TEM visualizations is:

```text
TraceTree + FigureContext
	-> (registry lookup)
	-> figure module (creates Figure + Axes grid)
	-> plot primitives called per panel
	-> figure module finalizes layout + guides
	-> sinks save/log
```

### Sequence: generate and save a figure

```text
client
	|-- ctx = FigureContext(...)
	|-- spec = REGISTRY.get(name)
	|-- with ctx.style.apply_context():
	|     fig = spec.plot(trace, ctx)
	|-- save_png(fig, path)
	|-- close(fig)   (ownership rule in sinks)
```

## Interfaces and contracts

### FigureContext (inputs)

`FigureContext` carries runtime configuration that should not be hard-coded into
primitives:

- `env_idx`: which environment in a batched trace to visualize.
- `freq_idx`: which frequency/module index to visualize for multi-scale models.
- `figsize`: (width, height) in inches.
- `style`: style configuration (typically a `StyleConfig`).
- `global_step`: optional metadata for titles/logging.
- `split_name`: optional metadata for filenames/annotations.

Design rule:

- Plot primitives MAY accept fine-grained styling kwargs (color/linewidth/cmap)
  but SHOULD NOT consume `FigureContext` directly.

### FigureSpec and registry (discoverability)

Each figure is registered as a `FigureSpec`:

- `name`: stable identifier, used in config.
- `description`: human-readable intent.
- `plot(trace, ctx) -> Figure`: figure constructor.
- `default_filename`: optional default stem.
- `tags`: optional grouping.

Registry behavior:

- registration is idempotent
- listing order is deterministic
- validation errors list available names

### Plot primitives (axis-level)

Plot primitives are small, reusable drawing functions.

Contract:

- **Inputs**: an `Axes` plus data and explicit rendering parameters.
- **Side effects**: only add artists to the provided `Axes`.
- **Forbidden**: figure/subplot creation, layout orchestration, global style
  mutation, file I/O.
- **Outputs**: artist handles or structured results for composition.

Recommended signature patterns:

```python
def scatter(ax, x, y, *, color=None, label=None, **kwargs) -> PathCollection:
		...

@dataclass
class HeatmapResult:
		image: AxesImage
		vmin: float
		vmax: float

def heatmap(ax, values, *, cmap=None, vmin=None, vmax=None, **kwargs) -> HeatmapResult:
		...
```

Why structured results matter:

- Figure modules can build a single shared legend/colorbar across subplots.
- Tests can assert on returned handles without pixel comparisons.

### Figure modules (figure-level)

Figure modules:

- Create `Figure` and `Axes` grid (e.g., via `plt.subplots` or `GridSpec`).
- Apply style (prefer `StyleConfig.apply_context()` to bound global effects).
- Orchestrate calls to primitives, passing explicit `Axes` to each.
- Centralize guide placement (legend/colorbars).
- Enforce consistent spacing and labeling conventions.

Recommended figure module shape:

```text
plot(trace, ctx):
	with style context:
		fig, axes = build_layout(ctx.figsize)
		primitives.draw_panel_1(axes[0], ...)
		primitives.draw_panel_2(axes[1], ...)
		finalize_guides(fig, axes)
		finalize_titles(fig, ctx)
	return fig
```

## Styling and templates

### StyleConfig

Centralize style parameters in a `StyleConfig` object.

Design choices:

- Use a context manager (`rc_context`) to scope style changes.
- Keep figure-level defaults (dpi, base font size, facecolors) in one place.
- Allow modules to override local styling explicitly without mutating globals.

### Templates

Templates are reusable recipes for:

- standard multi-panel grid structures (e.g., overview pages)
- consistent title/annotation placement
- consistent legend/colorbar sizing and placement

Templates live at the figure-level because they allocate layout space.

## Error handling

### Input validation

Figure modules should validate inputs early and fail with actionable errors:

- invalid `env_idx` / `freq_idx`
- missing trace nodes / missing required arrays
- incompatible shapes for primitives

Plot primitives should validate only what they must (e.g., shape compatibility)
and raise concise `ValueError`/`TypeError` with enough context to debug.

### Failure modes

- **Global style leakage**: caused by setting `mpl.rcParams` without a context.
  Mitigation: use `StyleConfig.apply_context()` and keep style application in
  the figure-level.

- **Layout drift**: caused by primitives trying to "fix" spacing or add their
  own legends/colorbars. Mitigation: forbid layout orchestration in primitives.

- **Backend fragility**: caused by relying on implicit current axes/figure.
  Mitigation: always accept explicit `Axes` and return handles.

## Test automation strategy

### Unit tests (plots)

Test axis-level primitives without file I/O:

- create `Figure, Axes` in a headless backend
- call primitive with explicit `ax`
- assert artists were added (type/count) and returned handles are non-null
- assert no global style mutation (optional but recommended)

### Integration tests (figures)

Test figure modules as layout/orchestration:

- assert the number of axes and expected grid structure
- assert shared legend/colorbar placement policy (single vs per-axis)
- avoid pixel-perfect comparisons unless absolutely necessary

### CI considerations

- prefer Matplotlib non-interactive backend for tests
- avoid long-running visual regression tests

## Rationale

This design improves:

- **Modularity**: primitives are reusable across different layouts.
- **Testability**: primitives can be unit-tested in isolation; orchestration can
  be integration-tested via structural assertions.
- **Backend independence**: fewer implicit global state dependencies.
- **Layout consistency**: spacing and guide rules are centralized at the figure
  level instead of duplicated across primitives.
