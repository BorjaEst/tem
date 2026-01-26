# Figures package design

This document describes the design of the `torch_tem.figures` package: its
components, the data flow from traces to matplotlib figures, and the extension
pattern for adding new figures.

## Goals

- Provide a coherent “figure ecosystem” for TEM analysis.
- Make figures discoverable by stable name (configuration-friendly).
- Keep figure code modular (primitives vs multi-panel compositions).
- Support batched traces and multi-scale model outputs.
- Support both artifact saving (PDF/PNG) and training-time preview logging.

## Additional goals: validation-oriented monitoring

- Make training-time figure generation reflect generalization by default
  (validation split).
- Support two sampling modes:
  - Contiguous episodes for temporal diagnostics.
  - Aggregated coverage for spatial/statistical maps.
- Provide deterministic and repeatable figure inputs (identical per validation
  epoch when configured).

## Non-goals

- The figures package does not own training loops or dataset construction.
- The figures package does not define the TEM model; it consumes trace objects.
- The figures package does not implement interactive GUIs.

## Scope note

This document describes the design of the figures package itself and the
recommended integration pattern for training-time figure generation via the
Lightning callback in `torch_tem.callbacks`.

## Architecture overview

The package is split into five layers:

1. **Trace layer** (`torch_tem.diagnostics.traces`)
   - Defines `TraceBase` and concrete traces such as `WorldTrace` and `RolloutTrace`.
   - Traces are sequences over time and carry batch dimension.

1. **Registry layer** (`torch_tem.figures.registry`)
   - Defines `FigureContext`, `FigureSpec`, and `FigureRegistry`.
   - Provides a global registry instance `REGISTRY`.

1. **Registration layer** (`torch_tem.figures.register`)
   - Defines `register_builtin_figures()` which registers built-in figures.
   - Keeps registration separate to avoid import cycles.

1. **Figure modules** (`torch_tem.figures.modules.*`)
   - Namespaced folders by domain: `overview/`, `environment/`, `walk/`, `split/`.
   - Each module exposes a `plot(trace, ctx) -> Figure` function.

1. **Utilities**
   - **Primitives** (`torch_tem.figures.primitives`): environment-centric drawing utilities.
   - **Styling** (`torch_tem.figures.style`): centralized matplotlib rcParams via `StyleConfig`.
   - **Sinks** (`torch_tem.figures.sinks`): saving and logging outputs.

## Training-time integration design (callback)

### Overview

The training-time figure generation mechanism lives outside the figures package
as a Lightning callback (currently `FiguresCallback`). The callback is
responsible for:

- Selecting which split to visualize (default: validation).
- Constructing one or more traces (`RolloutTrace`, `WorldTrace`, etc.) from
  batches sampled from that split.
- Dispatching requested figure specs by name through `REGISTRY`.
- Persisting outputs via sinks (PDF artifacts and TensorBoard images).

### Trigger policy

Recommended default:

- Trigger figure generation during the validation loop (e.g.
  `on_validation_epoch_end` or `on_validation_batch_start` with `batch_idx==0`).
- Keep step-based gating (`every_n_steps`) but evaluate it at validation time
  against `trainer.global_step`.

Rationale:

- Validation-oriented figures align with generalization monitoring.
- Training dynamics are already available via scalars logged every step.

### Sampling modes

The callback should support two sampling modes without requiring additional
dataset splits:

1. **Contiguous episode** (temporal diagnostics)
   - Sample a single batch/episode (typically the first validation batch after
     reset).
   - Construct a `RolloutTrace` preserving temporal ordering.

2. **Aggregated coverage** (spatial/statistical maps)
   - Sample multiple validation batches and concatenate steps to improve
     occupancy and rate-map stability.
   - Construct an aggregated `RolloutTrace` (or equivalent trace) for figures
     that benefit from coverage.

Figure selection should be controlled by configuration (e.g., two lists or a
name->mode mapping) and/or by figure tags.

### Determinism and repeatability

The data generation stack currently includes seeded validation/test datasets,
but repeatability requires addressing two properties:

- **Statefulness**: `TEMDataset` mutates internal walks and visited masks while
  iterating. To achieve identical inputs per validation epoch, the dataset must
  support a reset operation that restores deterministic initial state.
- **World RNG**: `World` currently relies on global NumPy randomness. For seeded
  datasets, stochastic world generation (start locations, shiny placement,
  random observation assignments, action sampling) should instead be driven by
  an injected RNG (`np.random.Generator`) owned by the dataset.

Recommended interfaces:

- `TEMDataset.reset()`: restores RNG state and reinitializes environments, walks
  and visited masks.
- `DataModule.reset_split("validate"|"test")`: convenience wrapper for callback.
- `World(..., rng=Generator)`: all stochastic choices come from `rng`.

### Trace construction

For a sampled batch `(chunk, visited)`:

- Build a `RolloutTrace` via the canonical constructor
  `RolloutTrace.from_batch(...)`.
- Apply `downsample_time(stride)`.
- Optionally cap time steps for episode traces.

For aggregated traces:

- Collect `N` batches from validation after reset.
- Concatenate time steps (world + output + state) in a consistent order.
- Ensure batch dimension consistency (same environments in the batch or a clear
  contract describing how aggregation behaves).

### Configuration surface

Suggested additions to callback settings (exact field names are implementation
details):

- Split selection: `split = "validate" | "test"` (default: `"validate"`).
- Repeatability: `reset_each_val_epoch: bool`.
- Sampling:
  - `episode_steps` (or reuse `max_rollout_steps` explicitly as episode length).
  - `aggregate_batches` (number of validation batches to concatenate).
- Figure grouping:
  - `figures_episode: list[str]`
  - `figures_aggregate: list[str]`
  - optional `figures_static: list[str]`

### Spatiotemporal interpretability

For spatially aggregated maps (rate maps, occupancy maps), temporal structure
is intentionally collapsed. To support inspection of revisits and temporal
ordering, prefer adding a companion panel rather than encoding time directly in
the rate-map colormap.

Recommended companion visualizations:

- 2D trajectory colored by time (early->late) with optional activation overlay.
- Location-id vs time (or position vs time) strips aligned with activation.

## Key data contracts

### TraceBase

`TraceBase[TStep]` is a `collections.abc.Sequence` over time with:

- `__len__()` = number of time steps.
- `batch_size` property = number of parallel environments $B$.
- `from_iter()` / `attach()` for streaming construction.
- `downsample_time(stride)` for time subsampling.

Design notes:

- Traces are append-only during construction.
- Traces store the actual data in Python lists; `get_item()` reconstructs a step object on demand.

### WorldTrace

`WorldTrace` models world observations across time:

- Batch-level:
  - `environments: list[World]` (length $B$)
  - `visited: list[list[bool]] | None` (optional)
- Time series (length $T$):
  - `locations: list[list[LocationLabel]]` where inner list length $B$
  - `observations: list[Observation]` where each item has batch dimension
  - `actions: list[list[Action]]` where inner list length $B$
- Convenience:
  - `location_ids -> list[list[int]]` returns per-env series of visited location ids

### RolloutTrace

`RolloutTrace` models a full TEM rollout:

- `world_step: WorldTrace`
- `output: TEMOutputTrace` (inference/generative/reconstruction)
- `state: TEMStateTrace` (LEC/MEC/HPC internal states)

Construction uses `RolloutStream(model, batch[0], initial)` so figure modules can
assume time alignment between `world_step`, `output`, and `state`.

### FigureContext

`FigureContext` is passed to every figure module:

- `env_idx`: which environment in a batch to render
- `freq_idx`: which scale in a multi-scale code to render
- `figsize`: figure size for `plt.subplots`
- `style`: optional style config (expected to support `apply_context()`)
- `global_step` / `split_name`: optional metadata for titles/annotations

## FigureSpec

`FigureSpec` binds a stable name to a plotting function:

- `name`: namespaced stable id (e.g., `walk.statistics`)
- `description`: human-readable description
- `plot`: callable `(trace, ctx) -> Figure`
- `accepts`: a `TraceBase` subclass used for validation/dispatch
- `default_filename`: derived from `name` by replacing `.` with `_` unless explicitly set
- `tags`: immutable tag set for filtering/grouping

## Execution flow

High-level sequence for producing a figure:

1. Call `register_builtin_figures()` (once per process).
1. Select a figure by name: `spec = REGISTRY.get(name)`.
1. Validate trace type (caller responsibility today; `spec.accepts` exists for this).
1. Construct context: `ctx = FigureContext(env_idx=..., freq_idx=..., figsize=..., style=...)`.
1. Render: `fig = spec.plot(trace, ctx)`.
1. Persist/log: `save_pdf(fig, path)` and/or `log_tensorboard_figure(logger, tag, fig, step)`.

## Built-in figure inventory

Built-ins are registered in `register_builtin_figures()`:

- `overview` (accepts `RolloutTrace`): time heatmaps for g_inf, g_gen and actions.
- `overview.rate_maps` (accepts `RolloutTrace`): time heatmaps plus spatial maps.
- `environment.layout` (accepts `WorldTrace`): static environment layout.
- `walk.trajectories` (accepts `WorldTrace`): walk overlaid on environment map.
- `walk.statistics` (accepts `WorldTrace`): walk summary statistics.
- `split.statistics` (accepts `WorldTrace`): dataset split composition summary.

## Environment rendering model

The environment plotting primitives assume an environment object with:

- `n_locations: int`
- `n_actions: int`
- `locations: list[dict]` where each location dict includes:
  - `id: int`
  - `o: float` (x in [0, 1])
  - `y: float` (y in [0, 1])
  - `shiny: bool` (optional)
  - `actions: list[dict]` where each action dict includes:
    - `id: int`
    - `probability: float`
    - `transition: Sequence[float]` (non-zero indicates reachable destinations)

`plot_map()` renders location markers colored by per-location values and can overlay action arrows.

## Extension workflow: adding a new figure

### 1) Decide the domain and name

- Choose a namespaced figure name: `<domain>.<figure>`.
- Keep names stable; treat them as part of the public interface.
- Add tags that match existing conventions (`model`, `rollout`, `data`, `debug`, `statistics`, `spatial`).

### 2) Implement a figure module

Create a new module file under `src/torch_tem/figures/modules/<domain>/` and implement:

```python
def plot(trace: <TraceType>, ctx: FigureContext) -> Figure:
    ...
```

Implementation guidelines:

- Validate `ctx.env_idx` and `ctx.freq_idx` against `trace.batch_size` and the available scales.
- Handle empty traces gracefully.
- Use `ctx.style.apply_context()` when style is provided; do not permanently mutate global rcParams.
- Prefer `torch.no_grad()` and `.detach().cpu()` when converting tensors.
- Use primitives (`plot_map`, `plot_actions`, `plot_walk`) for environment geometry to keep visuals consistent.

### 3) Register the figure

Edit the registration function to add a new `FigureSpec`:

- Use `accepts=<TraceBaseSubclass>`.
- Set a clear `description`.
- Keep `register_builtin_figures()` idempotent (registry already treats duplicate names as no-ops).

### 4) Validate outputs

Minimum validation:

- The figure returns a `matplotlib.figure.Figure`.
- Saving to PDF/PNG succeeds.
- The figure works for:
  - $B=1$ and $B>1$ with different `ctx.env_idx`
  - at least one value of `ctx.freq_idx` (when multi-scale)

## Error handling strategy

Preferred failure modes:

- Out-of-range indices: raise `IndexError` with the valid range.
- Missing required trace data: raise `ValueError` describing what is missing.
- Empty trace: return a Figure with an informative title or message.

Avoid:

- Silent shape mismatches.
- Partially rendered figures without warnings.

## Persistence and logging

The sinks implement a dual-output pattern:

- PDF is the canonical, publication-quality artifact.
- PNG is used for raster previews (including TensorBoard images).

`make_figure_path()` standardizes filenames under `<base_dir>/figures/` and can include
`step` and `version` suffixes.
