# Figures module design

This document describes the design of the `torch_tem.figures` submodule.

The figures system is intentionally split into:

- **Pure figure construction**: Convert a `TraceTree` + `FigureContext` into a
  matplotlib `Figure`.
- **Persistence and logging**: Save artifacts (PDF/PNG) and log preview images
  to TensorBoard.

This separation keeps plotting code testable and reusable in scripts, notebooks,
and training callbacks.

## Goals

- Provide a stable, name-based interface for selecting figures via a registry.
- Make figure generation configuration-driven (names + tags).
- Support long-horizon figures by stitching consecutive rollout batches into a
  single continuous episode trace.
- Produce high-quality, versionable artifacts (PDF) and lightweight previews
  (TensorBoard images).
- Avoid training interruptions and memory leaks during long runs.

## Non-goals

- The figures submodule does not define how traces are collected; it consumes
  `TraceTree` produced by `torch_tem.diagnostics`.
- The figures submodule does not own training schedule decisions (when to
  generate figures); this is handled by the Lightning callback.
- The figures submodule does not define environment generation or rollouts.

## Architecture overview

### Components

1. **Figure registry** (`torch_tem.figures.registry`)
   - `FigureContext`: Small container for runtime plotting context (env index,
     frequency index, global step, split name, optional style).
   - `FigureSpec`: `(name, description, plot, default_filename, tags)`.
   - `FigureRegistry`: Stores and validates specs.
   - `REGISTRY`: Global registry instance.

2. **Registration entrypoint** (`torch_tem.figures.register`)
   - `register_builtin_figures()` registers built-in figure specs.
   - Kept separate to avoid circular imports (registry imports nothing from
     modules).

3. **Figure modules** (`torch_tem.figures.modules.*`)
   - Implement `plot(trace, ctx) -> Figure`.
   - Use `torch_tem.diagnostics.trace_access` helpers to read trace fields.
   - Compose reusable panel-level plotting functions.

4. **Reusable plotting utilities**
   - `torch_tem.figures.primitives`: Environment-centric low-level drawings
     (`plot_map`, walk/actions rendering, axis initialization).
   - `torch_tem.figures.plots.*`: Reusable panels (trajectory, autocorr, insets).
   - `torch_tem.figures.utils.*`: Numerical helpers for aggregation and spatial
     transforms (rate-map aggregation, autocorr utilities).

5. **Sinks** (`torch_tem.figures.sinks`)
   - `save_pdf(fig, path)` / `save_png(fig, path)`.
   - `log_tensorboard_figure(logger, tag, fig, global_step)` rasterizes and logs
     to TensorBoard.
   - `make_figure_path(base_dir, figure_name, step?, version?, extension)`.

6. **Training integration** (`torch_tem.callbacks.figures`)
   - `FiguresCallback` captures batches during validation/test, builds a
     `TraceTree`, dispatches figure specs, and persists via sinks.

### Data flow

High-level figure generation pipeline:

```text
Lightning validation/test loop
  -> dataloader yields (walk_chunk, visited)
  -> FiguresCallback snapshots batch (protect against visited mutation)
  -> (optional) stitch consecutive batches into episode batch
  -> collect_rollout_trace_tree(batch, environments, model)
  -> (optional) downsample_trace(trace, stride)
  -> spec.plot(trace, FigureContext)
  -> sinks.save_pdf / sinks.log_tensorboard_figure
  -> close Figure
```

Key design choice: Figures operate on a **TraceTree**, not on raw model tensors.
This enforces stable, inspectable paths and simplifies multi-panel plots.

## Namespacing and discoverability

Registry names are stable identifiers used in configuration and outputs.

- Flat names: `overview`
- Namespaced names: `spatial.structure`

The dotted naming convention maps naturally to a module folder structure:

- `torch_tem.figures.modules.spatial.structure.plot` corresponds to
  `spatial.structure`.

Tags are used to group figures for policy decisions (e.g. “aggregate” vs
“episode” sampling in the callback).

## Built-in figures

Built-in figures are registered in `register_builtin_figures()`.

Current built-ins:

- `overview`: Multi-panel TEM circuit overview across LEC/MEC/HPC, plus
  observation ids and trajectory.
- `spatial.structure`: Occupancy, rate maps, spatial autocorrelograms, and
  trajectory with coverage inset.

## Trace contracts

Figure modules depend on specific `TraceTree` paths.
To avoid hard-coding string paths in many places, modules use
`torch_tem.diagnostics.trace_access` helpers.

Commonly used trace fields:

- `world_step/location_ids`: `(T, B)` integer location ids
- `world_step/action_ids`: `(T, B)` integer action ids
- `world_step/observation`: `(T, B, n_o)` observation vectors
- `state/lec/cells/<freq>/value`: `(T, B, C)` activity
- `output/inference/g_inf/<freq>/value`: `(T, B, C)` inferred abstract location
- `output/generative/g_gen/<freq>/value`: `(T, B, C)` generated abstract location
- Root metadata:
  - `environments`: list of environment/world objects aligned to batch
  - `visited`: visited masks aligned to environments

Design principle:

- Figure modules SHOULD tolerate missing optional fields and render “missing”
  panels where appropriate.
- For required fields, modules MAY fail fast with actionable errors.

## Styling strategy

`torch_tem.figures.style.StyleConfig` centralizes matplotlib style choices.

Two application modes are supported:

- Global: `StyleConfig.apply()` (modifies `mpl.rcParams`).
- Scoped: `StyleConfig.apply_context()` (temporary rcParams overrides).

Figure modules may accept `ctx.style` and use a context manager to keep styling
consistent without permanently modifying global state.

## Persistence and logging strategy

The system uses a dual-output pattern:

- **PDF**: canonical, high-quality artifact suitable for papers and reports.
- **TensorBoard image**: raster preview for fast training monitoring.

File naming:

- Deterministic and step-aware: `<base_dir>/figures/<name>_step<k>.pdf`.

## Error handling

- Registry validation fails early to prevent silent misconfiguration.
- The training callback catches and reports exceptions during plotting so figure
  failures do not crash training.
- The callback closes figures after saving/logging to prevent memory growth.

## Extending the figures system

To add a new figure:

1. Implement `plot(trace: TraceTree, ctx: FigureContext) -> Figure` under
   `src/torch_tem/figures/modules/...`.
2. Register it in `torch_tem.figures.register.register_builtin_figures()` with a
   stable name, description, and tags.
3. If it requires longer temporal context, assign a tag used by the callback’s
   sampling policy (e.g. `coverage`).
4. Ensure it renders a valid “no data” figure when `trace.length == 0`.
