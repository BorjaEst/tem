# Design — Figures Submodule Architecture (TEM)

## Overview

This design introduces a three-layer visualization architecture:

1. **Simulation / Rollout**
   - Produces `TEMOutput`, `TEMState`, and labels over time (already available via `Rollout` in [src/torch_tem/core/model.py](src/torch_tem/core/model.py)).

2. **Diagnostics / Extraction**
   - Converts runtime objects into plot-ready data: CPU arrays and small dataclasses (no plotting).

3. **Figures**
   - Renders multi-panel figures from plot-ready data; returns `matplotlib.figure.Figure`.

4. **Sinks**
   - Save canonical PDF artifacts to disk.
   - Log raster previews (PNG) to TensorBoard.

This ensures responsibility isolation: figures “draw”, diagnostics “compute arrays”, simulation “runs the model”.

## Current State (Inventory)

- Existing plotting utilities live in [src/torch_tem/figures/**init**.py](src/torch_tem/figures/__init__.py) and mix primitives + multi-panel routines.
- Training entrypoint uses Lightning + TensorBoard via [run.py](run.py).
- Training loop is in [src/torch_tem/training.py](src/torch_tem/training.py) and currently logs scalars via `self.log(...)` but does not log figures.

## Proposed Package Layout

- `src/torch_tem/figures/`
  - `primitives.py`
    - map drawing primitives migrated from current figures code: map, actions, walk overlays.
  - `style.py`
    - centralized Matplotlib rcParams and reusable style tokens.
  - `sinks.py`
    - `save_pdf(fig, path)`
    - `save_png(fig, path)` (optional)
    - `log_tensorboard_figure(writer, tag, fig, step)` (logs preview images)
  - `registry.py` (optional)
    - maps names -> `FigureSpec` for discoverability and CLI/callback selection.
  - `tem_overview.py` (multi-panel, end-to-end TEM)
  - `hpc_memory.py` (multi-panel HPC memory visualizations)
  - `mec_diagnostics.py` (grid / uncertainty / correction dynamics)
  - `lec_features.py` (observation encoding / reconstruction views)

- `src/torch_tem/diagnostics/` (new)
  - `rollout_trace.py`
    - defines plot-ready trace dataclasses.
  - `extract_tem.py`
    - functions that produce plot-ready data from `Rollout` / `TEMOutput` / `TEMState`.
  - (optional) `sampling.py`
    - selection/downsampling utilities for large cell populations.

- `src/torch_tem/callbacks/` (new, or placed under training module)
  - `figures.py`
    - Lightning Callback that triggers extraction -> figure rendering -> sinks.

## Data Contracts (Interfaces)

### Plot-ready types (examples)

- `TEMRolloutTrace`
  - time series arrays for:
    - sensory predictions (logits/probabilities)
    - abstract/grounded codes
    - uncertainties (if available)
    - labels (locations, observation ids)
- `HPCMemorySnapshot`
  - CPU arrays for memory matrices and summary stats.
- `EnvMapData`
  - environment geometry and per-location scalars to render map overlays.

Design rule:

- Figure modules consume _plot-ready_ dataclasses or numpy arrays, not live torch tensors on GPU and not `TEMModel`.

## Figure Module Contract

Each figure module exports:

- `make_figure(data, *, style=None, figsize=None, dpi=None) -> Figure`

Constraints:

- Must not call `plt.show()`.
- Must avoid global `plt.*` where possible; use OO Matplotlib (axes methods).
- Must not perform extraction/simulation.
- Must be deterministic for identical inputs.

## Sinks

### PDF Saving

- Save PDF as canonical artifact:
  - `fig.savefig(path, format="pdf", bbox_inches="tight")`

### TensorBoard Preview Logging

- TensorBoard does not render PDFs. We log a raster preview:
  - use `SummaryWriter.add_figure(tag, fig, global_step=step)` or render to an image array and use `add_image`.
- Always close figures after logging/saving:
  - `plt.close(fig)`

### Output Paths

Preferred location:

- inside the Lightning logger directory (e.g., `logs/<name>/<version>/figures/`)
  This keeps artifacts colocated with event files and checkpoints.

## Training Integration (Lightning)

Add a `FiguresCallback` that:

- runs at configurable intervals (every N steps / per epoch / on validation end)
- gathers a small batch or a cached trace (to limit overhead)
- calls diagnostics extraction
- calls one or more figure modules
- saves PDFs + logs TensorBoard previews

Rank-zero policy:

- Use a rank-zero guard to avoid duplicates in distributed settings.

## Error Handling Matrix

- Missing Matplotlib dependency → raise ImportError with install hint.
- Writer not available (logger disabled) → skip TB logging, still save PDFs if configured.
- Invalid shapes/missing keys in plot-ready data → raise ValueError with clear field name and expected shape.
- Excessive data size (too many cells) → diagnostics layer must downsample/select.

## Testing Strategy

- Unit tests for figure modules:
  - create minimal synthetic plot-ready data
  - assert `Figure` returned and contains expected number of axes
- Unit tests for sinks:
  - saving PDF to temp path produces a non-empty file
  - TB logging functions can be invoked with a dummy writer (or guarded)
- Callback smoke test:
  - instantiate callback and ensure it does not crash when logger is present/absent.

## Decision Record (Compressed)

Decision: Use PDF as canonical artifact + TensorBoard PNG previews | Rationale: TB cannot display PDFs reliably; previews are fast to scan | Impact: dual-output pipeline (pdf + preview) | Review: after first 2 figure modules are integrated and used in a training run.
