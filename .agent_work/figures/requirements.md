# Requirements — Figures Submodule (TEM Refactor)

## Scope

Introduce a maintainable, responsibility-isolated figures subsystem for TEM (Tolman-Eichenbaum Machine) and its submodules (LEC/MEC/HPC). Figures are PDF-first artifacts, with optional TensorBoard integration via raster previews.

## User Stories

- As a researcher, I want canonical, publication-quality PDF figures to understand TEMModel behavior and compare runs.
- As a developer, I want figure generation to be modular, testable, and separated from simulation/training code.
- As a practitioner, I want to browse figure previews in TensorBoard during training without slowing training significantly.

## Definitions

- **Figure module**: one Python module that builds one multi-panel figure (multiple subplots) and returns a `matplotlib.figure.Figure`.
- **Diagnostics/extraction**: code that converts model outputs/states/rollouts into plot-ready data (CPU arrays, scalars, small dataclasses).
- **Sinks**: code that persists or logs figures (save PDF to disk, log preview images to TensorBoard).

## Requirements (EARS Notation)

### R1 — Modular Figure Modules

WHEN a developer adds a new figure, THE SYSTEM SHALL allow adding it as a single module representing one multi-plot figure with a stable entrypoint (e.g., `make_figure(...) -> Figure`).

WHEN a figure is generated, THE SYSTEM SHALL return a Matplotlib `Figure` object without calling interactive rendering (no `plt.show()`).

### R2 — Responsibility Isolation

WHEN figure code is executed, THE SYSTEM SHALL NOT run simulations, environment rollouts, training steps, or forward passes; it SHALL only render from provided plot data.

WHEN simulation results are required for a figure, THE SYSTEM SHALL provide a diagnostics/extraction layer that produces plot-ready data separately from figure rendering.

### R3 — PDF-First Output

WHEN a figure is persisted to disk, THE SYSTEM SHALL save a vector PDF as the canonical artifact.

WHEN a figure is saved, THE SYSTEM SHALL support deterministic naming that includes at least: figure name, run identifier (or logger version), and step/epoch if applicable.

### R4 — TensorBoard Integration (Preview)

WHEN TensorBoard logging is enabled, THE SYSTEM SHALL log a rasterized preview of the figure (e.g., PNG) to TensorBoard at configurable intervals.

IF TensorBoard logging is enabled, THEN THE SYSTEM SHALL NOT require TensorBoard to render PDFs; PDFs are saved to disk while TensorBoard stores previews.

### R5 — Training Integration

WHEN training runs via the Lightning entrypoint, THE SYSTEM SHALL support optional figure generation and logging via a Lightning Callback (or equivalent hook) without polluting model code.

IF distributed training is used, THEN THE SYSTEM SHALL log figures only once (rank-zero only) to avoid duplicate events.

### R6 — Compatibility / Migration

WHEN existing code imports plot utilities from the figures package, THE SYSTEM SHALL provide a clear migration path:

- either maintain lightweight wrappers for legacy `plot_*` functions
- or provide a deprecation period with documented replacements

### R7 — Styling Consistency

WHEN figures are generated, THE SYSTEM SHALL apply a consistent style (fonts, sizes, colormaps) configurable in one place.

### R8 — Performance and Safety

WHEN figures are generated during training, THE SYSTEM SHALL minimize GPU/CPU memory pressure by requiring plot data to be CPU-based and by closing figures after save/log operations.

IF invalid data shapes or missing fields are provided to a figure, THEN THE SYSTEM SHALL raise a clear, actionable error message.

## Acceptance Criteria

- A new figure can be added without editing central plotting logic beyond optional registration.
- Figures can be saved as PDF and previewed in TensorBoard (as images) in a training run launched from [run.py](run.py).
- Figures accept plot-ready data objects and do not run rollouts internally.
- Figure generation does not cause runaway memory usage in long runs (figures are closed).
- A minimal smoke test suite confirms that core figures return a `Figure` object.

## Dependencies & Constraints

- Python: uses existing stack (PyTorch Lightning, TensorBoard logger).
- Matplotlib is required for figure rendering.
- TensorBoard preview logging requires rasterization (PNG), not PDF embedding.

## Edge Cases Matrix

- Headless environment (no display): must use non-interactive Matplotlib backend.
- DDP/multi-GPU: avoid duplicated logging.
- Very large environments (many locations/cells): figures must allow downsampling or selecting subsets.
- Missing optional signals (e.g., `p_xi` can be None): figures must handle optional series gracefully.

## Confidence Score

82% — architecture and integration points are clear (Lightning + existing figures), but figure content specifics (exact diagnostics per module) will iterate as the science questions sharpen.
