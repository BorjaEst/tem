# Tasks — Figures Submodule (TEM)

## Milestones

M1: Foundations (contracts + sinks)  
M2: First canonical figures (TEM overview + one module-specific)  
M3: Lightning/TensorBoard integration (callback + config)  
M4: Migration + docs + tests

---

## Task List

### T1 — Create new module skeletons

- Outcome: New packages exist with placeholder modules:
  - `torch_tem/figures/{primitives,style,sinks,registry}.py`
  - `torch_tem/diagnostics/{rollout_trace,extract_tem}.py`
  - `torch_tem/callbacks/figures.py` (or equivalent location)
- Depends on: none

### T2 — Refactor existing plotting primitives

- Outcome: Move reusable drawing utilities out of [src/torch_tem/figures/**init**.py](src/torch_tem/figures/__init__.py) into `primitives.py`:
  - map rendering, action arrows, walk overlay, axes initialization
- Constraints: preserve behavior; avoid breaking imports until migration is complete
- Depends on: T1

### T3 — Define plot-ready dataclasses (“viz data layer”)

- Outcome: Define minimal dataclasses for:
  - rollout traces over time
  - environment map data (geometry + per-location scalars)
  - memory/cell snapshots (CPU arrays)
- Acceptance: figure modules consume these dataclasses, not `TEMModel` and not GPU tensors
- Depends on: T1

### T4 — Implement diagnostics extraction from existing runtime objects

- Outcome: Provide extraction functions that take:
  - `Rollout` (or list of `(TEMOutput, TEMLabel, TEMState)`)
  - and return plot-ready dataclasses
- Notes: must detach and move to CPU; may include downsampling helpers
- Depends on: T3

### T5 — Implement sinks (PDF save + TB preview)

- Outcome:
  - `save_pdf(fig, path)`
  - `log_tensorboard_figure(writer, tag, fig, step)` with safe guards
  - consistent path conventions (e.g. under logger dir)
- Acceptance: PDFs saved; TB previews visible in Images tab
- Depends on: T1

### T6 — Implement first “canonical” multi-panel figure modules

- Outcome:
  - `tem_overview.py`: end-to-end panels (sensory reconstructions, abstract vs grounded, uncertainty summaries)
  - `mec_diagnostics.py` or `hpc_memory.py`: one submodule-focused figure
- Acceptance: each exports `make_figure(...) -> Figure` and is testable with synthetic data
- Depends on: T2, T4

### T7 — Add Lightning callback for periodic figure generation

- Outcome: `FiguresCallback` integrated into the Trainer creation path in [run.py](run.py):
  - configurable frequency
  - uses diagnostics extraction + figure modules + sinks
  - rank-zero only behavior
- Acceptance: running training produces:
  - PDFs on disk under the run folder
  - preview images in TensorBoard
- Depends on: T5, T6

### T8 — Migration / compatibility layer

- Outcome:
  - keep a thin compatibility surface in [src/torch_tem/figures/**init**.py](src/torch_tem/figures/__init__.py) re-exporting primitives and/or providing deprecated wrappers
  - update any internal callers to the new APIs
- Acceptance: no import breakage for existing code; deprecation warnings are clear
- Depends on: T2, T6

### T9 — Documentation and examples

- Outcome:
  - short usage guide: how to generate PDFs offline and how to view TB previews
  - example script in `examples/` for offline figure generation from a checkpoint
- Depends on: T6, T7

### T10 — Tests

- Outcome:
  - smoke tests for: `make_figure` returns `Figure`, PDF save works, callback doesn’t crash when logger is absent
- Depends on: T5, T6, T7

---

## Execution Notes

- Prefer non-interactive Matplotlib backend for CI/headless.
- Ensure figures are closed after save/log to avoid memory leaks.
- Keep figure modules “render only”; do not run rollouts inside them.
