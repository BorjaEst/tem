## TEM Figures Submodule — Implementation Tasks

This file is the execution plan for implementing `RolloutTrace` (Option C) and
rate-map overview figures while preserving the existing registry-driven
infrastructure.

### Scope

- Implement `RolloutTrace` (combined model + world + per-step location IDs).
- Add collector utilities for offline usage.
- Update the Lightning callback dispatch to support `RolloutTrace`.
- Add a new overview-style figure that renders square-grid rate maps.
- Keep the existing `overview` figure working for `ModelTrace` during the
  transition.

### Milestones (recommended order)

1. Trace foundation: `RolloutTrace` + unit tests.
2. Trace construction: offline collector + callback integration.
3. Figure: spatial rate-map overview figure.
4. Registry + callback dispatch updated to `isinstance`.
5. Validation: offline example + smoke tests.
6. Legacy cleanup (only after usage is eliminated).

## 1. Baseline Verification

- [ ] **T-BASE-001**: Run the offline example once to ensure baseline is green.
  - Command: `python examples/data_datamodule.py`.
  - Expected: generates `environment.layout` and `walk.trajectories` without exceptions.
  - Traceability: AC-001.

- [ ] **T-BASE-002**: Confirm current callback generates `overview` and at least one data figure.
  - File: [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py).
  - Traceability: AC-002.

## 2. Implement `RolloutTrace`

- [ ] **T-RT-001**: Add `RolloutTrace` implementing `PlotTrace`.
  - Add file: `src/torch_tem/figures/core/rollout_trace.py`.
  - Required fields:
    - `worlds: list[World]`
    - `walks: list[list[step]]`
    - `location_ids: list[list[int]]` (derived from `walks[e][t][0]["id"]`)
    - `model: ModelTrace`
    - `meta: dict[str, Any]`
  - Required behavior:
    - `batch_size`, `n_steps`, `select_env(env_idx)`, `downsample_time(stride)`.
    - Strict invariants and alignment checks (raise `ValueError` on mismatch).
  - Traceability: REQ-TRC-003, REQ-TRC-004.

- [ ] **T-RT-002**: Export `RolloutTrace` from `torch_tem.figures.core`.
  - Update: [src/torch_tem/figures/core/**init**.py](src/torch_tem/figures/core/__init__.py).
  - Traceability: REQ-TRC-003.

## 3. Collectors (Offline + Shared)

- [ ] **T-COL-001**: Add `collect_rollout_trace(...) -> RolloutTrace`.
  - New module (suggested): `src/torch_tem/figures/core/collectors.py`.
  - Responsibilities:
    1. Sample a batch walk from a datamodule split.
    2. Run `RolloutStream(model, walk, initial=None)`.
    3. Build `ModelTrace.from_rollout(...)` with the same truncation/downsampling.
    4. Derive `location_ids` from the sampled walk.
    5. Return `RolloutTrace` with aligned lengths.
  - Traceability: REQ-TRC-003, REQ-TRC-004.

- [ ] **T-COL-002**: Fix or clarify `collect_data_trace(...)` behavior.
  - File: [src/torch_tem/figures/core/data_trace.py](src/torch_tem/figures/core/data_trace.py).
  - Goal: Ensure the function returns a trace derived from the sampled batch
    (not a dataset snapshot), or rename/document accordingly.
  - Traceability: AC-001 (offline stability), REQ-TRC-002.

## 4. Callback Integration (Online)

- [ ] **T-CB-001**: Update callback dispatch to use `isinstance(trace, spec.accepts)`.
  - File: [src/torch_tem/callbacks/figures.py](src/torch_tem/callbacks/figures.py).
  - Build (when possible): `ModelTrace`, `DataTrace`, `RolloutTrace`.
  - Dispatch rule:
    - Choose the first available trace matching `spec.accepts` via `isinstance`.
    - Otherwise warn and skip (training behavior).
  - Traceability: REQ-EXE-002, design “Implementation gap” section.

- [ ] **T-CB-002**: Construct `RolloutTrace` in the callback.
  - Inputs already available:
    - `walk_limited` (batch walk)
    - `ModelTrace` from `RolloutStream`
    - `datamodule.dataset.environments` (world geometry)
  - Output:
    - `location_ids` derived from `walk_limited` and aligned to model timesteps.
  - Traceability: REQ-TRC-003, REQ-TRC-004, AC-002.

## 5. Registry Updates

- [ ] **T-REG-001**: Register the combined-trace overview rate-map figure.
  - File: [src/torch_tem/figures/register.py](src/torch_tem/figures/register.py).
  - Recommendation:
    - Keep existing `overview` accepting `ModelTrace` (transition fallback).
    - Add a new entry: `overview.rate_maps` accepting `RolloutTrace`.
  - Traceability: REQ-REG-001, REQ-OVR-003.

## 6. Implement the Spatial Overview Figure

- [ ] **T-OVR-001**: Implement rate-map aggregation helper.
  - Policy:
    - Mean activity per visited location.
    - Unvisited locations must remain missing/blank (use `NaN`).
  - Traceability: REQ-OVR-002.

- [ ] **T-OVR-002**: Add a new figure module that renders square-grid rate maps.
  - New file (suggested): `src/torch_tem/figures/modules/overview_rate_maps.py`.
  - Input: `RolloutTrace`, `FigureContext`.
  - Layout (preferred): 2×2.
    - top-left: `g_inf` time heatmap (selected env, selected freq)
    - top-right: `g_gen` time heatmap
    - bottom-left: square-grid rate maps for a small set of cells
    - bottom-right: optional occupancy map or additional rate maps
  - Rendering:
    - Use `plot_map(world, values, shape="square")`.
  - Traceability: REQ-OVR-001, REQ-OVR-002, AC-003.

- [ ] **T-OVR-003**: Keep existing `overview` behavior as a `ModelTrace` fallback.
  - File: [src/torch_tem/figures/modules/overview.py](src/torch_tem/figures/modules/overview.py).
  - Do not break current usage; only add the new figure via registry.
  - Traceability: REQ-OVR-003.

## 7. Tests

- [ ] **T-TEST-001**: Unit tests for `RolloutTrace` invariants and alignment.
  - New file (suggested): `tests/test_rollout_trace.py`.
  - Must cover:
    - `select_env` keeps alignment
    - `downsample_time` keeps consistent indices across walk, location_ids, and model
    - mismatch raises `ValueError` with useful message
  - Traceability: REQ-TRC-004.

- [ ] **T-TEST-002**: Unit test for rate-map aggregation.
  - Given a tiny walk visiting known locations and known activity values,
    assert computed per-location means and `NaN` for unvisited.
  - Traceability: REQ-OVR-002.

## 8. Validation / Smoke

- [ ] **T-VAL-001**: Offline smoke: `examples/data_datamodule.py` still runs.
  - Traceability: AC-001.

- [ ] **T-VAL-002**: Online smoke: callback can generate:
  - `overview` (ModelTrace)
  - at least one data figure (DataTrace)
  - `overview.rate_maps` (RolloutTrace)
  - Traceability: AC-002, AC-003.

## 9. Legacy Removal (Only After No Usage Remains)

### 9.1 Deprecation gate

- [ ] **T-LEG-001**: Search repo for internal imports/usages of legacy plotting wrappers.
  - Target legacy files:
    - [plot.py](plot.py) (root-level legacy wrappers)
    - [legacy.py](legacy.py) (legacy model/analysis)
  - Gate: proceed only when internal references are removed or intentionally kept.

### 9.2 Remove legacy plotting wrappers

- [ ] **T-LEG-002**: Migrate any internal imports from [plot.py](plot.py) to:
  - [src/torch_tem/figures/primitives.py](src/torch_tem/figures/primitives.py) and/or
  - figure modules under `src/torch_tem/figures/modules/`.

- [ ] **T-LEG-003**: Add a deprecation note to [plot.py](plot.py) (one release/window).
  - Gate: only after internal usage is eliminated.

- [ ] **T-LEG-004**: Delete [plot.py](plot.py) _only if_:
  - repo-wide search shows no references, and
  - you accept breaking any external scripts relying on it.

### 9.3 Remove legacy model code

- [ ] **T-LEG-005**: Remove or archive [legacy.py](legacy.py) only if:
  - repo-wide search shows no imports/usages, and
  - you have no requirement to keep it as historical reference.

## 10. Done Criteria

- [ ] All tasks up to **T-VAL-002** complete.
- [ ] `overview.rate_maps` renders at least one square-grid rate map.
- [ ] Existing `overview` continues to work with `ModelTrace`.
- [ ] Legacy removal tasks are gated and only executed after usage is eliminated.
