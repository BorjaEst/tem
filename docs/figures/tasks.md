# Figures Subsystem — Implementation Tasks

## Milestone 1: Generic tracing foundation (RolloutEvent + collect_trace)

- [ ] TASK-001 Create RolloutEvent dataclass
  - Location: new module (recommended) src/torch_tem/diagnostics/events.py
  - Fields: t, locations, observation, action, a_prev, output, label, state
  - Outcome: a stable event type for extractors.

- [ ] TASK-002 Add RolloutStream.iter_events() (Option 1)
  - Location: src/torch_tem/model.py
  - Requirements:
    - Must not change existing **next** output type/behavior.
    - Must yield RolloutEvent for each timestep with correct t, action, a_prev.
  - Validation:
    - New unit test compares iter_events() output count to base iteration count on a small walk.

- [ ] TASK-003 Add generic tracing Protocols + collect_trace
  - Location: src/torch_tem/figures/core/types.py and src/torch_tem/figures/core/collect.py
  - Include:
    - PlotTrace Protocol
    - TraceExtractor[EventT, TraceT] Protocol
    - collect_trace(stream, extractor, max_steps=None, downsample_stride=1)
  - Validation:
    - Unit tests for stride + max_steps semantics.

## Milestone 2: Replace special-case extraction with extractor class

- [ ] TASK-004 Implement ModelRolloutTrace concrete trace (CPU/NumPy)
  - Location: src/torch_tem/diagnostics/traces/model_rollout.py (or keep in rollout_trace.py initially)
  - Must implement PlotTrace capabilities: select_env, downsample_time, batch_size, n_steps, meta.

- [ ] TASK-005 Implement ModelRolloutTraceExtractor
  - Location: src/torch_tem/diagnostics/extractors/model_rollout.py
  - Input: RolloutEvent
  - Output: ModelRolloutTrace
  - Must:
    - Fill actions from event.action (no placeholder).
    - Optionally compute episode_boundary from event.a_prev.
    - Detach/move to CPU, store NumPy arrays.

- [ ] TASK-006 Provide compatibility wrapper extract_rollout_trace(...)
  - Location: src/torch_tem/diagnostics/extract_tem.py
  - Implementation: construct extractor + call collect_trace.
  - Goal: keep existing imports working while migrating callers.

## Milestone 3: Registry + integrate existing figure + callback dispatch

- [ ] TASK-007 Implement registry types and API
  - Location: src/torch_tem/figures/core/registry.py
  - Provide: FigureSpec, FigureRegistry, global REGISTRY
  - Provide: validate(names) for settings and callback.

- [ ] TASK-008 Register existing figure module tem_overview as "tem.overview"
  - Location: src/torch_tem/figures/core/registry.py (explicit import registration)
  - Update tem_overview signature if needed to accept FigureContext and/or style consistently.

- [ ] TASK-009 Update callback to use registry lookup
  - Location: src/torch_tem/callbacks/figures.py
  - Replace \_make_figure hardcoded if/else with:
    - spec = REGISTRY.get(name)
    - type check isinstance(trace, spec.accepts) (or predicate)
    - fig = spec.plot(trace, ctx)
  - Keep current behavior:
    - rank zero only
    - save pdf via sinks
    - log tensorboard preview
    - exception handling

## Milestone 4: Data figures foundation (environment/walk/split)

- [ ] TASK-010 Define DataRolloutTrace + extractor(s)
  - Location: src/torch_tem/diagnostics/traces/data_rollout.py
  - Inputs: dataset state or batch chunk + visited
  - Output: CPU plot-ready structures and derived stats.

- [ ] TASK-011 Create figure modules:
  - src/torch_tem/figures/environment/layout.py registered as "environment.layout"
  - src/torch_tem/figures/walk/trajectories.py registered as "walk.trajectories"
  - src/torch_tem/figures/walk/statistics.py registered as "walk.statistics"
  - src/torch_tem/figures/split/statistics.py registered as "split.statistics"
  - Each exports plot(trace, ctx) -> Figure.

## Milestone 5: Validation + docs

- [ ] TASK-012 Add tests
  - Registry tests
  - collect_trace tests
  - RolloutEvent correctness tests
  - Smoke test: generate "tem.overview" from a small trace

- [ ] TASK-013 Update docs / README as needed
  - Add section: How to add a figure
  - Add section: How to generate figures in Lightning vs notebooks

## Definition of Done

- Registry-based figure selection works in Lightning callback via stable names.
- RolloutStream.iter_events() exists and trace extraction records actions correctly.
- Traces are CPU/NumPy and implement PlotTrace operations.
- Existing tem_overview still works and is registered as "tem.overview".
- Unit tests cover core invariants (stride, max_steps, registry validation).
