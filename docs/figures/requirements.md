# Figures Subsystem — Requirements

## Goal

Provide a composable, extensible figures subsystem that:

- Keeps plotting logic pure and reusable (notebook + Lightning).
- Uses stable, discoverable figure names (registry-based).
- Builds plot inputs via traces (CPU/NumPy), extracted from streaming sources (e.g., RolloutStream).

## User Stories

- As a researcher, I want to generate standardized figures from experiment artifacts (data, model rollouts) so I can understand behavior quickly and reproducibly.
- As a developer, I want to add a new figure module by implementing a small contract and registering it, without editing central dispatch code in multiple places.
- As a training user, I want Lightning callbacks to generate selected figures by name, saving PDF artifacts and logging previews to TensorBoard, without plotting code depending on Lightning.

## Acceptance Criteria (EARS)

### Plotting Purity / Boundaries

- REQ-001: WHEN a figure is generated, THE SYSTEM SHALL require a plot-ready trace input (not a live model or DataModule) as the primary plotting input.
- REQ-002: WHEN figure code is executed, THE SYSTEM SHALL NOT depend on Lightning objects (Trainer/Logger) or filesystem paths.
- REQ-003: WHEN a plot is generated, THE SYSTEM SHALL return a matplotlib Figure and SHALL NOT call plt.show().

### Trace Invariants

- REQ-010: WHEN a trace is created for plotting, THE SYSTEM SHALL detach tensors and move data to CPU and store them as NumPy arrays (or Python scalars/structures).
- REQ-011: WHEN a trace is created from a stream, THE SYSTEM SHALL support downsampling via a stride parameter.
- REQ-012: WHEN a trace is created from a stream, THE SYSTEM SHALL support a maximum-step limit.

### Rollout Reuse

- REQ-020: WHEN extracting model-rollout plot traces, THE SYSTEM SHALL reuse torch_tem.model.RolloutStream as the underlying rollout mechanism.
- REQ-021: WHEN action/episode-boundary information is needed for plotting, THE SYSTEM SHALL expose it via an event iterator API (Option 1) without breaking existing RolloutStream iteration semantics.

### Figure Registry / Naming

- REQ-030: WHEN figures are configured by string name (e.g., via FigureCallbackSettings.figures), THE SYSTEM SHALL resolve names via a registry (not hardcoded if/else dispatch).
- REQ-031: WHEN a figure name is unknown, THE SYSTEM SHALL fail with a clear error that includes available figure names.
- REQ-032: WHEN listing figures (for help/CLI/validation), THE SYSTEM SHALL provide a deterministic list of registered figures including name and description.

### Compatibility / Migration

- REQ-040: WHEN migrating from the current TEMRolloutTrace, THE SYSTEM SHALL provide a compatible trace type that can be used by existing figures (tem_overview) with minimal refactor.
- REQ-041: WHEN migrating the callback dispatch, THE SYSTEM SHALL preserve existing behaviors: periodic generation, PDF saving, TensorBoard logging, rank-zero safety, and figure closing.

## Security / Safety

- SEC-001: IF figure generation fails during training, THEN THE SYSTEM SHALL catch exceptions and continue training (log/print error), preserving current callback fault-tolerance behavior.
- SEC-002: WHEN saving/logging figures, THE SYSTEM SHALL ensure figures are closed when configured to prevent memory leaks during long runs.

## Constraints

- CON-001: Use Matplotlib as the figure backend (current codebase usage).
- CON-002: Keep the figures package layered: primitives/style/sinks + figure modules.
- CON-003: Prefer minimal OOP: Protocols + dataclasses + registry; avoid deep inheritance trees.

## Out of Scope (for this iteration)

- Formal CLI tooling for figure generation beyond what already exists.
- Reworking all legacy plotting functions (plot_map, plot_walk, etc.) beyond what is required to integrate the new architecture.
