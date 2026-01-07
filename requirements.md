# Requirements — MEC action encoding + reset semantics

## Context

We are refactoring TEM/MEC integration to:

- Use tensor one-hot action encoding for MEC transitions.
- Preserve legacy behavior for reset boundaries (legacy represents reset as `a_prev is None`).
- Keep the model compatible and testable during migration.

## User stories

- As a developer, I want `MECModel` to accept actions as a one-hot tensor so its logic is tensor-first and easier to reason about.
- As a developer, I want reset boundaries to be represented unambiguously in batched training.
- As a developer, I want behavior to remain equivalent to legacy TEM semantics (up to the chosen time indexing convention).

## Acceptance criteria (EARS)

- WHEN a batch element corresponds to a reset boundary (legacy `a_prev is None`), THE SYSTEM SHALL prevent a transition step from propagating state from the previous walk.
- WHEN `has_static_action` is enabled and the previous action is “stand still”, THE SYSTEM SHALL encode the action as an all-zeros one-hot vector.
- WHEN `has_static_action` is enabled, THE SYSTEM SHALL distinguish “stand still” from “reset boundary” even though both can map to all-zeros one-hot.
- WHEN training or evaluating with multiple environments in the same batch, THE SYSTEM SHALL support per-environment reset boundaries.
- WHEN migrating MEC logic, THE SYSTEM SHALL preserve API compatibility for the current TEM forward pass until the refactor is validated.

## Non-goals

- Redesign the full model to a different time indexing convention in this iteration.
- Change the environment/data generator semantics.
