# Design — MEC action encoding + `do_step` reset mask

## Overview

Legacy TEM uses an **incoming-action** convention:

- At step `t`, the transition prior is computed from the action that brought the agent into state `t`.
- At the first state of a new walk there is no incoming action, and legacy represents this as `a_prev is None`.

The key migration goal is to move MEC transition computation to a tensor-first API while preserving legacy semantics.

## Key design decision

### Decision

Represent actions for MEC as:

- `a_onehot: Tensor[B, n_a]`
- `do_step: Tensor[B]` (boolean)

### Why `do_step` is required

With `has_static_action=True`, both of these cases map to an all-zeros one-hot row:

- Stand still action (`a_prev == 0`) -> valid step, no directional drive
- Reset boundary (`a_prev is None`) -> must reset to priors, do not propagate previous state

Therefore, `a_onehot` alone is ambiguous in mixed batches; `do_step` provides the missing information.

## Semantics table

Assume `has_static_action=True` and `n_actions = N` (excluding standstill).

| legacy `a_prev` | `do_step` | `a_onehot` row   | Meaning                   | MEC transition behavior                                                    |
| --------------: | :-------: | :--------------- | :------------------------ | :------------------------------------------------------------------------- |
|          `None` |   False   | all zeros        | reset boundary / new walk | do not step from previous `g`; use priors (`g_init`, `exp(logsig_g_init)`) |
|             `0` |   True    | all zeros        | stand still               | valid step; step from previous `g` with zero action drive                  |
|   `k in [1..N]` |   True    | one-hot at `k-1` | move                      | valid step; step from previous `g` driven by action                        |

For `has_static_action=False`, `a_onehot` uses standard `one_hot(k)` and `do_step` only indicates reset (`None`).

## Proposed API (current migration)

- `TEMModel` performs conversion from legacy-style `a_prev` (list[int|None]) to `(a_onehot, do_step)`.
- `MECModel.forward(a_onehot, do_step, state, locations)` computes transition prior.

## Compatibility constraints

- Keep `TEMModel` and `MECModel` consistent during migration.
- Preserve TBPTT correctness: reset boundaries must not leak state across episodes.

## Alternative design (not implemented)

A full redesign could use an **outgoing-action** convention:

- At step `t`, infer `g_t` from `x_t`, then apply `a_t` to predict a prior for `g_{t+1}`.

This can remove the notion of “missing incoming action”, but still requires an explicit per-env reset signal for batched training and changes how losses are indexed (time shift).
