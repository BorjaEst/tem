# Tasks — MEC action encoding refactor

## Goal

Make MEC transitions tensor-first and unambiguous in batched training while preserving legacy behavior.

## Plan

1. Document legacy semantics

   - Confirm where legacy uses `a_prev is None` and what it triggers.
   - Record mapping for `has_static_action`.

2. Implement/verify encoding helper

   - Ensure `one_hot_with_zero` produces all-zeros for action 0 and one-hot for actions 1..N.
   - Ensure dtype/device correctness.

3. Update TEM → MEC integration

   - Compute `do_step = (a_prev is not None)` per env.
   - Pass `a_onehot` and `do_step` into `MECModel`.

4. Update MEC API

   - `MECModel.forward(a_onehot, do_step, state, locations)`.
   - Ensure reset path uses priors when `do_step=False`.

5. Validate equivalence
   - Add a small equivalence check (optional) comparing legacy transition math vs new MEC for controlled inputs.
   - Run a minimal training/inference sanity run.

## Done / Notes

- This refactor does not remove the need for a per-env boundary signal; it makes it explicit (`do_step`).
