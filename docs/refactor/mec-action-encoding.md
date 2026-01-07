# MEC action encoding + reset boundaries

## Design decision

**Reset logic is the caller's responsibility**, not MEC's.

MECModel always applies transition dynamics from the provided state. At episode boundaries or resets, the caller (TEMModel/Rollout) must reset `state.g` to `g_init` before calling MEC.

## Why this is cleaner

1. **MECModel is simpler**: No special handling for "None/reset" cases inside transition logic
2. **Clear separation of concerns**: 
   - MEC = pure dynamics (action + state → next state)
   - Caller = boundary management (when to reset state)
3. **No ambiguous encoding**: Actions are always valid one-hot tensors; no need for `has_prev_action`/`do_step` masks
4. **Easier to reason about**: MEC behavior is deterministic from inputs (no hidden reset semantics)

## Action encoding (simplified)

With `has_static_action=True` and `n_actions = N` (excluding standstill):

| Action value | One-hot encoding | Meaning |
|---:|:---|:---|
| `0` | all-zeros `[0,0,...,0]` (length N) | Stand still (valid step, no directional drive) |
| `k` where `1..N` | one-hot at index `k-1` | Move in direction `k` |

With `has_static_action=False`:
- Actions are `0..N-1` with standard one-hot encoding

**No `None` encoding needed**: Reset boundaries are handled by the caller resetting state before calling MEC.

## Caller responsibilities

At episode boundaries or TBPTT chunk starts where `a_prev is None`:

```python
# Before calling MEC, reset state for envs with no previous action
reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool)
if torch.any(reset_mask):
    mec_state.g = [
        torch.where(reset_mask.unsqueeze(-1), mec.g_init[f].unsqueeze(0), mec_state.g[f])
        for f in range(n_f)
    ]

# Then call MEC with valid actions (use 0 for None, state is already reset)
a_onehot = one_hot_with_zero(a_prev, n_actions)
mec_state = mec(a_onehot, mec_state, locations)
```

This keeps MEC focused on dynamics while the orchestration layer handles episode structure.
