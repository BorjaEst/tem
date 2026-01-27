# Data module design

This document describes the design of the TEM data generation stack:

- `torch_tem.data.datamodule.DataModule` (Lightning integration)
- `torch_tem.data.datamodule.TEMDataset` (stateful iterable dataset)

The key design tension is that training requires streaming, chunked rollouts
for truncated BPTT, while diagnostics and figure generation often require
**continuous episodes** longer than a single `n_rollout` chunk.

This design adopts the simplest robust protocol for evaluation:

- Validation/test are **deterministic, finite streams**.
- Evaluation streams are reset **once per epoch**.
- **No mid-stream world swaps** are permitted for seeded validation/test.
- Continuous episodes for diagnostics/figures are obtained by \*\*stitching

  consecutive chunks\*\* from the evaluation stream.

## Goals

- Provide an infinite training stream of fixed-size rollout chunks.
- Provide deterministic validation/test streams for comparability.
- Keep chunking (`n_rollout`) as an implementation detail for transport and

  compute, not a limitation on analysis.

- Support downstream trace/figure use cases that require longer contiguous

  rollouts than a single batch.

## Non-goals

- The data module does not define model state reset policies.
- The data module does not own plotting or trace semantics.
- The data module does not provide a general-purpose RL environment API.

## Architecture overview

### Components

1. **DataConfig**
   - Composes environment settings, rollout chunking, evaluation protocol,
     sampling policy, and walk curriculum.

1. **DataModule**
   - Owns per-split `TEMDataset` instances.
   - Provides Lightning `train_dataloader`, `val_dataloader`, `test_dataloader`.
   - Provides convenience controls like `reset_split(split)`.

1. **TEMDataset**
   - Is stateful and maintains per-environment buffers:
     - `environments`: list of `World` objects for each env slot.
     - `walks`: per-env remaining steps (a long walk consumed over time).
     - `visited`: per-env visited location masks.
   - Yields batches as `(chunk, visited)` where `chunk` has exactly `n_rollout`
     timesteps.

### Data model: chunk structure

Conceptual shape:

```text
batch = (chunk, visited)

chunk[t] = (locations[t], observations[t], actions[t])  for t in [0..n_rollout)
observations[t] has leading batch dimension B
visited is aligned to the current environments/worlds
```

The dataset consumes a longer per-env walk buffer and emits it in fixed-size
segments of length `n_rollout`.

## Deterministic evaluation stream protocol

### Stream reset semantics

Validation/test are evaluated as **non-independent episodes**:

- The system resets the evaluation stream at the start of a validation/test

  epoch (or whenever `reset_split(split)` is called).

- The system does not reset between chunks inside the same epoch.

This supports the intended behavior that the agent can accumulate knowledge
(e.g., a map) over the validation stream before figures/traces are produced.

### No mid-stream world swaps

World swaps mid-stream are not desired for seeded validation/test because they
introduce hidden boundaries that invalidate “continuous episode” analyses.

Design rule:

- For seeded validation/test datasets, the initial per-env walk MUST be long

  enough to cover the entire finite stream without depletion.

Minimum required walk length per env slot:

$$
L_\min = max\_batches \cdot n\_rollout
$$

If this cannot be satisfied, the correct behavior is to fail fast with an
actionable error, not to regenerate a new `World` or walk mid-stream.

Implementation guidance:

- Prefer computing `L_min` from evaluation settings and generating walks of

  length at least `L_min` when `seed is not None`.

- If walk curriculum logic is retained, it must still respect the invariant

  `len(walk) >= L_min` for seeded evaluation splits.

## Continuous rollouts for diagnostics and figures

### Problem: chunked transport vs continuous episodes

Training/evaluation batches are limited to `n_rollout` steps by design.
However, many figures and trace-based diagnostics require longer temporal
structure.

### Solution: stitch consecutive chunks from the same stream

Because the dataset emits consecutive segments from an underlying longer walk,
clients can obtain a continuous episode by concatenating chunks in order:

```text
episode_walk = chunk0 + chunk1 + ... + chunkK
```

Then run a single rollout stream over `episode_walk` to ensure model state is
carried across the entire stitched episode.

This relies on the evaluation protocol invariant:

- No world swaps occur while consuming the chunks being stitched.

### Why this design is preferred

- Avoids changing the dataset interface for common analysis tasks.
- Avoids introducing additional per-step reset signals when the evaluation

  stream is guaranteed to be uninterrupted.

- Keeps evaluation determinism strong and easy to reason about.

## Failure modes and mitigations

### Walk depletion during evaluation

Failure mode:

- The evaluation stream consumes more steps than were generated initially.

Mitigation:

- Generate walks of length `>= L_min` for seeded evaluation.
- Raise a configuration error if depletion is detected.

### Mutable visited masks in captured batches

Failure mode:

- A callback captures `(chunk, visited)` by reference; later mutations change

  what is used for trace/figure generation.

Mitigation:

- Callers that store batches for later processing should snapshot `visited`.

## Open questions (explicitly deferred)

- Whether to add a general `reset_mask[t,b]` signal to support mid-chunk resets

  or true episodic tasks. This is not required under the “no world swaps"
  protocol, but is a future extensibility path.
