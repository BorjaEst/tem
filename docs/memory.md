---
post_title: TEM Memory API
author1: Borja Est
post_slug: tem-memory-api
microsoft_alias: na
featured_image: https://placehold.co/1200x630?text=TEM+Memory+API
categories: [AI-ML, Documentation]
tags: [PyTorch, TEM, Memory, Hebbian, Attractor]
ai_note: Drafted with AI assistance and reviewed by a human.
summary: API reference for torch_tem.memory — MemoryStorage (Hebbian associative memory) and AttractorDynamics (iterative retrieval with hierarchical early-stopping), including contracts, shapes, and usage.
post_date: 2025-11-02
---

## Overview

This document describes the public API of the memory submodule in `torch_tem`.
The memory system provides two core components:

- Hebbian associative storage of grounded locations (place cells) via `MemoryStorage`
- Iterative content-addressable retrieval via `AttractorDynamics` with hierarchical masks

Exports from `torch_tem.memory.__init__`:

- `MemoryStorage`
- `AttractorDynamics`

Related design context: see `design.md` (Memory section) for equations, shapes, and
how memory interacts with inference and generation pipelines.

## Components

### MemoryStorage

Manages Hebbian memory matrices that associate grounded locations `p` across time.
Supports either a single shared memory bank or dual banks for inference vs generation.

- Module: `torch_tem.memory.storage`
- Class: `MemoryStorage`

#### Construction

Inputs:

- `params: MemoryStorageParams`
  - Protocol dependencies (satisfied by `Parameters`):
    - `n_p_calculated: List[int]` — place cell dims per frequency module
    - `p_update_mask_calculated: Tensor[sum(n_p), sum(n_p)]` — hierarchical update mask
    - `use_p_inf: bool` — whether inference path exists (controls dual memory)
    - `common_memory: bool` — when false and `use_p_inf` true, enable dual memory

Example:

```python
from torch_tem.config import Parameters
from torch_tem.memory import MemoryStorage

params = Parameters(n_g_subsampled=[10,10,8,6,6], n_x_c=10, use_p_inf=True, common_memory=False)
storage = MemoryStorage(params)
```

#### Attributes

- `n_p: List[int]` — place cell dims per frequency
- `p_update_mask: Tensor[sum(n_p), sum(n_p)]` — hierarchical low→high mask
- `use_dual_memory: bool` — true iff `use_p_inf and not common_memory`
- `M_gen: Tensor[sum(n_p), sum(n_p)]` — generative memory matrix
- `M_inf: Optional[Tensor]` — inference memory matrix when `use_dual_memory`

#### Methods

- `update(p_inferred: Tensor, p_generated: Tensor, eta: float, lamb: float) -> None`

  - Shapes: `p_inferred, p_generated ∈ R^{B × sum(n_p)}`
  - Update rule (averaged over batch):
    - `M_new = λ · M_old + η · mean_b( p_inf ⊗ p_gen ) ⊙ mask`
  - Notes: moves `mask` to data device; updates both banks if dual memory.

- `get_memory(for_inference: bool = False) -> Tensor`

  - Returns `M_inf` when `for_inference=True` and dual memory is enabled; otherwise `M_gen`.

- `get_all_memories() -> List[Tensor]`

  - Returns a list containing one or two matrices for checkpointing.

- `set_memories(memories: List[Tensor]) -> None`
  - Restores matrices from a list produced by `get_all_memories()`.

#### Shapes and equations

- Per-frequency dims: `n_p[f] = n_g_subsampled[f] · n_x_c`
- Total size: `sum(n_p)`
- Hebbian update (conceptual):
  - `M ← λ · M + η · (p_inf ⊗ p_gen)` then `M ← M ⊙ p_update_mask`

#### Edge cases and recommendations

- If `common_memory=True` or `use_p_inf=False`, a single bank (`M_gen`) is used; `for_inference`
  is ignored in `get_memory`.
- Ensure `p_inferred` and `p_generated` use the same batch size and total dimension;
  the update computes a batched outer product followed by mean over `B`.
- Persist and restore with `get_all_memories`/`set_memories` for checkpointing.

### AttractorDynamics

Iterative content-addressable retrieval with coarse-to-fine early stopping across
frequency modules.

- Module: `torch_tem.memory.attractor`
- Class: `AttractorDynamics`

#### Construction

Inputs:

- `params: AttractorParams`
  - Protocol dependencies (satisfied by `Parameters`):
    - `kappa: float` — decay/stability term, `0 < κ < 1`
    - `i_attractor_calculated: int` — number of retrieval iterations
    - `p_retrieve_mask_inf_calculated: List[Tensor]` — per-iteration masks (inference)
    - `p_retrieve_mask_gen_calculated: List[Tensor]` — per-iteration masks (generation)

Example:

```python
from torch_tem.config import Parameters
from torch_tem.memory import AttractorDynamics

params = Parameters(n_g_subsampled=[10,8,6], n_x_c=5, kappa=0.8, i_attractor=3)
attractor = AttractorDynamics(params)
```

#### Attributes

- `kappa: float`
- `i_attractor: int`
- `p_retrieve_mask_inf: List[Tensor]`
- `p_retrieve_mask_gen: List[Tensor]`

#### Methods

- `retrieve(p_query: Tensor, M: Tensor, for_inference: bool = False) -> Tensor`
  - Inputs:
    - `p_query ∈ R^{B × sum(n_p)}` — initial grounded location query
    - `M ∈ R^{sum(n_p) × sum(n_p)}` — Hebbian memory matrix
    - `for_inference` — choose mask schedule (inference vs generation)
  - Iterative update for `t = 0 .. i_attractor-1`:
    - `p_update = (p @ M)`
    - `p_update = p_update ⊙ mask[t]`
    - `p ← κ · p + p_update`
  - Returns: `p_retrieved ∈ R^{B × sum(n_p)}`

#### Shapes and behavior

- Early iterations only update low-frequency (coarse) components; later iterations
  progressively enable higher frequencies, improving stability.
- `M` is typically symmetric or near-symmetric for stable dynamics (implementation may
  symmetrize in examples).

#### Edge cases and recommendations

- Move `M`/masks to the same device as `p_query` before multiplication to avoid device
  mismatch; the implementation handles this for masks.
- Choose `i_attractor` to at least cover the number of frequency modules; using more
  iterations can refine high-frequency details.
- With poorly scaled `M` (too large), dynamics can diverge; reduce `η` during storage
  updates or increase `κ`.

## Interactions and dataflow

- In inference pipelines, sensory features are projected to `p`-space; retrieval (`retrieve`
  with `for_inference=True`) refines that estimate using memory.
- In generative pipelines, abstract transitions produce `g` which maps to an initial `p`
  query; retrieval (`for_inference=False`) stabilizes the prediction before decoding to `x`.
- Memory updates use `p_inferred` (from inference) and `p_generated` (from transitions) in
  the Hebbian rule.

## Minimal usage example

```python
import torch
from torch_tem.config import Parameters
from torch_tem.memory import MemoryStorage, AttractorDynamics

# Configure a small 3-frequency setup
params = Parameters(n_g_subsampled=[10,8,6], n_x_c=5, lambda_=0.95, eta=0.3, kappa=0.8, i_attractor=3,
                    use_p_inf=True, common_memory=False)

# Initialize memory and attractor
data_dim = sum(params.n_p_calculated)
storage = MemoryStorage(params)
attractor = AttractorDynamics(params)

# Fake batch of inferred/generated p for an update
B = 8
p_inf = torch.randn(B, data_dim)
p_gen = torch.randn(B, data_dim)
storage.update(p_inf, p_gen, eta=params.eta, lamb=params.lambda_)

# Retrieve with generative mask schedule
M = storage.get_memory(for_inference=False)
p_query = torch.randn(B, data_dim)
p_retrieved = attractor.retrieve(p_query, M, for_inference=False)
```

## See also

- `design.md` — memory equations, masks, and interaction with pipelines
- `src/torch_tem/memory/storage.py` — implementation of `MemoryStorage`
- `src/torch_tem/memory/attractor.py` — implementation of `AttractorDynamics`
- `docs.generation.md` — how memory is used in generation (g→p→x)
