---
post_title: TEM Inference API
author1: Borja Est
post_slug: tem-inference-api
microsoft_alias: borja
featured_image: https://placehold.co/1200x630?text=TEM+Inference+API
categories: [AI-ML, Documentation]
tags: [PyTorch, TEM, Inference, Sensory, Grounded, Abstract, Precision]
ai_note: Drafted with AI assistance and reviewed by a human.
summary: API reference for torch_tem.inference covering SensoryProcessor, GroundedLocationInference, AbstractLocationInference, and precision utilities with contracts, shapes, and usage.
post_date: 2025-11-02
---

## Overview

This document describes the public API of the inference submodule in `torch_tem`.
The inference system processes sensory input, computes grounded location codes (p),
and infers abstract locations (g) by combining multiple sources via precision
weighting.

Exports from `torch_tem.inference.__init__`:

- `SensoryProcessor`
- `GroundedLocationInference`
- `AbstractLocationInference`

Related design context: see `design.md` (Inference section) for dataflow and
component responsibilities.

## Components

### SensoryProcessor

Temporal filtering and normalization of compressed sensory input `x_c`.

- Module: `torch_tem.inference.sensory`
- Class: `SensoryProcessor`
- Inherits: `torch.nn.Module`

#### Construction

- `SensoryProcessor(params: SensoryProcessorParams)`
  - Protocol dependencies (satisfied by `Parameters`):
    - `n_f_calculated: int` — number of frequency channels
    - `n_x_c: int` — sensory feature dimension
    - `n_x_f_calculated: int` — downstream convenience (unused here)
    - `f_initial_extended: List[float]` — EMA coefficients in (0, 1]

#### Attributes

- `n_f: int` — number of frequencies
- `n_x_c: int` — sensory dimension
- `n_x_f: int` — calculated downstream size
- `f_initial: Sequence[float]` — per-frequency smoothing factors
- `w_x: nn.Parameter` — affine weight `[1, n_x_c]`
- `b_x: nn.Parameter` — affine bias `[1, n_x_c]`

#### Methods

- `filter_temporal(x_c: Tensor, x_prev: List[Tensor]) -> List[Tensor]`
  - `x_f[f] = f * x_c + (1 - f) * x_prev[f]`
- `normalize(x_f: List[Tensor]) -> List[Tensor]`
  - `x_norm = (w_x * x + b_x) / ||w_x * x + b_x||_2`
- `forward(x_c: Tensor, x_prev: List[Tensor]) -> List[Tensor]`
  - Returns filtered and normalized list of tensors, each `[B, n_x_c]`

#### Example

```python
from torch_tem.inference import SensoryProcessor
proc = SensoryProcessor(params)
x_out = proc(x_c, [torch.zeros_like(x_c) for _ in range(proc.n_f)])
```

---

### GroundedLocationInference

Compute grounded location (place cells) by binding abstract location with sensory
context via outer product: `p[f] = g[f] ⊗ x[f]`.

- Module: `torch_tem.inference.grounded`
- Class: `GroundedLocationInference`
- Inherits: `torch.nn.Module`

#### Construction

- `GroundedLocationInference(params: GroundedInferenceParams)`
  - Protocol dependencies (satisfied by `Parameters`):
    - `n_f_calculated: int`
    - `n_p_calculated: List[int]` — `n_p[f] = n_g_subsampled[f] * n_x_c`
    - `n_x_c: int`
    - `W_repeat_calculated: List[Tensor]` — expands `g`
    - `W_tile_calculated: List[Tensor]` — expands `x`

#### Behavior

Uses Kronecker product matrices for efficient batched computation:

- `G = g[f] @ W_repeat[f]   # [B, n_p[f]]` expands grid cells per sensory slot
- `X = x[f] @ W_tile[f]     # [B, n_p[f]]` tiles sensory across grid positions
- `p[f] = σ(w_p[f]) · (G ⊙ X)` with learnable scalar `w_p[f]` per frequency

#### Methods

- `forward(g_downsampled: List[Tensor], x_filtered: List[Tensor]) -> List[Tensor]`
  - Inputs: `g_downsampled[f]: [B, n_g_subsampled[f]]`, `x_filtered[f]: [B, n_x_c]`
  - Output: `p[f]: [B, n_p[f]]`

#### Example

```python
from torch_tem.inference import GroundedLocationInference
grounded = GroundedLocationInference(params)
p = grounded(g_downsampled, x_filtered)
```

---

### AbstractLocationInference

Infer abstract location `g` by fusing multiple sources via precision-weighted
averaging:

1. Transition prediction `(g_gen, σ_g_gen)`
2. Memory-based `p_x → g` (when `use_p_inf=True`)
3. Optional shiny-object signals `(μ_g_shiny, σ_g_shiny)`

- Module: `torch_tem.inference.abstract`
- Class: `AbstractLocationInference`
- Inherits: `torch.nn.Module`

#### Construction

- `AbstractLocationInference(params: AbstractInferenceParams)`
  - Protocol dependencies (satisfied by `Parameters`):
    - `n_f_calculated: int`
    - `n_g_calculated: List[int]`
    - `n_g_subsampled_combined: List[int]` — input size for memory path
    - `use_p_inf: bool`
    - `g_init_std: float`, `g_mem_std: float`

Creates:

- `mlp_mu_g_mem`: maps `p_x[f]` to `μ_g_mem[f]`
- `mlp_sigma_g_mem`: maps quality signals `[B, 2]` to `σ_g_mem[f]`
- `g_init[f]` and `logsig_g_init[f]`: learnable priors for new environments

#### Methods

- `forward(g_gen, sigma_g_gen, p_x, shiny_signals, p2g_scale_offset) -> List[Tensor]`
  - Inputs:
    - `g_gen, sigma_g_gen`: lists length `n_f`, each `[B, n_g[f]]`
    - `p_x`: optional list `[B, n_g_subsampled_combined[f]]` (if `use_p_inf`)
    - `shiny_signals`: optional tuple `(mu_g_shiny, sigma_g_shiny)`
    - `p2g_scale_offset`: float to schedule memory influence
  - Output: `g_inf[f]: [B, n_g[f]]`

Internal steps:

- Compute `μ_g_mem` via `mlp_mu_g_mem(p_x)` when available
- Estimate memory uncertainty `σ_g_mem` from quality signals:
  - `g_norm = ||μ_g_mem||^2`, `recon_error = 0` placeholder → concat to `[B,2]`
- Collect sources and apply precision-weighted mean per frequency
- Scale memory sigmas with `offset` for curriculum scheduling

#### Example

```python
from torch_tem.inference import AbstractLocationInference
abst = AbstractLocationInference(params)
g_inf = abst(g_gen, sigma_g_gen, p_x=None, shiny_signals=None, p2g_scale_offset=0.0)
```

---

## Utilities (precision weighting)

- Module: `torch_tem.inference.precission`

Functions:

- `precision_weighted_mean(means: List[Tensor], sigmas: List[Tensor], epsilon=1e-8) -> Tensor`
- `precision_weighted_mean_list(means_list: List[List[Tensor]], sigmas_list: List[List[Tensor]], epsilon=1e-8) -> List[Tensor]`

Definition:

- `precision_i = 1 / (σ_i^2 + ε)`
- `mean = sum(precision_i * μ_i) / sum(precision_i)`

## Interactions and dataflow

Typical inference pipeline (see `design.md`):

1. `x` → `SensoryEncoder` (core) → `x_c`
2. `x_c` + `x_prev` → `SensoryProcessor` → `x_filtered`
3. If `use_p_inf=True`: `x_filtered` → `SensoryProjection` (core) → `x_*` and memory → `p_x`
4. `(g_gen, σ_g_gen)` + optional `(p_x, shiny)` → `AbstractLocationInference` → `g_inf`
5. `(g_inf, x_filtered)` → `GroundedLocationInference` → `p_inf`

## Minimal wiring example

```python
from torch_tem.inference import SensoryProcessor, GroundedLocationInference, AbstractLocationInference

sens = SensoryProcessor(params)
grounded = GroundedLocationInference(params)
abst = AbstractLocationInference(params)

x_norm_list = sens(x_c, x_prev)
p_inf = grounded(g_downsampled, x_norm_list)
# Fuse sources to infer g
g_inf = abst(g_gen, sigma_g_gen, p_x=None, shiny_signals=None, p2g_scale_offset=0.0)
```

## Error handling and edge cases

- Ensure per-frequency lists all have length `n_f` and matching batch sizes.
- When `use_p_inf=False` or `p_x is None`, the memory path is skipped.
- `p2g_scale_offset` increases `σ_g_mem` effectively reducing memory influence.
- Numerical stability: small epsilon added in norms and precision weighting.

## See also

- `design.md` — architecture and responsibilities
- `docs.core.md` — core encoders/decoders/projection utilities
- `docs.generation.md` — generative path APIs
