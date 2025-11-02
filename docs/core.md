---
post_title: "torch_tem Core API Reference"
author1: "Borja Est"
post_slug: "docs-core-api"
microsoft_alias: "borja"
featured_image: ""
categories: ["documentation"]
tags: ["torch_tem", "api", "pytorch", "tem"]
ai_note: "Generated with assistance from an AI; reviewed for accuracy."
summary: "Concise API documentation for the core submodule of torch_tem, covering encoders, decoders, transition and projection components, and the utility MLP."
post_date: "2025-11-02"
---

## Scope and conventions

- Module: `src/torch_tem/core`
- Audience: contributors and advanced users wiring TEM components
- Notation: `[B, ·]` denotes batch dimension; lists index frequency modules `f = 0..n_f-1`
- Protocols: constructors accept configuration via Protocols from `torch_tem.config.facets`
- Source links are provided per class for quick navigation

## Imports

Use the public package exports:

```python
from torch_tem.core import (
  SensoryEncoder,
  ObservationDecoder,
  TransitionModel,
  ProjectionHead,
  SensoryProjection,
  MLP,
)
```

## SensoryEncoder

- File: `src/torch_tem/core/encoder.py`
- Purpose: Compress one-hot observations `x[B, n_x]` to two-hot codes `x_c[B, n_x_c]`
- Constructor: `SensoryEncoder(params: EncoderParams)`
  - Requires: `n_x: int`, `n_x_c: int`, `two_hot_table_calculated: List[Tensor]`
- Forward: `__call__(x: Tensor) -> Tensor`
  - Input: `x` — one-hot, shape `[B, n_x]`
  - Output: `x_c` — two-hot, shape `[B, n_x_c]` (exactly two active bits per row)
- Example:

```python
x_c = SensoryEncoder(params)(x)
```

## ObservationDecoder

- File: `src/torch_tem/core/decoder.py`
- Purpose: Reconstruct observation distribution from grounded location features
- Constructor: `ObservationDecoder(params: DecoderParams)`
  - Requires: `n_x: int`, `n_x_c: int`, `n_x_f_calculated: List[int]`
- Forward: `__call__(p: List[Tensor]) -> Tuple[Tensor, Tensor]`
  - Input: `p[f]` — grounded location per frequency; uses `p[0][:, :n_x_f[0]]`
  - Output: `(x_probs, x_logits)` — both `[B, n_x]`
- Notes: Internally uses an `MLP` with hidden size auto-set to mean of in/out dims.
- Example:

```python
x_probs, x_logits = ObservationDecoder(params)(p)
```

## TransitionModel

- File: `src/torch_tem/core/transition.py`
- Purpose: Predict next abstract location `g` conditioned on action or no action
- Constructor: `TransitionModel(params: TransitionParams)`
  - Requires: `n_f_calculated: int`, `n_g_calculated: List[int]`, `n_actions: int`,
    `g_connections_calculated: List[List[bool]]`, `do_sample: bool`,
    `g_init_std: float`, `g_mem_std: float`, `d_hidden_dim: int`
- Public methods:
  - `forward(g_prev: List[Tensor], a: Tensor, use_action: bool = True)
     -> Tuple[List[Tensor], List[Tensor]]`
    - Returns: `(g_gen, sigma_g)` with shapes `g_gen[f]: [B, n_g[f]]`,
      `sigma_g[f]: [B, n_g[f]]`
  - `transition_with_action(g_prev, a) -> (mu_g, sigma_g)`
  - `transition_no_action(g_prev) -> (mu_g, sigma_g)`
  - `sample(mu_g, sigma_g) -> g` (uses Gaussian sampling if `do_sample=True`)
- Notes:
  - Action pathway: frequency-parallel `MLP_D_a`; connections across modules
  - No-action pathway: per-frequency parameters `D_no_a`
  - Uncertainty: `MLP_sigma_g_path` conditioned on action

## ProjectionHead

- File: `src/torch_tem/core/projection.py`
- Purpose: Transform and downsample abstract location codes
- Constructor: `ProjectionHead(params: ProjectionParams)`
  - Requires: `n_f_calculated: int`, `n_g_calculated: List[int]`,
    `g_downsample_calculated: List[Tensor]`, `f_initial_extended: List[float]`
- Methods:
  - `transform(g: List[Tensor]) -> List[Tensor]` — Laplacian-like transform via learnable `alpha`
  - `normalize_g(g)` — clamp each `g[f]` to `[-1, 1]`
  - `normalize_p(p)` — sigmoid per frequency
  - `downsample(g)` — multiply by `g_downsample[f]` to reach subsampled space
  - `forward(g)` — `transform → downsample`
- Shapes: `g[f]: [B, n_g[f]]`; downsampled to `[B, n_g_subsampled[f]]`

## SensoryProjection

- File: `src/torch_tem/core/tiling.py`
- Purpose: Project normalized sensory features to grounded location space for memory
- Constructor: `SensoryProjection(params: SensoryProjectionParams)`
  - Requires: `n_f_calculated: int`, `n_x_f_calculated: List[int]`,
    `W_tile_calculated: List[Tensor]`
- Forward: `__call__(x_normalized: List[Tensor]) -> List[Tensor]`
  - Computes: `x_[f] = sigmoid(w_p[f]) * (x_normalized[f] @ W_tile[f])`
  - Output shapes: `x_[f]: [B, n_p[f]]`
- Notes: `w_p[f]` are learnable scalars; `W_tile` are fixed tiling matrices

## MLP (utility)

- File: `src/torch_tem/core/mlp.py`
- Purpose: Simple 2-layer MLP, single or frequency-parallel
- Constructor:
  - `MLP(
       in_dim: Union[int, List[int]],
       out_dim: Union[int, List[int]],
       activation: Tuple[Optional[Callable], Optional[Callable]] = (F.elu, None),
       hidden_dim: Optional[Union[int, List[int]]] = None,
       bias: Tuple[bool, bool] = (True, True),
     )`
- Methods:
  - `set_weights(from_layer: int, value: Union[float, Tensor, List])`
  - `get_weights(from_layer: int) -> List[Tensor]`
  - `forward(data: Union[Tensor, List[Tensor]]) -> Union[Tensor, List[Tensor]]`
- Notes:
  - If `hidden_dim` is `None`, uses mean of input/output dims per module.
  - Accepts list inputs/outputs when configured with parallel modules.

## Protocols (constructor contracts)

These live in `src/torch_tem/config/facets.py` and are satisfied by `Parameters`:

- `EncoderParams` → `n_x`, `n_x_c`, `two_hot_table_calculated`
- `DecoderParams` → `n_x`, `n_x_c`, `n_x_f_calculated`
- `TransitionParams` → `n_f_calculated`, `n_g_calculated`, `n_actions`,
  `g_connections_calculated`, `do_sample`, `g_init_std`, `g_mem_std`, `d_hidden_dim`
- `ProjectionParams` → `n_f_calculated`, `n_g_calculated`, `g_downsample_calculated`,
  `f_initial_extended`
- `SensoryProjectionParams` → `n_f_calculated`, `n_x_f_calculated`, `W_tile_calculated`

## Minimal wiring example

```python
# Assume `params` is an instance of torch_tem.config.parameters.Parameters
encoder = SensoryEncoder(params)
proj    = ProjectionHead(params)
decode  = ObservationDecoder(params)
trans   = TransitionModel(params)
s2p     = SensoryProjection(params)
```

## Source index

- `src/torch_tem/core/encoder.py`
- `src/torch_tem/core/decoder.py`
- `src/torch_tem/core/transition.py`
- `src/torch_tem/core/projection.py`
- `src/torch_tem/core/tiling.py`
- `src/torch_tem/core/mlp.py`
