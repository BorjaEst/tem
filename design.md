# torch_tem — Compact Design Overview

This document distills the essential architecture and contracts of torch_tem for fast navigation and LLM-friendly context. Use it as the single reference while coding.

## What it is

PyTorch implementation of the Temporal Experience Model (TEM), a hierarchical predictive coding model for navigation and memory. Configuration is centralized in a single Pydantic v2 `Parameters` model; components consume only the fields they need via Protocol-based structural typing.

## Core principles

- Single source of truth: one `Parameters` instance at runtime
- Narrow contracts: components depend on small `typing.Protocol` facets
- Computed fields: dimensions, masks, and matrices via `@computed_field`
- Validation: Pydantic v2 validators ensure self-consistent config
- Immutability/testability: prefer frozen models and Protocol-conforming fakes in tests

## Minimal contracts (Protocols)

Components type-hint against Protocols; any object with the required attributes works (including `Parameters`).

```python
# Encoder needs sensory dims and lookup tables
class EncoderParams(Protocol):
    n_x: int
    n_x_c: int
    @property
    def two_hot_table_calculated(self) -> List[Tensor]: ...

# SensoryProjection needs per-frequency sensory tiling to p-space
class SensoryProjectionParams(Protocol):
    @property
    def n_f_calculated(self) -> int: ...
    @property
    def n_x_f_calculated(self) -> List[int]: ...
    @property
    def W_tile_calculated(self) -> List[Tensor]: ...

# Grounded inference needs p-space sizes and repeat/tile matrices
class GroundedInferenceParams(Protocol):
    @property
    def n_f_calculated(self) -> int: ...
    @property
    def n_p_calculated(self) -> List[int]: ...
    n_x_c: int
    @property
    def W_repeat_calculated(self) -> List[Tensor]: ...
    @property
    def W_tile_calculated(self) -> List[Tensor]: ...
```

## Parameters model (Pydantic v2)

- Raw fields (examples): `n_x`, `n_actions`, `n_g_subsampled`, `eta`, `lambda_`, `d_hidden_dim`, flags like `use_p_inf`, `do_sample`.
- Computed fields (examples): `n_f_calculated`, `n_p_calculated`, `n_g_calculated`, `g_downsample_calculated`, frequency banks, masks, and tiling matrices.
- Validators ensure monotonicity, ranges, and feasibility.

Example:

```python
params = Parameters(
    n_x=50,
    n_actions=4,
    n_g_subsampled=[12, 12, 10, 8, 6],
    eta=0.6,
    lambda_=0.999,
)
# Pass everywhere; components pick only what they need via Protocols
encoder = SensoryEncoder(params)
```

## Package layout (essentials)

- config: `facets.py` (Protocols), `parameters.py` (unified Parameters)
- core: `encoder.py`, `decoder.py`, `transition.py`, `projection.py`, `sensory_projection.py`
- inference: `abstract.py`, `grounded.py`, `sensory.py`, `precission.py`
- memory: `storage.py`, `attractor.py`
- generation: `location.py`, `observation.py`
- model: `tem.py`, `state.py`, `pipelines.py`
- data: `environment.py`, `policies.py`, `walks.py`, `synthetic.py`, `patterns.py`, `shiny.py`, `datamodule.py`
- losses: `computer.py`

## Component responsibilities and deps

Core

- SensoryEncoder: x → x_c using two-hot tables
  - needs: `n_x`, `n_x_c`, `two_hot_table_calculated`
- ObservationDecoder: p → x reconstruction
  - needs: `n_x`, `n_x_c`, `n_x_f_calculated`
- TransitionModel: (g_prev, a) → g_gen with hierarchical connections and σ
  - needs: `n_f`, `n_g`, `n_actions`, `g_connections_calculated`, `do_sample`, `g_init_std`, `g_mem_std`, `d_hidden_dim`
- ProjectionHead: transforms/normalizes g and downsamples
  - needs: `n_f`, `n_g`, `g_downsample_calculated`, `f_initial_extended`
- SensoryProjection: x*normalized[f] @ W_tile[f] → x*[f] in p-space
  - needs: `n_f`, `n_x_f_calculated`, `W_tile_calculated`

Memory

- MemoryStorage: Hebbian matrices, hierarchical updates
  - update: M_new = λ·M_old + η·outer(p, p)
  - needs: `n_p_calculated`, `p_update_mask_calculated`, flags `use_p_inf`, `common_memory`
- AttractorDynamics: iterative retrieval with masks
  - core step: p_new = κ·p + Mᵀ @ p; early-stop per frequency
  - needs: `kappa`, `i_attractor_calculated`, `p_retrieve_mask_*`

Inference

- SensoryProcessor: temporal filtering per frequency, normalization
  - x_f[f] = f·x_c + (1−f)·x_prev[f]
  - needs: `n_f`, `n_x_c`, `n_x_f_calculated`, `f_initial_extended`
- AbstractLocationInference: precision-weighted fusion of (g_gen, memory p_x→g, shiny)
  - needs: `n_f`, `n_g`, `n_g_subsampled_combined`, `use_p_inf`, `g_init_std`, `g_mem_std`
- GroundedLocationInference: p = g ⊗ x (outer product via repeat/tile)
  - needs: `n_f`, `n_p_calculated`, `n_x_c`, `W_repeat_calculated`, `W_tile_calculated`

Generation

- LocationGenerator: g → p via memory (attractor); optional sampling
  - needs: `n_f`, `n_p_calculated`, `do_sample`, MemoryStorage, AttractorDynamics
- ObservationGenerator: p → x via decoder

Losses

- LossComputer: aggregates losses
  - L_p_g (p_inf vs p_gen), L_p_x (p_inf vs p_x), cross-entropies for x from three paths, L_g (g_inf vs g_gen), regularizers

## Pipelines (explicit dataflow)

- TransitionPipeline: g_prev + a → (g_gen, σ_g)
- InferencePipeline: x → x*c → x_f → (optional p_x via memory from x*) → g_inf → p_inf
  - If use_p_inf: project x_f to p-space with `SensoryProjection` then retrieve `p_x`
- GenerativePipeline: three paths
  1. p_inf → x
  2. g_inf → p → x
  3. g_gen → p → x

## TEM orchestrator

`tem.py` wires components and pipelines, manages memory updates, shiny signals, and loss.

- Input per step: x (one-hot), a (index or one-hot), optional locations
- Steps per iteration
  1. Transition with action; also no-action branch (for shiny/generative)
  2. Optional shiny signals
  3. Inference pipeline → (g_inf, p_inf, x_filtered, p_x)
  4. Generative pipeline for all three paths
  5. Memory update using p_inf and p_gen (from g_inf)
  6. Loss computation (training mode)
- Output: immutable `IterationState` (Pydantic) with all tensors and metrics

## Data system (for training/inference examples)

- Environment: load from JSON or generate grids; provides adjacency and shortest paths
- PolicyGenerator: random, distance-based (softmax over graph distances), Q-learning; mix policies
- WalkGenerator: produces sequences (observations, actions, locations) and batching utilities
- ShinyConfig + ShinyEnvironmentBuilder: place/mark shiny objects and goal policies
- TEMDataModule (Lightning): infinite on-the-fly walk generation; optional curriculum schedule

## Key equations and shapes

- Hebbian update: M ← λ·M + η·(p ⊗ p)
- Attractor step: p ← κ·p + Mᵀ @ p
- Sensory projection: x\_[f] = σ(w_p[f]) · (x_normalized[f] @ W_tile[f])
- Grounded location: p = g ⊗ x (implemented via repeat/tile matrices)
- Typical per-frequency shapes:
  - g[f] ∈ R^{B × n_g[f]} | x_f[f] ∈ R^{B × n_x_c} | p[f] ∈ R^{B × n_p[f]} with n_p[f] = n_g_subsampled[f]·n_x_c

## Usage quickstart

```python
from torch_tem.config import Parameters
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.inference.grounded import GroundedLocationInference

params = Parameters(n_x=50, n_actions=4, n_g_subsampled=[12,12,10,8,6], eta=0.6, lambda_=0.999)
encoder = SensoryEncoder(params)                    # params conforms to EncoderParams
# ... build other components with the same params; keep signatures typed to Protocols
```

## Testing guidance

- Prefer Protocol-conforming fakes (Pydantic BaseModel with `arbitrary_types_allowed=True`) for unit tests
- For integration, instantiate `Parameters` once and pass everywhere

## Out-of-scope for core logic (optional)

- Visualization in `figures/` (data, sensory, grounded) — returns matplotlib Figures, no side effects

---

Keep component signatures against Protocols, not the concrete `Parameters` type; derive dependent values via `@computed_field` to preserve immutability and conformance at instantiation.
