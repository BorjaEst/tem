---
post_title: torch_tem.config API Reference
author1: Borja Esteban
post_slug: torch-tem-config-api
microsoft_alias: borja
featured_image: ""
categories: ["Documentation"]
tags: ["torch_tem", "config", "pydantic", "protocols", "api"]
ai_note: Generated with AI assistance and reviewed.
summary: Authoritative API guide for the torch_tem.config submodule, covering the Parameters model and Protocol-based facets used across the TEM architecture.
post_date: "2025-11-02"
---

## Overview

The `torch_tem.config` submodule provides a single source of truth for all configuration used by the Temporal Experience Model (TEM) and a set of Protocol-based facets that expose minimal, stable contracts to the rest of the codebase.

- `parameters.py`: Unified Pydantic v2 `Parameters` model with raw and computed fields.
- `facets.py`: Lightweight `typing.Protocol` interfaces that components depend on instead of the full `Parameters` type.

Design goals:

- One `Parameters` instance per run; immutability encouraged via computed fields.
- Narrow contracts for components to improve testability and coupling.
- Explicit, typed, and validated configuration with Pydantic v2.

## Modules and public surface

- `torch_tem.config.parameters`
  - `Parameters`: Main configuration model; satisfies all Protocols via duck typing.
- `torch_tem.config.facets`
  - Protocols describing the minimal attributes required by each component.

## Parameters model

`Parameters` is a Pydantic v2 `BaseModel` with `extra='forbid'` and `arbitrary_types_allowed=True`. Fields are grouped conceptually; several values are computed via `@computed_field` and exposed as read-only properties.

### World configuration

- `has_static_action: bool = True` — Include stand-still action.
- `n_actions: int = 4` — Actions excluding stand-still.
- `explore_bias: float = 2.0` — Repeat-action bias for straighter walks.
- Shiny behavior (reward-driven policy shaping):
  - `shiny_rate: float = 0.0` — Fraction of envs with shiny objects.
  - `shiny_gamma: float = 0.7` — Q-learning discount for shiny objective.
  - `shiny_beta: float = 1.5` — Softmax inverse temperature.
  - `shiny_n: int = 2` — Number of shiny objects.
  - `shiny_returns: int = 15` — Re-visits after finding a shiny object.
  - `shiny: dict[str, Any]` — Grouped shiny parameters.
- Computed:
  - `shiny_calculated: dict[str, Any]` — Aggregated shiny configuration for world builders.

### Neural architecture

- Sensory:
  - `n_x: int = 45` — Observation neurons.
  - `n_x_c: int = 10` — Compressed sensory neurons.
  - `n_x_f: List[int] = []` — Per-frequency filtered sensory sizes (computed if empty).
- Abstract location (grid cells):
  - `n_g_subsampled: List[int] = [10, 10, 8, 6, 6]` — Per-module subsampled sizes.
  - `n_g: List[int] = []` — Full grid sizes (computed: typically 3× subsampled).
- Grounded location (place cells):
  - `n_p: List[int] = []` — Per-module grounded sizes (computed: `n_g_subsampled[f]*n_x_c`).
- Object-vector cells (optional):
  - `n_ovc: List[int] = []` — Added per module; integrated or separate by flag.
- Module organization:
  - `n_f_g: int = 5` — Grid modules.
  - `n_f_ovc: int = 0` — OVC modules (computed if `separate_ovc`).
  - `n_f: int = 5` — Total modules (computed).
  - `f_initial: List[float] = [0.99, 0.3, 0.09, 0.03, 0.01]` — Per-module temporal factors.
- Computed (selected):
  - `n_g_subsampled_combined: List[int]` — Combine grid and OVC per policy.
  - `n_f_ovc_calculated: int` — Derived OVC module count.
  - `n_f_g_calculated: int` — Grid module count (alias of `n_f_g`).
  - `n_f_calculated: int` — Total module count (grid ± OVC).
  - `n_g_calculated: List[int]` — Grid sizes per module.
  - `n_x_f_calculated: List[int]` — Filtered sensory sizes per module.
  - `n_p_calculated: List[int]` — Grounded sizes per module.
  - `f_initial_extended: List[float]` — Extended factors including OVC if separate.

### Model behavior flags

- `do_sample: bool = False` — Stochastic sampling vs. means.
- `use_p_inf: bool = True` — Use inferred p when inferring new g.
- `separate_ovc: bool = False` — Use separate OVC modules.

### Network initialization

- `g_init_std: float = 0.5` — Initial g std.
- `g_mem_std: float = 0.1` — MLP hidden→out init std.
- `d_hidden_dim: int = 20` — Transition MLP hidden size.

### Memory system (Hebbian)

- Plasticity:
  - `lambda_: float = 0.9999` — Forgetting (alias `lambda`).
  - `eta: float = 0.5` — Remembering.
  - `kappa: float = 0.8` — Retrieval decay.
- Attractor:
  - `i_attractor: int = 5` — Iterations per step.
  - `i_attractor_max_freq_inf: List[int] = []` — Early stop caps per frequency (inference).
  - `i_attractor_max_freq_gen: List[int] = []` — Early stop caps per frequency (generative).
- Memory sharing:
  - `common_memory: bool = False` — Share between generative/inference.
- Computed:

  - `i_attractor_calculated: int` — Equals `n_f_g_calculated` by default.
  - `p_retrieve_mask_inf_calculated: List[Tensor]` — Early-stop masks.
  - `p_retrieve_mask_gen_calculated: List[Tensor]` — Early-stop masks.

### Training configuration

- Basics:
  - `train_it: int = 20000` — Number of walks.
  - `n_rollout: int = 20` — Steps before BPTT.
  - `batch_size: int = 16` — Parallel walks.
- Walk lengths:
  - `walk_it_min: int = 25`
  - `walk_it_max: int = 300`
  - `walk_it_window: float = 55.0`
  - Computed: `walk_it_window_calculated: float` — Window used for sampling.
- Learning rate schedule:
  - `lr_max: float = 9.4e-4`, `lr_min: float = 8e-5`, `lr_decay_rate: float = 0.5`, `lr_decay_steps: int = 4000`.
- Loss weights:
  - Scalar knobs: `loss_weights_x`, `loss_weights_p`, `loss_weights_g`, `loss_weights_reg_g`, `loss_weights_reg_p`.
  - `loss_weights: Tensor = zeros(8)` — In order: `[L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]`.
- Curriculum schedules (iterations to “fully on”):
  - `loss_weights_p_g_it`, `loss_weights_reg_p_it`, `loss_weights_reg_g_it`, `eta_it`, `lambda_it`.
- Precision-weighted mean (p→g):
  - `p2g_scale_offset`, `p2g_sig_val`, `p2g_sig_half_it`, `p2g_sig_scale_it`.

### Connectivity and static matrices

- Hierarchical connectivity:
  - `p_update_mask: Tensor` — Hebbian connections low→high.
  - `p_retrieve_mask_inf: List[Tensor]` — Early-stop masks (inference).
  - `p_retrieve_mask_gen: List[Tensor]` — Early-stop masks (generative).
  - `g_connections: List[List[bool]]` — Grid transition connectivity.
- Outer-product helpers:
  - `W_repeat: List[Tensor]` — Repeat g for `g ⊗ x`.
- Computed:
  - `p_update_mask_calculated: Tensor` — Mask from module hierarchy.
  - `g_connections_calculated: List[List[bool]]` — Derived grid connectivity.
  - `W_repeat_calculated: List[Tensor]` — Repeat matrices per frequency.
  - `W_tile_calculated: List[Tensor]` — Tile matrices per frequency.
  - `two_hot_table_calculated: List[Tensor]` — One-hot → two-hot tables for `x → x_c`.
  - `g_downsample_calculated: List[Tensor]` — `g → g_subsampled` downsampling.

## Computed-field naming conventions

- Suffix `_calculated` indicates a read-only, lazily computed property derived from the raw fields. These properties are used throughout the system via Protocols and should be preferred over manually precomputing tensors in user code.

## Protocol facets (contracts)

Each component depends on a small Protocol defined in `facets.py`. The `Parameters` class satisfies all of them via duck typing. Highlights:

### Core

- `EncoderParams` — needs: `n_x`, `n_x_c`, `two_hot_table_calculated`.
- `DecoderParams` — needs: `n_x`, `n_x_c`, `n_x_f_calculated`.
- `TransitionParams` — needs: `n_f_calculated`, `n_g_calculated`, `n_actions`, `g_connections_calculated`, `do_sample`, `g_init_std`, `g_mem_std`, `d_hidden_dim`.
- `ProjectionParams` — needs: `n_f_calculated`, `n_g_calculated`, `g_downsample_calculated`, `f_initial_extended`.
- `SensoryProjectionParams` — needs: `n_f_calculated`, `n_x_f_calculated`, `W_tile_calculated`.

### Memory

- `MemoryStorageParams` — needs: `n_p_calculated`, `p_update_mask_calculated`, `use_p_inf`, `common_memory`.
- `AttractorParams` — needs: `kappa`, `i_attractor_calculated`, `p_retrieve_mask_inf_calculated`, `p_retrieve_mask_gen_calculated`.

### Inference

- `AbstractInferenceParams` — needs: `n_f_calculated`, `n_g_calculated`, `n_g_subsampled_combined`, `use_p_inf`, `g_init_std`, `g_mem_std`.
- `GroundedInferenceParams` — needs: `n_f_calculated`, `n_p_calculated`, `n_x_c`, `W_repeat_calculated`, `W_tile_calculated`.
- `SensoryProcessorParams` — needs: `n_f_calculated`, `n_x_c`, `n_x_f_calculated`, `f_initial_extended`.

### Generation and loss

- `LocationGeneratorParams` — needs: `n_f_calculated`, `n_p_calculated`, `do_sample`.
- `LossComputerParams` — needs: `loss_weights`.

### Composite

- `TEMParams` — composite of common fields plus all frequently used computed properties; pass a `Parameters` instance in practice.

## Shapes and utilities

- Per frequency `f`:
  - `g[f] ∈ R^{B × n_g[f]}`, `x_c ∈ R^{B × n_x_c}`, `x_f[f] ∈ R^{B × n_x_c}`.
  - `p[f] ∈ R^{B × n_p[f]}` with `n_p[f] = n_g_subsampled[f] · n_x_c`.
  - `W_tile[f] ∈ R^{n_x_c × n_p[f]}`, `W_repeat[f] ∈ R^{n_g[f] × n_p[f]}`.

## Usage example (minimal)

```python
from torch_tem.config.parameters import Parameters

params = Parameters(
		n_x=50,
		n_actions=4,
		n_g_subsampled=[12, 12, 10, 8, 6],
		eta=0.6,
		lambda_=0.999,
)

# Pass `params` anywhere a Protocol-typed config is expected, e.g.:
# encoder = SensoryEncoder(params)  # conforms to EncoderParams
```

## Testing guidance

- Prefer Protocol-conforming fakes (lightweight Pydantic models) for unit tests.
- For integration tests, instantiate a single `Parameters` and share it.

## Notes

- Keep component signatures typed to Protocols, not the concrete `Parameters` class.
- When adding new fields, also consider whether a computed counterpart is needed and whether any Protocol needs to be updated.
