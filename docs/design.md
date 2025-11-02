---
post_title: "TEM Design Overview"
author1: "Borja Est"
post_slug: "tem-design-overview"
microsoft_alias: "borja"
featured_image: ""
categories: ["Documentation"]
tags: ["torch_tem", "design", "architecture", "tem", "pytorch"]
ai_note: "Generated with AI assistance and reviewed."
summary: "High-level design for torch_tem: core principles, configuration contracts, components, dataflow, and key equations with shapes."
post_date: "2025-11-02"
---

## Overview

The Temporal Experience Model (TEM) in `torch_tem` is a hierarchical predictive
coding system for navigation and memory. Its configuration is centralized in a
single Pydantic v2 `Parameters` model. Components consume only the fields they
need through narrow `typing.Protocol` contracts. This document summarizes the
architecture, contracts, and dataflow for contributors.

## Core principles

- Single source of truth: one `Parameters` instance per run.
- Narrow contracts: constructors depend on Protocol facets, not the concrete type.
- Computed fields: sizes, masks, and matrices are provided via read‑only
  computed properties.
- Validation: Pydantic v2 validators ensure mutually consistent settings.
- Immutability and testability: prefer read‑only semantics and Protocol‑conforming
  fakes in unit tests.

## Configuration and Protocols

Configuration lives in `torch_tem.config.parameters.Parameters`. Minimal contracts
are defined in `torch_tem.config.facets` as Protocols. Any object that provides
these attributes can be used, enabling duck typing and easier tests.

Examples of constructor contracts used by components:

- EncoderParams → `n_x`, `n_x_c`, `two_hot_table_calculated`.
- DecoderParams → `n_x`, `n_x_c`, `n_x_f_calculated`.
- TransitionParams → `n_f_calculated`, `n_g_calculated`, `n_actions`,
  `g_connections_calculated`, `do_sample`, `g_init_std`, `g_mem_std`, `d_hidden_dim`.
- ProjectionParams → `n_f_calculated`, `n_g_calculated`, `g_downsample_calculated`,
  `f_initial_extended`.
- SensoryProjectionParams → `n_f_calculated`, `n_x_f_calculated`,
  `W_tile_calculated`.
- GroundedInferenceParams → `n_f_calculated`, `n_p_calculated`, `n_x_c`,
  `W_repeat_calculated`, `W_tile_calculated`.

## Package layout (essentials)

- `src/torch_tem/config`: `facets.py` (Protocols), `parameters.py` (unified model)
- `src/torch_tem/core`: `encoder.py`, `decoder.py`, `transition.py`,
  `projection.py`, `tiling.py`, `mlp.py`
- `src/torch_tem/inference`: `sensory.py`, `grounded.py`, `abstract.py`,
  `precission.py`
- `src/torch_tem/memory`: `storage.py`, `attractor.py`
- `src/torch_tem/generation`: `location.py`, `observation.py`
- `src/torch_tem/model`: `tem.py`, `state.py`, `pipelines.py`
- `src/torch_tem/data`: `environment.py`, `policies.py`, `walks.py`, `synthetic.py`,
  `patterns.py`, `shiny.py`, `datamodule.py`
- `src/torch_tem/losses`: `computer.py`

## Components and responsibilities

### Core

- SensoryEncoder: maps one‑hot `x[B, n_x]` to two‑hot `x_c[B, n_x_c]` using a
  lookup table. Needs: `n_x`, `n_x_c`, `two_hot_table_calculated`.
- ObservationDecoder: reconstructs observation `(x_probs, x_logits)` from grounded
  location. Needs: `n_x`, `n_x_c`, `n_x_f_calculated`.
- TransitionModel: predicts next abstract location `g` with action and no‑action
  paths, produces uncertainty `σ_g`. Needs: `n_f_calculated`, `n_g_calculated`,
  `n_actions`, `g_connections_calculated`, `do_sample`, `g_init_std`, `g_mem_std`,
  `d_hidden_dim`.
- ProjectionHead: transforms and downsamples `g` to subsampled space using
  `g_downsample_calculated`. Needs: `n_f_calculated`, `n_g_calculated`,
  `g_downsample_calculated`, `f_initial_extended`.
- SensoryProjection: projects normalized sensory features to grounded `p`
  space with tiling matrices. Needs: `n_f_calculated`, `n_x_f_calculated`,
  `W_tile_calculated`.

### Inference

- SensoryProcessor: temporal filtering per frequency and normalization.
  Uses smoothing factors `f_initial_extended`.
- GroundedLocationInference: computes `p = g ⊗ x` efficiently using repeat and
  tile matrices. Needs `n_p_calculated`, `W_repeat_calculated`, `W_tile_calculated`.
- AbstractLocationInference: fuses `(g_gen, σ_g_gen)` with optional memory‑based
  `p_x → g` and shiny signals via precision weighting.

### Generation

- LocationGenerator: `g → p` via `MemoryStorage` and `AttractorDynamics`, with
  optional learned uncertainty when `do_sample=True`.
- ObservationGenerator: `p → x` using the ObservationDecoder.

### Memory

- MemoryStorage: Hebbian associative memory over `p` with optional dual banks for
  inference vs generation. Masked hierarchical updates.
- AttractorDynamics: iterative retrieval `p ← κ·p + Mᵀ @ p` with early‑stopping
  per frequency using retrieval masks.

### Losses and model orchestration

- LossComputer: aggregates losses across paths, including `L_p_g`, `L_p_x`,
  reconstruction losses for `x`, `L_g`, and regularizers.
- TEM orchestrator (`model/tem.py`): wires components and pipelines, manages
  memory updates, optional shiny signals, and returns an immutable step state.

## Pipelines and dataflow

- Transition: `(g_prev, a) → (g_gen, σ_g)`.
- Inference: `x → x_c → x_f → (optional p_x) → g_inf → p_inf`.
  - If `use_p_inf=True`, `x_f` is projected to `p` and optionally refined with
    memory for `p_x`.
- Generative: three decoding paths used for training and visualization:
  1. `p_inf → x`, 2) `g_inf → p → x`, 3) `g_gen → p → x`.

## Key equations and shapes

- Hebbian update: `M ← λ·M + η·(p ⊗ p)` then apply `p_update_mask`.
- Attractor step: `p ← κ·p + Mᵀ @ p` using iteration‑dependent masks.
- Sensory projection: `x_[f] = σ(w_p[f]) · (x_norm[f] @ W_tile[f])`.
- Grounded location: `p = g ⊗ x` via repeat/tile matrices.
- Shape conventions per frequency `f`:
  - `g[f] ∈ R^{B × n_g[f]}`, `x_f[f] ∈ R^{B × n_x_c]}`,
    `p[f] ∈ R^{B × n_p[f]}` with `n_p[f] = n_g_subsampled[f] · n_x_c`.

## Minimal wiring example

```python
from torch_tem.config import Parameters
from torch_tem.core import SensoryEncoder, ObservationDecoder, TransitionModel, ProjectionHead
from torch_tem.inference import SensoryProcessor, GroundedLocationInference, AbstractLocationInference

params = Parameters(n_x=50, n_actions=4, n_g_subsampled=[12,12,10,8,6])
encoder = SensoryEncoder(params)
decoder = ObservationDecoder(params)
trans   = TransitionModel(params)
proj    = ProjectionHead(params)
sens    = SensoryProcessor(params)
ground  = GroundedLocationInference(params)
abst    = AbstractLocationInference(params)
```

## Notes

- Prefer Protocol‑typed constructor signatures over the concrete `Parameters`
  type to keep components loosely coupled.
- Use computed fields for matrices, masks, and derived sizes instead of building
  them ad‑hoc in user code.
