---
post_title: "TEM Figures API"
author1: "Project maintainers"
post_slug: "tem-figures-api"
microsoft_alias: "none"
featured_image: ""
categories:
  - "documentation"
tags:
  - "tem"
  - "pytorch"
  - "visualization"
  - "matplotlib"
ai_note: "This post was drafted with AI assistance and reviewed by a human."
summary: "API reference for the torch_tem.figures submodule: plotting utilities for data, sensory processing, grounded location inference, and memory diagnostics."
post_date: "2025-11-02"
---

## Overview

The `torch_tem.figures` submodule provides standardized, side-effect-free plotting
utilities for the Tolman‑Eichenbaum Machine (TEM). All functions:

- Return matplotlib `Figure` objects (caller chooses to show/save)
- Accept narrow Protocol-based interfaces for loose coupling
- Follow consistent styling and naming
- Emphasize shapes and reproducible visuals

## Conventions

- Inputs prefer PyTorch `Tensor` types; lists/NumPy accepted where noted.
- Shapes are written as [T, B, D] etc. Frequencies are per-module.
- Functions never call `plt.show()` or `plt.savefig()`.
- Optional dependency: `networkx` improves graph layouts for environments.

## Install and imports

- Required: `matplotlib`, `torch`, `numpy`
- Optional: `networkx` (for force‑directed layout in environment graphs)

Example import:

```python
from torch_tem import figures as F
# or import individual helpers, e.g.:
from torch_tem.figures import plot_walks, plot_frequency_bank
```

## Public surface

Exported names from `torch_tem.figures`:

- Data: `plot_environment_layout`, `plot_policy_comparison`, `plot_walks`,
  `plot_walk_statistics`, `plot_batch_tensors`
- Grounded: `plot_grounded_location_activity`, `plot_outer_product_structure`,
  `plot_place_cell_dynamics`
- Memory: `plot_memory_matrices`, `plot_attractor_convergence`, `plot_learning_curve`,
  `plot_hierarchical_masks`, `plot_retrieval_quality`, `plot_memory_difference`
- Sensory: `plot_frequency_bank`, `plot_temporal_filtering`, `plot_frequency_comparison`,
  `plot_normalization_effects`, `plot_multi_frequency_representation`

---

## Data visualizations

### plot_environment_layout(env, title="Environment Layout", figsize=(10, 10))

- Description: Renders the environment graph with an automatically chosen layout
  (grid, force‑directed via NetworkX if available, otherwise circular).
- Params:
  - env: Protocol with `n_locations`, `n_observations`, `n_actions`, `adjacency: Tensor`
  - title: Plot title
  - figsize: (width, height)
- Returns: `matplotlib.figure.Figure`
- Notes: Uses `torch_tem.utils.compute_graph_layout` under the hood.

### plot_policy_comparison(env, policies, goal_location=None, figsize=(6, 6))

- Description: Heatmap comparison of multiple policies across locations and actions.
- Params:
  - env: Environment protocol
  - policies: `List[Tuple[str, List[LocationProtocol]]]` where each location has
    an `actions` collection with `.id` and `.probability`.
  - goal_location: Optional location id to annotate title
  - figsize: Per‑subplot size
- Returns: `Figure`

### plot_walks(env, walks, title="Generated Walks", figsize=(12, 8))

- Description: Overlays trajectories of multiple walks on the environment graph.
- Params:
  - env: Environment protocol
  - walks: `List[WalkProtocol]` with `observations`, `actions`, `locations`
  - title, figsize
- Returns: `Figure`

### plot_walk_statistics(walks, figsize=(14, 10))

- Description: 2×2 dashboard: walk length histogram, location visit frequency,
  action distribution, and action repeat analysis.
- Params: `walks: List[WalkProtocol]`, `figsize`
- Returns: `Figure`

### plot_batch_tensors(obs, actions, locations, figsize=(14, 10), max_walks=5)

- Description: 2×2 dashboard for a training/eval batch.
- Shapes: `obs [B, T, n_obs]`, `actions [B, T]`, `locations [B, T]`.
- Params: tensors above; `max_walks` controls overlay count in line plots.
- Returns: `Figure`

---

## Grounded (place cell) visualizations

### plot_grounded_location_activity(p_history, observations, locations, frequencies, n_cells_per_freq, max_cells=30, figsize=None, title=None)

- Description: Time series overview of place cell activity per frequency module,
  plus the corresponding observations and locations.
- Shapes:
  - `p_history`: List[T] of List[n_f] of `[B, n_p[f]]`
  - `observations`: `[T, n_x]` (one‑hot)
  - `locations`: `[T]`
- Params: `frequencies: List[float]`, `n_cells_per_freq: List[int]`, `max_cells`,
  `figsize`, `title`
- Returns: `Figure`

### plot_outer_product_structure(g_sample, x_sample, p_sample, frequencies, figsize=None, title=None)

- Description: Single‑timepoint visualization of the conjunctive code `p = g ⊗ x`
  for each frequency (bar charts for g, x; heatmap for p reshaped).
- Shapes per frequency f: `g[1, n_g_sub[f]]`, `x[1, n_x_c]`, `p[1, n_p[f]]` with
  `n_p[f] = n_g_sub[f] * n_x_c`.
- Params: lists per frequency; `frequencies`, `figsize`, `title`
- Returns: `Figure`

### plot_place_cell_dynamics(p_history, observations, frequencies, n_cells_per_freq, cell_indices=None, figsize=None, title=None)

- Description: Tracks individual place cells over time, one subplot per frequency.
- Params/Shapes: as in `plot_grounded_location_activity`; `cell_indices` defaults
  to middle index per frequency.
- Returns: `Figure`

---

## Memory visualizations

### plot_memory_matrices(M_gen, M_inf=None, n_p_per_freq=None, n_training_steps=None, title=None, figsize=(12, 5), cmap="RdBu_r")

- Description: Heatmaps for generative and optionally inference memory matrices,
  with optional per‑frequency boundaries.
- Shapes: `[Σ n_p, Σ n_p]` for matrices; `n_p_per_freq: List[int]`.
- Returns: `Figure`

### plot_attractor_convergence(queries, retrievals, targets, query_labels=None, title="Attractor Dynamics: Query → Retrieval Convergence", figsize=None, ylim=(-0.5, 1.5))

- Description: Side‑by‑side bars for query (noisy), retrieved (after attractor),
  and target patterns across test cases.
- Params: lists of 1D tensors `[n_p_total]`; layout sizes auto‑computed.
- Returns: `Figure`

### plot_learning_curve(memory_strengths, cosine_sims=None, title=None, figsize=(12, 4))

- Description: Training curves for Hebbian memory strength and, if provided,
  cosine similarity between `M_gen` and `M_inf`.
- Params: `memory_strengths: List[float]`, optional `cosine_sims: List[float]`.
- Returns: `Figure`

### plot_hierarchical_masks(masks, n_p_per_freq=None, title="Hierarchical Mask Schedule (Coarse-to-Fine Convergence)", figsize=None, ylim=(0, 1.2))

- Description: Attractor iteration schedule showing progressive unfreezing from
  coarse to fine frequencies.
- Params: `masks: List[Tensor]` over iterations, optional `n_p_per_freq` to draw
  boundaries.
- Returns: `Figure`

### plot_retrieval_quality(errors_by_mode, noise_levels, title="Attractor Dynamics: Robustness to Noisy Queries", figsize=(10, 6), xlabel="Query Noise Level", ylabel="Retrieval Error (MSE)")

- Description: Error curves vs. noise for different retrieval modes (e.g.,
  inference vs. generative memory).
- Params: dict of mode→errors; `noise_levels: List[float]`.
- Returns: `Figure`

### plot_memory_difference(M1, M2, labels=("Memory 1", "Memory 2"), n_p_per_freq=None, title="Memory Matrix Difference", figsize=(15, 5), cmap="RdBu_r")

- Description: Side‑by‑side matrices and their difference (`M2 − M1`), with
  optional per‑frequency boundaries.
- Params: two matrices `[Σ n_p, Σ n_p]`; labels and styling.
- Returns: `Figure`

---

## Sensory visualizations

### plot_frequency_bank(frequencies, title="Frequency Bank Configuration", figsize=(12, 4))

- Description: Dual panel showing frequency values and effective time constants
  (τ = 1/f). Values are annotated on bars.
- Params: `frequencies: List[float]`, `title`, `figsize`
- Returns: `Figure`

### plot_temporal_filtering(x_c_history, x_f_history, frequencies, title="Temporal Filtering Across Frequencies", figsize=(14, 2), cmap="viridis")

- Description: Heatmaps for original compressed sensory and per‑frequency filtered
  outputs over time.
- Shapes: `x_c_history [T, n_x_c]`, `x_f_history` as List[T] of List[n_f] `[n_x_c]`.
- Returns: `Figure`

### plot_frequency_comparison(x_c_history, x_f_history, frequencies, feature_idx=0, title="Single Feature Across Frequencies", figsize=(14, 6))

- Description: Line plots of a single feature across all frequencies, with
  annotations for approximate time constants.
- Returns: `Figure`

### plot_normalization_effects(x_f_raw, x_f_normalized, frequencies, title="L2 Normalization Effects", figsize=(3, 6), cmap="coolwarm")

- Description: Before/after normalization comparison per frequency.
- Shapes: `x_f_raw[f], x_f_normalized[f]` are `[B, n_x_c]`.
- Returns: `Figure`

### plot_multi_frequency_representation(x_f_list, frequencies, timesteps, title="Multi-Frequency Representation", figsize=(4, 3), cmap="viridis")

- Description: Grid of representations at selected timesteps × frequencies.
- Params: `x_f_list`: List over time of List[n_f] `[B, n_x_c]`; `timesteps: List[int]`.
- Returns: `Figure`

---

## Tips and troubleshooting

- If bar/heatmap axes appear crowded, reduce `n_f`, subsample cells (e.g.,
  `max_cells`), or increase `figsize`.
- For reproducible layouts in environment plots, seed any random layout engine
  in NetworkX if you provide it externally.
- Always operate on CPU copies for large tensors before plotting to avoid GPU
  sync overhead.

## References

- See `design.md` for the architectural overview and plotting contracts.
- Public API is exported via `src/torch_tem/figures/__init__.py`.
