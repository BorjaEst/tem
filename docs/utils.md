---
post_title: "torch_tem utils API"
author1: "Project Authors"
post_slug: "utils-api"
microsoft_alias: "borjaest"
featured_image: "https://via.placeholder.com/1200x630.png?text=torch_tem+utils"
categories: ["Documentation", "API"]
tags: ["torch_tem", "utils", "PyTorch", "matrix", "graph"]
ai_note: "This document was drafted with AI assistance and reviewed by a human."
summary: "Reference guide for torch_tem.utils: matrices, masks, grid detection, and graph layout helpers used across TEM components."
post_date: "2025-11-02"
---

## Overview

Utilities used by TEM components to generate matrices, masks, and graph
layouts. These helpers encapsulate common tensor constructions and structural
operations so core modules can focus on model logic.

- Frameworks: NumPy, PyTorch; optional NetworkX and SciPy.
- All tensors are float unless noted. Shapes are per-frequency when lists are
  returned.
- Public API is re-exported from `torch_tem.utils` for convenience.

## Public modules

### torch_tem.utils.matrices

Matrix constructors for outer-product computation, downsampling, two-hot
encoding, and grid detection.

### torch_tem.utils.masks

Mask constructors for hierarchical memory updates and retrieval, and
inter-module transition connections.

### torch_tem.utils.layouts

Graph layout utilities that auto-detect grid topology or fall back to standard
graph layouts.

## Re-exports (import surface)

From `torch_tem.utils` you can import:

- `create_W_repeat`, `create_W_tile`, `create_g_downsample`,
  `create_two_hot_table`, `create_p_update_mask`, `create_p_retrieve_masks`,
  `create_g_connections`, `detect_grid_structure`, `compute_graph_layout`.

## Function reference

### matrices.create_W_repeat

Signature:

```python
create_W_repeat(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Tensor]
```

Build repeat matrices to expand abstract location `g` for elementwise outer
product with sensory features `x`. Each matrix has shape `[n_p[f], n_g'[f]]`
where `n_p[f] = n_g_subsampled[f] * n_x_f[f]` and `n_g'[f] = n_g_subsampled[f]`.

- Parameters:
  - n_g_subsampled: per-frequency compressed g sizes.
  - n_x_f: per-frequency sensory sizes in p-space.
- Returns: list of PyTorch tensors, one per frequency.
- Used by: grounded inference and generation when computing `p = g ⊗ x`.

Example:

```python
from torch_tem.utils import create_W_repeat
W = create_W_repeat([3], [4])[0]  # shape [12, 3]
```

### matrices.create_W_tile

Signature:

```python
create_W_tile(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Tensor]
```

Build tile matrices to replicate sensory `x` for elementwise outer product with
`g`. Each matrix has shape `[n_p[f], n_x_f[f]]`.

- Parameters: same as `create_W_repeat`.
- Returns: list of tiling matrices.
- Used by: sensory projection and grounded inference.

### matrices.create_g_downsample

Signature:

```python
create_g_downsample(n_g: List[int], n_g_subsampled: List[int]) -> List[Tensor]
```

Downsample full grid-cell vectors to compressed indices by taking the first
`n_g_subsampled[f]` entries. Each matrix has shape `[n_g[f], n_g_subsampled[f]]`.

- Parameters:
  - n_g: per-frequency full g sizes.
  - n_g_subsampled: per-frequency compressed g sizes.
- Returns: list of downsampling matrices.
- Notes: implemented via vertical concatenation of identity and zeros.

### matrices.create_two_hot_table

Signature:

```python
create_two_hot_table(n_x: int, n_x_c: int) -> List[Tensor]
```

Create lookup table of 2-hot codes (length `n_x_c`) up to `min(C(n_x_c, 2), n_x)`
entries. Used to compress one-hot observations to two-hot representations.

- Parameters:
  - n_x: number of observations.
  - n_x_c: compressed sensory dimension.
- Returns: list of column vectors (float tensors) representing two-hot codes.
- Dependency: SciPy `comb` for code count; falls back to algorithmic generation.

Example:

```python
from torch_tem.utils import create_two_hot_table
codes = create_two_hot_table(10, 6)  # list of tensors of shape [6]
```

### matrices.detect_grid_structure

Signature:

```python
detect_grid_structure(adj: np.ndarray, n_locs: int) -> Optional[Tuple[int, int]]
```

Detect if the adjacency describes a rectangular 4-neighbor grid. Returns
`(width, height)` if detected, otherwise `None`.

- Parameters:
  - adj: square adjacency matrix as NumPy array.
  - n_locs: number of nodes.
- Returns: optional `(width, height)`.

### layouts.compute_graph_layout

Signature:

```python
compute_graph_layout(adj: np.ndarray, n_locs: int) -> Tuple[np.ndarray, np.ndarray]
```

Compute node positions for visualization:

1. If `detect_grid_structure` succeeds → grid layout normalized to `[-1, 1]`.
2. Else try NetworkX (Kamada-Kawai for small graphs, spring otherwise).
3. Else fall back to circular layout.

- Parameters:
  - adj: adjacency matrix.
  - n_locs: number of nodes.
- Returns: `(x, y)` arrays in range `[-1, 1]`.
- Notes: y is flipped to match top-down orientation for grids.

Example:

```python
import numpy as np
from torch_tem.utils import compute_graph_layout
adj = np.array([[0,1,1,0],[1,0,0,1],[1,0,0,1],[0,1,1,0]])
x, y = compute_graph_layout(adj, 4)
```

### masks.create_p_update_mask

Signature:

```python
create_p_update_mask(
	n_p: List[int], n_f: int, n_f_g: int, n_f_ovc: int, f_initial: List[float]
) -> Tensor
```

Create binary mask `[sum(n_p), sum(n_p)]` for Hebbian memory updates. Enables
hierarchical updates from lower to higher frequency modules. When OVC modules
are present, allows full connectivity between OVC and grid modules and applies
hierarchy within each family.

- Parameters:
  - n_p: per-frequency grounded location sizes.
  - n_f: total modules; n_f_g: grid modules; n_f_ovc: OVC modules.
  - f_initial: per-module initial frequencies (ordering reference).
- Returns: float tensor mask.

### masks.create_p_retrieve_masks

Signature:

```python
create_p_retrieve_masks(
	n_p: List[int], i_attractor: int,
	i_attractor_max_freq_inf: List[int],
	i_attractor_max_freq_gen: List[int],
) -> tuple[List[Tensor], List[Tensor]]
```

Create per-iteration retrieval masks implementing early stopping per frequency
for both inference and generation. Each list has `i_attractor` vectors of
length `sum(n_p)`.

- Parameters: see signature.
- Returns: `(p_retrieve_mask_inf, p_retrieve_mask_gen)`.

### masks.create_g_connections

Signature:

```python
create_g_connections(
	n_f: int, n_f_g: int, n_f_ovc: int, f_initial: List[float]
) -> List[List[bool]]
```

Create hierarchical inter-module connection matrix for abstract location
transitions. Grid→grid and OVC→OVC obey low→high frequency; cross-family
connections are disabled.

- Parameters: see signature.
- Returns: `connections[f_to][f_from]` boolean matrix.

## Usage notes and conventions

- All functions are pure and side-effect free; they only construct arrays or
  tensors from sizes and hyperparameters from `Parameters`.
- Shapes follow the design: `n_p[f] = n_g_subsampled[f] * n_x_f[f]`.
- Use `torch_tem.config.Parameters` to compute the sizes passed to these utils.
- Optional deps: `networkx` for non-grid layouts; `scipy` for combinatorics.

## Examples: common flows

### Grounded location outer product

```python
from torch_tem.utils import create_W_repeat, create_W_tile
W_r, W_t = create_W_repeat([8,6],[10,10]), create_W_tile([8,6],[10,10])
# For each frequency f:
#   g_rep = g @ W_r[f].T      # [B, n_p[f]]
#   x_til = x @ W_t[f].T      # [B, n_p[f]]
#   p = g_rep * x_til         # elementwise
```

### Memory masks

```python
from torch_tem.utils import create_p_update_mask, create_p_retrieve_masks
mask = create_p_update_mask([80,60], n_f=2, n_f_g=2, n_f_ovc=0, f_initial=[0.3,0.1])
inf, gen = create_p_retrieve_masks([80,60], 5, [5,3], [5,4])
```
