---
post_title: TEM Generation API
author1: Borja Est
post_slug: tem-generation-api
microsoft_alias: na
featured_image: https://placehold.co/1200x630?text=TEM+Generation+API
categories: [AI-ML, Documentation]
tags: [PyTorch, TEM, Generative, Memory, Attractor, Decoder]
ai_note: Drafted with AI assistance and reviewed by a human.
summary: API reference for torch_tem.generation: LocationGenerator (g→p via memory and attractor dynamics) and ObservationGenerator (p→x via decoder), including contracts, shapes, and usage.
post_date: 2025-11-02
---

## Overview

This document describes the public API of the generation submodule in `torch_tem`.
The generation system implements the generative pathways in TEM:
- g → p using associative memory and attractor dynamics
- p → x using the observation decoder

Exports from `torch_tem.generation.__init__`:
- `LocationGenerator`
- `ObservationGenerator`

Related design context: see `design.md` (Generation section) for dataflow and
component responsibilities.

## Components

### LocationGenerator

Generates grounded locations (place cells, p) from abstract locations (grid
cells, g) via memory retrieval and attractor dynamics.

- Module: `torch_tem.generation.location`
- Class: `LocationGenerator`
- Inherits: `torch.nn.Module`

#### Construction

Inputs:
- `params: LocationGeneratorParams`
  - Protocol dependencies provided by the unified `Parameters` object:
    - `n_f_calculated: int` (number of frequency modules)
    - `n_p_calculated: List[int]` (place cell dims per frequency)
    - `do_sample: bool` (deterministic vs stochastic mode)
- `memory: MemoryStorage`
  - Provides learned Hebbian associations; handles separate inference/generative
    memories if configured.
- `attractor: AttractorDynamics`
  - Iterative retrieval mechanism with per-frequency early-stopping.

Example:
```python
from torch_tem.config import Parameters
from torch_tem.memory.storage import MemoryStorage
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.generation import LocationGenerator

params = Parameters(n_g_subsampled=[10,10,8,6,6], n_x_c=10)
memory = MemoryStorage(params)
attractor = AttractorDynamics(params)
generator = LocationGenerator(params, memory, attractor)
```

#### Attributes

- `n_f: int`
- `n_p: List[int]`
- `memory: MemoryStorage`
- `attractor: AttractorDynamics`
- `do_sample: bool`
- `mlp_sigma_p: Optional[MLP]` — present only when `do_sample=True`

#### Methods

- `generate(g: List[Tensor], for_inference: bool = False) -> List[Tensor]`
  - Inputs:
    - `g`: list length `n_f`, each tensor shape `[B, n_g_subsampled[f]]`
    - `for_inference`: choose inference (`True`) vs generative (`False`) memory
  - Output: list length `n_f`, each `[B, n_p[f]]`
  - Behavior:
    1. Concatenates `g` to a flat query.
    2. Selects memory matrix via `memory.get_memory(for_inference)`.
    3. Retrieves `p` using `attractor` (iterative refinement).
    4. If `do_sample=True`, adds learned uncertainty: `p = p_mu + σ(p_mu) * ε`.

- `forward(g: List[Tensor], for_inference: bool = False) -> List[Tensor]`
  - Alias of `generate` (PyTorch convention).

Note: `_split_to_frequencies(p_flat: Tensor) -> List[Tensor]` is an internal
helper splitting a flat vector back into per-frequency tensors.

#### Shapes and equations

- Per-frequency shapes:
  - `g[f] ∈ R^{B × n_g_subsampled[f]}`
  - `p[f] ∈ R^{B × n_p[f]}` with `n_p[f] = n_g_subsampled[f] · n_x_c`
- Attractor step (conceptual):
  - `p ← κ · p + Mᵀ @ p` with early stopping per frequency
- Stochastic mode:
  - `p = p_mu + σ(p_mu) · ε`, `ε ~ N(0, I)`

#### Edge cases and recommendations

- When `common_memory=True`, `for_inference` is ignored (single memory used).
- If `do_sample=False`, `mlp_sigma_p` is not created; results are deterministic.
- Ensure batch dims match across all frequency tensors.
- Input `g` must already be downsampled/normalized per design; the generator
  does not normalize `g`.

### ObservationGenerator

Generates sensory observations (x) from grounded locations (p) using the
ObservationDecoder. This completes the generative path g → p → x.

- Module: `torch_tem.generation.observation`
- Class: `ObservationGenerator`
- Inherits: `torch.nn.Module`

#### Construction

Inputs:
- `decoder: ObservationDecoder`
  - Typically constructed with the unified `Parameters` object; handles
    `n_x`, `n_x_c`, and `n_x_f_calculated`.

Example:
```python
from torch_tem.config import Parameters
from torch_tem.core.decoder import ObservationDecoder
from torch_tem.generation import ObservationGenerator

params = Parameters()
decoder = ObservationDecoder(params)
generator = ObservationGenerator(decoder)
```

#### Methods

- `generate(p: List[Tensor]) -> Tuple[Tensor, Tensor]`
  - Inputs: `p` list length `n_f`, each `[B, n_p[f]]`; uses only `p[0]` for
    decoding (highest-frequency module).
  - Returns: `(x_probs, x_logits)`
    - `x_probs`: `[B, n_x]`, softmax probabilities (sum to 1.0)
    - `x_logits`: `[B, n_x]`, raw logits for loss computation

- `forward(p: List[Tensor]) -> Tuple[Tensor, Tensor]`
  - Alias of `generate`.

#### Notes

- Using only `p[0]` assumes finest-scale place cells carry sufficient detail for
  observation prediction; coarser modules encode more abstract structure.

## Interactions and dataflow

- Typical generative flow:
  1. `g` (from transition/inference) → `LocationGenerator` → `p`
  2. `p` → `ObservationGenerator` → `(x_probs, x_logits)`
- Memory update and attractor iteration counts are configured via the unified
  `Parameters` object and managed by `MemoryStorage`/`AttractorDynamics`.

## Testing and examples

- `torch_tem/generation/location.py` includes a runnable example under
  `if __name__ == "__main__":` demonstrating memory training and generation.
- Minimal smoke test:
  - Instantiate `Parameters`, `MemoryStorage`, `AttractorDynamics`.
  - Create `LocationGenerator` and run `generate` with random `g` tensors.
  - Create `ObservationGenerator` with a configured `ObservationDecoder` and
    decode `p` to `(x_probs, x_logits)`.

## Error handling and edge cases

- Shape mismatches across frequency lists will raise runtime errors in PyTorch
  ops (concatenation/matrix multiplications). Verify `n_f`, `n_g_subsampled`,
  and `n_p` consistency from `Parameters`.
- When `common_memory=False`, ensure you select the correct memory bank via
  `for_inference`.
- For stochastic mode, confirm that the uncertainty head (`mlp_sigma_p`) is
  present and that gradients are enabled during training.

## See also

- `design.md` — architecture and detailed responsibilities
- `docs/core.md` — core encoder/decoder/transition APIs
- `src/torch_tem/memory/` — MemoryStorage and AttractorDynamics implementation
