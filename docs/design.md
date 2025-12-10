# TEM Technical Design Document

## Overview

This document describes the technical architecture for implementing the Tolman-Eichenbaum Machine (TEM) using a modular PyTorch design. The implementation follows the reference paper (Whittington et al., 2020) while leveraging modern PyTorch patterns and typed configuration management.

## Component Integration Patterns

### Modular Architecture

The `TEMModel` class orchestrates multiple specialized submodules, each responsible for a distinct aspect of the TEM computational graph:

```
TEMModel
├── SensoryEncoder: One-hot → two-hot compression
├── SensoryProcessor: Temporal filtering and normalization
├── TransitionModel: Action-conditioned abstract location dynamics
├── GroundedLocInference: Conjunctive coding (g ⊗ x → p)
├── AbstractLocInference: Precision-weighted fusion
├── MemoryStorage: Hebbian associative memory
├── AttractorDynamics: Iterative pattern completion
├── Projection: Abstract location transformations
└── ObservationDecoder: Place cells → sensory predictions
```

### Component Initialization Pattern

Each component receives configuration via protocol-based dependency injection:

```python
class TEMModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Configuration-derived matrices (computed once)
        self.two_hot_table = compute_two_hot_table(config.architecture.n_x_c)
        self.W_repeat = compute_repeat_matrices(config.architecture.n_g, config.architecture.n_x_c)
        self.W_tile = compute_tile_matrices(config.architecture.n_g, config.architecture.n_x_c)
        # ... (g_downsample, g_connections, p_update_mask, p_retrieve_masks)

        # Component instantiation
        self.encoder = SensoryEncoder(config, self.two_hot_table)
        self.processor = SensoryProcessor(config)
        self.transition = TransitionModel(config, self.g_connections)
        # ... (remaining components)
```

**Key Principle**: Components receive only what they need—minimal configuration protocols rather than the full `ModelConfig`.

### Data Flow Orchestration

The `TEMModel` acts as a **thin orchestration layer** that:

1. **Routes tensors** between specialized components
2. **Converts data formats** (per-frequency ↔ concatenated)
3. **Maintains state** across iterations (memory M, filtered sensory x_prev, abstract location g_prev)
4. **Computes losses** by comparing inference and generative paths

Components remain **stateless** and **pure** (except for learnable parameters), enabling independent testing and reuse.

---

## Data Format Conversions

### Per-Frequency vs. Concatenated Representations

TEM operates on **multi-scale frequency modules** (typically 5: 4 grid cell modules + 1 object-vector cell module). Different operations require different tensor layouts:

#### Per-Frequency Format

**Structure**: `List[Tensor]` where each element corresponds to one frequency module  
**Shape**: `List[[B, n_p[f]]]` for f ∈ [0, n_f)  
**Use cases**:

- Hierarchical operations (low→high frequency dependencies)
- Frequency-specific transformations (scaling, downsampling)
- Loss computation per module
- Component inputs/outputs (GroundedLocInference, Projection)

**Example**:

```python
p_inf = [
    torch.randn(32, 100),  # Frequency 0: batch=32, n_p[0]=10×10=100
    torch.randn(32, 100),  # Frequency 1: batch=32, n_p[1]=10×10=100
    torch.randn(32, 64),   # Frequency 2: batch=32, n_p[2]=8×8=64
    torch.randn(32, 36),   # Frequency 3: batch=32, n_p[3]=6×6=36
    torch.randn(32, 36),   # Frequency 4: batch=32, n_p[4]=6×6=36 (OVC)
]
```

#### Concatenated Format

**Structure**: `Tensor`  
**Shape**: `[B, sum(n_p)]` where sum(n_p) = n_p[0] + ... + n_p[n_f-1]  
**Use cases**:

- Memory operations (Hebbian outer products, attractor dynamics)
- Efficient batch matrix operations
- Storage and checkpointing

**Example**:

```python
p_flat = torch.randn(32, 336)  # 32 × (100+100+64+36+36) = 32 × 336
```

### Conversion Utilities

**`concatenate_frequencies(p_list: List[Tensor]) -> Tensor`**

```python
# From: List[[B, n_p[f]]] → To: [B, sum(n_p)]
p_flat = torch.cat(p_list, dim=1)
```

**`split_to_frequencies(p_flat: Tensor, n_p: List[int]) -> List[Tensor]`**

```python
# From: [B, sum(n_p)] → To: List[[B, n_p[f]]]
p_list = torch.split(p_flat, n_p, dim=1)
```

### Format Flow Through Pipeline

```
Observation (x: [B, n_x])
  ↓ encoder
Compressed (x_c: [B, n_x_c])
  ↓ processor.filter
Filtered (x_f: List[[B, n_x_f[f]]])  ← PER-FREQUENCY
  ↓ sensory_projection.tile
Tiled (x_: List[[B, n_p[f]]])  ← PER-FREQUENCY
  ↓ concatenate_frequencies
Query (x_flat: [B, sum(n_p)])  ← CONCATENATED
  ↓ attractor.retrieve
Retrieved (p_flat: [B, sum(n_p)])  ← CONCATENATED
  ↓ split_to_frequencies
Grounded (p: List[[B, n_p[f]]])  ← PER-FREQUENCY
```

**Critical**: Memory operations (`MemoryStorage.update`, `AttractorDynamics.retrieve`) always operate on **concatenated format**. All other components prefer **per-frequency format**.

---

## Configuration Management

### Typed Configuration Hierarchy

```
ModelConfig (src/torch_tem/config/__init__.py)
├── ArchitectureConfig (architecture.py)
│   ├── Base dimensions: n_x, n_x_c, n_g, f_initial
│   └── Derived dimensions: n_f, n_g_total, n_x_f, n_p
├── EnvironmentConfig (environment.py)
│   ├── Grid: width, height, observation_mode
│   └── Shiny: n_shiny_objects, shiny_influence
├── InferenceConfig (inference.py)
│   ├── Runtime: do_sample, use_memory_for_inference
│   └── Hebbian: eta_g (learning), eta_p (decay)
└── TrainingConfig (training.py)
    ├── Optimization: n_epochs, batch_size, learning_rate
    └── Curriculum: loss_weights_*, walk_len_curriculum
```

### Configuration Validation

**Pydantic v2** provides automatic validation:

```python
from pydantic import BaseModel, Field, computed_field, model_validator

class ArchitectureConfig(BaseModel):
    model_config = ConfigDict(strict=True, validate_assignment=True)

    n_x: int = Field(gt=0, description="Sensory observation dimensions")
    n_x_c: int = Field(gt=0, description="Compressed sensory dimensions")
    n_g: List[int] = Field(min_length=1, description="Grid cells per frequency")

    @computed_field
    @property
    def n_f(self) -> int:
        """Total frequency modules (grid + OVC)."""
        return len(self.n_g)

    @computed_field
    @property
    def n_p(self) -> List[int]:
        """Place cells per frequency = n_g[f] × n_x_f[f]."""
        return [self.n_g[f] * self.n_x_f[f] for f in range(self.n_f)]

    @model_validator(mode='after')
    def validate_dimensions(self):
        if sum(self.n_p) > 10000:
            raise ValueError("Total place cells exceed memory limit")
        return self
```

**Benefits**:

- Runtime type checking (no silent type coercion)
- Clear error messages on invalid configuration
- Automatic computed fields (no manual calculation)
- Dependency validation (e.g., n_p depends on n_g and n_x_f)

### Configuration Access Pattern

**Never pass raw dictionaries**. Always use typed config objects:

```python
# ❌ Bad: Dictionary access
n_g = params['n_g']

# ✅ Good: Typed access with autocomplete
n_g = config.architecture.n_g
```

---

## Hierarchical Connectivity Schemes

### Grid Cell Hierarchy

TEM uses **hierarchical grid cells** inspired by the entorhinal cortex:

```
Frequency 0 (highest) ──┐
Frequency 1 ────────────┼──┐
Frequency 2 ────────────┼──┼──┐
Frequency 3 ────────────┼──┼──┼──┐
Frequency 4 (lowest) ────┴──┴──┴──┘
                      (low → high connections)
```

**Principle**: Low-frequency modules (global position) influence high-frequency modules (fine-grained details), but not vice versa.

### Abstract Location Transitions (g → g')

**Connectivity Matrix**: `g_connections[f_to][f_from]`

```python
# Example for n_f=5
g_connections = [
    [1, 0, 0, 0, 0],  # f=0: only self-connections
    [1, 1, 0, 0, 0],  # f=1: receives from f=0, f=1
    [1, 1, 1, 0, 0],  # f=2: receives from f=0, f=1, f=2
    [1, 1, 1, 1, 0],  # f=3: receives from f=0, f=1, f=2, f=3
    [1, 1, 1, 1, 1],  # f=4: receives from all
]
```

**Transition MLP** for frequency `f_to`:

```
Input: concatenate([g[f_from] for f_from where g_connections[f_to][f_from]])
Hidden: tanh(W1 × input + b1)
Output: W2 × hidden  (shape: [n_g[f_to],])
```

### Hebbian Memory Connectivity (p ↔ p)

**Update Mask**: `p_update_mask[i, j]` indicates whether connection M[i,j] should be updated

```python
# Hierarchical: only low→high and same-frequency connections
p_update_mask = compute_hebbian_mask(
    n_p,
    hierarchical=True,
    allow_same_frequency=True
)
# Shape: [sum(n_p), sum(n_p)]
# Non-zero entries: connections respecting hierarchy
```

**Memory Update**:

```python
M_new = lambda * M_old + eta * (p_inferred ⊗ p_generated) ⊙ p_update_mask
```

where `⊙` denotes element-wise multiplication (masking).

### Attractor Retrieval Early-Stopping

**Retrieval Masks**: `p_retrieve_masks[iteration]` specifies which frequencies to update at each iteration

```python
# Example for n_f=5, n_retrieve_iterations=5
p_retrieve_masks = [
    [1, 0, 0, 0, 0],  # Iteration 0: only update frequency 0
    [1, 1, 0, 0, 0],  # Iteration 1: update f=0, f=1
    [1, 1, 1, 0, 0],  # Iteration 2: update f=0, f=1, f=2
    [1, 1, 1, 1, 0],  # Iteration 3: update f=0, f=1, f=2, f=3
    [1, 1, 1, 1, 1],  # Iteration 4: update all frequencies
]
```

**Algorithm**:

```python
for it, mask in enumerate(p_retrieve_masks):
    p_new = tanh(p_query + M @ p)
    # Update only masked frequencies
    for f in range(n_f):
        if mask[f]:
            p[f] = p_new[f]
```

**Intuition**: Low-frequency (coarse) representations stabilize first, providing context for high-frequency (fine) representations to refine.

---

## Memory Architecture (Dual Optional)

### Generative Memory (M_gen)

**Purpose**: Support generation pathway `g → p → x`  
**Update**: `M_gen = λM_gen + η(p_inf ⊗ p_gen)`  
**Usage**:

- `gen_p(g_inf, M_gen)`: Retrieve p from inferred abstract location
- `gen_p(g_gen, M_gen)`: Retrieve p from path-integrated abstract location

**Always present** in TEM.

### Inference Memory (M_inf)

**Purpose**: Support inference pathway `x → p → g`  
**Update**: `M_inf = λM_inf + η(p_inf ⊗ p_x)`  
**Usage**:

- `attractor.retrieve(x_, M_inf)`: Retrieve p from sensory input

**Optional** (controlled by `config.inference.use_memory_for_inference`):

- If `True` and `common_memory=False`: Maintain separate M_inf
- If `True` and `common_memory=True`: Alias M_inf = M_gen (share weights)
- If `False`: Skip sensory-based retrieval entirely

### Memory Storage Implementation

```python
class MemoryStorage(nn.Module):
    def __init__(
        self,
        config: ArchitectureConfigProtocol,
        inference_config: InferenceConfigProtocol,
        p_update_mask: Tensor,
    ):
        super().__init__()
        n_p_total = sum(config.n_p)

        # Always create M_gen
        self.register_buffer('M_gen', torch.zeros(1, n_p_total, n_p_total))

        # Optionally create M_inf
        if inference_config.use_memory_for_inference:
            if inference_config.common_memory:
                self.M_inf = self.M_gen  # Alias (shared weights)
            else:
                self.register_buffer('M_inf', torch.zeros(1, n_p_total, n_p_total))
        else:
            self.M_inf = None

        self.register_buffer('p_update_mask', p_update_mask)
        self.eta = inference_config.eta_g
        self.lamb = inference_config.eta_p

    def update(self, p_inferred_flat: Tensor, p_generated_flat: Tensor):
        """Hebbian update for both memories."""
        # Update M_gen
        outer = torch.bmm(p_inferred_flat.unsqueeze(2), p_generated_flat.unsqueeze(1))
        self.M_gen = self.lamb * self.M_gen + self.eta * (outer * self.p_update_mask)

        # Update M_inf if separate
        if self.M_inf is not None and self.M_inf is not self.M_gen:
            # For M_inf, use p_inferred ⊗ p_x (not p_generated)
            # Note: requires p_x to be passed separately
            pass  # Implementation details in actual code

    def get_memory(self, for_inference: bool = False) -> Tensor:
        """Retrieve appropriate memory."""
        if for_inference and self.M_inf is not None:
            return self.M_inf
        return self.M_gen
```

**Design Decision**: Optional dual memory provides flexibility:

- **Dual memory**: Better zero-shot inference (separate sensory and abstract pathways)
- **Common memory**: Simpler, fewer parameters, faster training
- **No inference memory**: Ablation study (pure path integration)

---

## Tensor Shape Specifications

### Notation

- `B`: Batch size
- `T`: Time steps in walk sequence
- `n_x`: Sensory observation dimensions (e.g., 45 for 9×5 one-hot grid)
- `n_x_c`: Compressed sensory dimensions (e.g., 10 for two-hot)
- `n_x_f[f]`: Temporally filtered sensory dimensions per frequency
- `n_g[f]`: Abstract location (grid cell) dimensions per frequency
- `n_p[f]`: Grounded location (place cell) dimensions per frequency = n_g[f] × n_x_f[f]
- `n_f`: Number of frequency modules (grid + OVC)
- `sum(n_p)`: Total place cells across all frequencies

### Pipeline Tensor Shapes

| Stage          | Variable     | Shape                     | Format   | Notes                              |
| -------------- | ------------ | ------------------------- | -------- | ---------------------------------- |
| **Input**      |              |                           |          |                                    |
|                | `x`          | `[B, n_x]`                | Single   | One-hot sensory observation        |
|                | `a`          | `List[int]` length B      | List     | Discrete actions per walk          |
|                | `locations`  | `List[Dict]` length B     | List     | Environment metadata               |
| **Encoding**   |              |                           |          |                                    |
|                | `x_c`        | `[B, n_x_c]`              | Single   | Two-hot compressed                 |
| **Filtering**  |              |                           |          |                                    |
|                | `x_f`        | `List[[B, n_x_f[f]]]`     | Per-freq | Temporally filtered sensory        |
| **Tiling**     |              |                           |          |                                    |
|                | `x_`         | `List[[B, n_p[f]]]`       | Per-freq | Tiled for outer product            |
|                | `x_flat`     | `[B, sum(n_p)]`           | Concat   | For memory retrieval               |
| **Transition** |              |                           |          |                                    |
|                | `g_prev`     | `List[[B, n_g[f]]]`       | Per-freq | Previous abstract location         |
|                | `g_gen`      | `List[[B, n_g[f]]]`       | Per-freq | Path-integrated abstract           |
|                | `sigma_gen`  | `List[[B, n_g[f]]]`       | Per-freq | Transition uncertainty             |
| **Inference**  |              |                           |          |                                    |
|                | `p_x`        | `List[[B, n_p[f]]]`       | Per-freq | Retrieved from sensory             |
|                | `g_inf`      | `List[[B, n_g[f]]]`       | Per-freq | Inferred abstract location         |
|                | `p_inf`      | `List[[B, n_p[f]]]`       | Per-freq | Inferred grounded location         |
| **Generation** |              |                           |          |                                    |
|                | `p_gen`      | `List[[B, n_p[f]]]`       | Per-freq | Retrieved from g_inf               |
|                | `x_p`        | `[B, n_x]`                | Single   | Observation from p_inf             |
|                | `x_g`        | `[B, n_x]`                | Single   | Observation from g_inf→p           |
|                | `x_gt`       | `[B, n_x]`                | Single   | Observation from g_gen→p           |
|                | `x_logits`   | `Tuple[3 × [B, n_x]]`     | Tuple    | Pre-softmax scores                 |
| **Memory**     |              |                           |          |                                    |
|                | `p_inf_flat` | `[B, sum(n_p)]`           | Concat   | For Hebbian update                 |
|                | `p_gen_flat` | `[B, sum(n_p)]`           | Concat   | For Hebbian update                 |
|                | `M_gen`      | `[B, sum(n_p), sum(n_p)]` | Concat   | Generative memory matrix           |
|                | `M_inf`      | `[B, sum(n_p), sum(n_p)]` | Concat   | Inference memory matrix (optional) |
| **Loss**       |              |                           |          |                                    |
|                | `L_p_g`      | `[B]`                     | Single   |                                    |
|                | `L_p_x`      | `[B]`                     | Single   |                                    |
|                | `L_x_gen`    | `[B]`                     | Single   | Cross-entropy(x, x_gt)             |
|                | `L_x_g`      | `[B]`                     | Single   | Cross-entropy(x, x_g)              |
|                | `L_x_p`      | `[B]`                     | Single   | Cross-entropy(x, x_p)              |
|                | `L_g`        | `[B]`                     | Single   |                                    |
|                | `L_reg_g`    | `[B]`                     | Single   |                                    |
|                | `L_reg_p`    | `[B]`                     | Single   |                                    |

### Memory Operation Shapes

**Hebbian Update**:

```python
p_inferred: [B, sum(n_p)]
p_generated: [B, sum(n_p)]
outer = p_inferred[:, :, None] @ p_generated[:, None, :]  # [B, sum(n_p), sum(n_p)]
M = lambda * M + eta * outer  # [B, sum(n_p), sum(n_p)]
```

**Attractor Retrieval**:

```python
p_query: [B, sum(n_p)]
M: [B, sum(n_p), sum(n_p)]
p_retrieved = tanh(p_query + (M @ p_prev.unsqueeze(2)).squeeze(2))  # [B, sum(n_p)]
```

### Outer Product (Conjunctive Coding)

**Naive approach** (memory-intensive):

```python
g: [B, n_g]
x: [B, n_x]
p = g[:, :, None] @ x[:, None, :]  # [B, n_g, n_x] → flatten → [B, n_g × n_x]
```

**Efficient approach** (using projection matrices):

```python
W_repeat: [n_g × n_x, n_g]  # Kronecker repeat
W_tile: [n_g × n_x, n_x]    # Kronecker tile
g_expanded = g @ W_repeat.T  # [B, n_g × n_x]
x_expanded = x @ W_tile.T    # [B, n_g × n_x]
p = g_expanded * x_expanded  # [B, n_g × n_x] element-wise
```

**Rationale**: Avoids explicit outer product, reduces memory from O(B × n_g × n_x) to O(B × (n_g + n_x)).

---

## Implementation Checklist

### Phase 1: Infrastructure (Complete)

- [x] Configuration system (`config/`)
- [x] Core components (`core/`)
- [x] Inference components (`inference/`)
- [x] Memory components (`memory/`)
- [x] Utility functions (`utils/`)
- [x] Data pipeline (`data/`)

### Phase 2: Integration (Complete)

- [x] TEMModel.**init**: Instantiate all components
- [x] TEMModel.forward: Walk sequence processing loop
- [x] TEMModel.iteration: Single-step orchestration
- [x] TEMModel.inference: Encode → filter → tile → retrieve → infer
- [x] TEMModel.generative: Generate via 3 routes (p, g_inf, g_gen)
- [x] TEMModel.loss: Compute 8 loss components
- [x] Helper methods: gen_g, gen_p, gen_x, inf_g, inf_p
- [x] Initialization: init_walks, init_iteration
- [x] Data transformations: x*prev2x, x2x*, g2g\_

### Phase 3: Validation (Complete)

- [x] Smoke tests: Forward pass with random data (scripts/smoke_test_tem.py)
- [x] Shape checks: Verify all tensor shapes match specification
- [x] Gradient flow: Ready for backpropagation through all paths
- [x] Component integration: All modules work together correctly

---

## Implementation Status

**Status**: ✅ **COMPLETE** (November 21, 2025)

### Architectural Decisions

**1. Modular Component Architecture**

The implementation replaces the original model.py's monolithic approach with specialized, reusable components. This decision was made to improve:

- **Testability**: Each component can be unit-tested independently
- **Maintainability**: Clear separation of concerns, easier to debug
- **Reusability**: Components can be used in other models or experiments
- **Type Safety**: Protocol-based interfaces with Pydantic validation

**2. Legacy Method Removal**

Removed 16 legacy method stubs (f_mu_g_path, f_sigma_g_path, f_mu_g_mem, f_sigma_g_mem, f_mu_g_shiny, f_sigma_g_shiny, f_sigma_p, f_x, f_c_star, f_c, f_n, f_g, f_g_clamp, f_p, attractor, hebbian) from the original model.py API. These were replaced by modular component methods:

| Legacy Method(s)              | Replacement Component | Rationale                                |
| ----------------------------- | --------------------- | ---------------------------------------- |
| f_mu_g_path, f_sigma_g_path   | TransitionModel       | Encapsulates action-conditioned dynamics |
| f_mu_g_mem, f_sigma_g_mem     | AbstractLocInference  | Handles precision-weighted fusion        |
| f_mu_g_shiny, f_sigma_g_shiny | AbstractLocInference  | Integrates object-vector signals         |
| f_c, f_c_star                 | SensoryEncoder        | Two-hot encoding/decoding                |
| f_n                           | SensoryProcessor      | Normalization and temporal filtering     |
| f_g, f_g_clamp                | Projection            | Downsampling and normalization           |
| f_p                           | GroundedLocInference  | Sparse activation in outer product       |
| attractor                     | AttractorDynamics     | Iterative pattern completion             |
| hebbian                       | MemoryStorage         | Hebbian learning with hierarchical masks |

**3. Data Format Discipline**

Strict separation between per-frequency (List[Tensor]) and concatenated (Tensor) formats:

- **Per-frequency**: Used by all hierarchical operations (inference, transition, projection)
- **Concatenated**: Used only by memory operations (storage, retrieval)
- **Conversions**: Explicit via concatenate_frequencies() and split_to_frequencies()

This discipline prevents format mismatches and makes data flow explicit.

**4. Configuration Management**

Pydantic v2 with strict mode provides:

- Runtime type checking (no silent coercion)
- Clear error messages on invalid configs
- Automatic computed fields (e.g., n_p = n_g × n_x_f)
- Dependency validation across config sections

No manual validation code required—configuration errors caught at initialization.

### Performance Considerations

**Optimized Operations**:

- Outer products via Kronecker matrices (avoids explicit 3D tensors)
- Single-pass format conversions (torch.cat, torch.split)
- Pre-computed connectivity masks (no runtime computation)
- Batched memory operations (parallel across walks)

**Future Optimizations** (if needed):

- @torch.jit.script for critical paths
- Mixed precision training (FP16)
- Gradient checkpointing for long sequences
- Custom CUDA kernels for outer products

### Testing & Validation

**Smoke Test** (`scripts/smoke_test_tem.py`):

- ✅ Model instantiation with default config
- ✅ Single iteration with synthetic data
- ✅ Full forward pass (5-step walk)
- ✅ Tensor shape verification throughout pipeline

**Ready for**:

- Integration with environment (gridworld, continuous)
- Training experiments (supervised, self-supervised)
- Ablation studies (dual memory, hierarchical connectivity)
- Deployment in navigation tasks

### Phase 3: Future Enhancements

- [ ] Shiny object support (optional feature for salient landmarks)
- [ ] Probabilistic sampling (do_sample mode with Gumbel-softmax)
- [ ] Reference comparison with original model.py outputs
- [ ] Performance profiling and optimization
- [ ] Extended environment support (continuous spaces, 3D)

---

## Design Principles Summary

1. **Separation of Concerns**: Each component handles one transformation
2. **Stateless Components**: No hidden state except learnable parameters
3. **Explicit Data Flow**: Tensor routing through TEMModel orchestration
4. **Type Safety**: Pydantic validation catches configuration errors
5. **Format Discipline**: Clear conventions for per-frequency vs. concatenated
6. **Hierarchical Respect**: Low→high frequency information flow
7. **Memory Flexibility**: Optional dual memory for different use cases
8. **Testability**: Components can be unit-tested independently

---

## References

- **Paper**: Whittington et al. (2020), "The Tolman-Eichenbaum Machine: Unifying space and relational memory through generalisation in the hippocampal formation", bioRxiv
- **Original Implementation**: `model.py` (reference code by James Whittington & Jacob Bakermans)
- **Configuration**: `src/torch_tem/config/` (Pydantic v2)
- **Components**: `src/torch_tem/core/`, `src/torch_tem/inference/`, `src/torch_tem/memory/`
