# TEM Refactoring & Enhancement TODO

## Architecture Improvements

### Biologically-Inspired EC→HPC Projections

**Motivation:**
Current implementation uses structured Kronecker products (downsample + W_repeat) for entorhinal cortex to hippocampus projections. Recent CAN models (Chandra et al. 2025, "Episodic and associative memory from spatial scaffolds in the hippocampus") demonstrate that EC→HPC connectivity is actually random and sparse (~10-20% connectivity).

**Current Implementation:**

```python
# Two-step structured transformation
g_ = projection(g)                    # Downsample: [36,30,24] → [12,10,8]
g_ = [g_[f] @ W_repeat[f] for f ...]  # Expand: [12,10,8] → [96,80,64]
```

**Proposed Implementation:**

```python
# Single-step random projection
W_random = create_W_random_projection(n_g, n_p, sparsity=0.15)
g_ = [g[f] @ W_random[f] for f in range(n_f)]
```

**Benefits:**

- ✅ Biologically realistic (matches experimental EC→CA3/CA1 connectivity)
- ✅ Simpler architecture (one operation instead of two)
- ✅ More expressive (random projections less constrained than Kronecker)
- ✅ Can be sparse like real neural circuits
- ✅ Aligns with modern CAN literature

**Implementation Status:**

- [x] Utility function added: `create_W_random_projection()` in `src/torch_tem/utils/matrices.py`
- [ ] Update `Projection` in `src/torch_tem/core/projection.py`
- [ ] Update inverse projection in `AbstractLocInference._compute_memory_estimate()`
- [ ] Add configuration flag: `use_random_projection: bool` (default: False for compatibility)
- [ ] Ablation study: Compare structured vs random projections on standard benchmarks
- [ ] Hyperparameter tuning: Find optimal sparsity levels

**Files to Modify:**

1. `src/torch_tem/core/projection.py` - Replace downsample+W_repeat with W_random option
2. `src/torch_tem/inference/abstract.py` - Update p_x → g projection (currently uses W_repeat.T)
3. `src/torch_tem/config/architecture.py` - Add configuration parameters
4. `tests/` - Add tests for random projection variant
5. `examples/` - Create comparison example

**Considerations:**

- ⚠️ Breaks mathematical symmetry: `p = g ⊗ x` → `g = mean(p)` no longer exact
- ⚠️ Need to update both forward (g→p) and backward (p→g) projections
- ⚠️ May require hyperparameter retuning (learning rates, initialization scales)
- ⚠️ Should maintain backward compatibility with existing trained models

**Research Questions:**

- Does random projection improve generalization?
- What sparsity level works best (10%, 15%, 20%)?
- Should projection be learnable or fixed?
- How does it affect zero-shot inference performance?

---

## Other TODOs

(Add other tasks below as needed)
