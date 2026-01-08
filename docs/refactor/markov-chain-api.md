# Markov Chain API for TEM

## Overview

The TEM model has been refactored to cleanly separate state transitions, observations, and predictions following a standard Markov chain pattern. This makes the information flow explicit and aligns with the theoretical formulation.

## Architecture

### Data Structures

**TEMState**: Represents the complete latent state at timestep `t`

- `M`: Hebbian memory matrices
- `lec_state`: LEC temporal filtering state (sensory feature cells)
- `mec_state`: MEC grid cell state (contains `g`, `g_gen`, `g_path`)
- `g_inf`: Inferred abstract location (corrected grid cells)
- `p_inf`: Inferred grounded location (corrected place cells)
- `p_inf_x`: Place cells from sensory retrieval (for loss computation)

**TEMPrediction**: Contains all predictive outputs for loss computation

- `o_hat`: Sensory predictions from 3 pathways (tuple of 3 tensors)
- `o_logits`: Logits for loss computation (tuple of 3 tensors)
- `p_gen`: Retrieved place cells from `g_inf` (for loss)

### Core Methods

The model now exposes four core methods that implement the Markov chain flow:

#### 1. `transition(mec_state, a_prev, locations, device) -> MECState`

**Purpose**: Action-driven state transition (pure MEC path integration)

**Biological analog**: Grid cell path integration without sensory input

**Computes**: $S_{t|t-1} = f(S_{t-1|t-1}, a_{t-1})$

**Returns**:

- Updated `MECState` with:
  - `g_path`: Path integration prior (mean, uncertainty) - `Transition` type
  - `g_gen`: Ancestral prediction for generative pathway

**Note**: This is the "predict" step in predict-update filters. It produces a _prior_ estimate before seeing the new observation.

#### 2. `observe(o, locations, M_prev, lec_state, mec_state) -> dict`

**Purpose**: Observation-driven state correction

**Biological analog**: Landmark/memory correction of grid cells, sensory filtering

**Computes**: $S_{t|t} = h(S_{t|t-1}, o_t)$

**Combines**:

- LEC: Temporal filtering of sensory input ($o_t \to c \to x$)
- HPC: Memory retrieval from sensory input ($x \to p_x$)
- MEC: Precision-weighted correction of grid cells ($g_{path} + p_x \to g_{inf}$)

**Returns** dictionary with:

- `lec_state`: Updated LEC state
- `g_inf`: Inferred (corrected) grid cells - **this is "what MEC represents now"**
- `p_inf`: Inferred place cells
- `p_inf_x`: Place cells from sensory retrieval (for loss)

**Note**: This is the "update" step in predict-update filters. It produces the _posterior_ estimate after incorporating the observation.

#### 3. `predict(M_prev, p_inf, g_inf, g_gen) -> TEMPrediction`

**Purpose**: Generate predictions from corrected state

**Computes**: $\hat{o}_t = g(S_{t|t})$

**Generates** sensory predictions from three pathways:

1. From inferred place cells: $p_{inf} \to \hat{o}$
2. From inferred grid cells via memory: $g_{inf} \to p \to \hat{o}$
3. From generated grid cells via memory: $g_{gen} \to p \to \hat{o}$

**Returns**: `TEMPrediction` with `o_hat`, `o_logits`, `p_gen`

**Note**: All three pathways are used for training (different loss terms test different aspects of the model).

#### 4. `update_memory(M_prev, p_inf, p_inf_x, p_gen) -> List[Tensor]`

**Purpose**: Hebbian memory update

**Computes**: $M_{t+1}$ after predictions are made

**Why after predictions?** To avoid a trivial "write then immediately read the same thing" shortcut during training.

**Returns**: Updated memory matrices `[M_gen, M_inf]` (M_inf only if `use_p_inf=True`)

## Usage Example

### Standard Markov Chain Flow

```python
from torch_tem.core import TEMModel, Parameters

# Create model
params = Parameters()
model = TEMModel(params)
model.set_runtime_hyperparams(eta=0.1, hebbian_decay=0.8, p2g_scale_offset=0.5)

# Initialize state
state = model.init_iteration(locations, o_0, a_0, M=None)

# Process timestep
for o_t, a_t, locations_t in walk:
    # 1. Transition: MEC path integration (action-driven, no observation)
    mec_state = model.transition(state.mec_state, a_prev, locations_t, device)

    # 2. Observe: LEC/HPC/MEC correction (observation-driven)
    obs_dict = model.observe(o_t, locations_t, state.M, state.lec_state, mec_state)

    # Update mec_state.g for next transition (legacy parity)
    mec_state.g = obs_dict["g_inf"]

    # 3. Predict: Generate predictions from corrected state
    prediction = model.predict(state.M, obs_dict["p_inf"], obs_dict["g_inf"], mec_state.g_gen)

    # 4. Update memory (after predictions to avoid write-then-read shortcut)
    M_next = model.update_memory(state.M, obs_dict["p_inf"], obs_dict["p_inf_x"], prediction.p_gen)

    # Compute loss
    loss = loss_fn(o_t, prediction, obs_dict, mec_state)

    # Update state for next iteration
    state.M = M_next
    state.lec_state = obs_dict["lec_state"]
    state.mec_state = mec_state
    a_prev = a_t
```

### Legacy API (Still Supported)

The original `forward()` method is preserved for backward compatibility:

```python
M, mec_state, p_gen, x_gen, o_logits, lec_state, g_inf, p_inf, p_inf_x = model.forward(
    o, locations, a_prev, M_prev, lec_state, mec_state
)
```

This internally calls the new methods in the correct order.

## Key Insights

### What MEC "represents biologically"

From the user's notes and the refactor:

- **`g_gen`** (from `mec_state.g_gen`): Pure path integration prediction (ancestral/generative branch)
- **`g_path`** (from `mec_state.g_path`): Path integration prior with uncertainty
- **`g_inf`**: **Most plausible "what MEC represents right now"** - includes path integration _plus_ landmark/memory correction

**Recommendation**: If you need a single MEC output, use **`g_inf`** from the `observe()` step.

### Time Indexing Convention

Standard state-space model: "State at t predicts observation at t+1"

In TEM training:

- We use **state(t) to reconstruct observation(t)** (current state explains current observation)
- This is fine because we're learning a world model, not forecasting the future
- The three prediction pathways test different aspects: direct inference, memory retrieval, and transition dynamics

### Separation of Concerns

| Component         | Input       | Output          | Biological Analog                      |
| ----------------- | ----------- | --------------- | -------------------------------------- |
| `transition()`    | action      | grid prior      | Path integration (no sensory input)    |
| `observe()`       | observation | corrected state | Landmark correction, sensory filtering |
| `predict()`       | state       | predictions     | Generating expectations                |
| `update_memory()` | state       | memory          | Hebbian plasticity                     |

This clean separation makes it easy to:

- Test components independently
- Understand information flow
- Modify individual components without breaking others
- Add new inference or prediction pathways

## Future Extensions

With this architecture, you can easily:

1. **Add new inference pathways**: Modify `observe()` to include additional correction sources
2. **Change prediction targets**: Modify `predict()` to generate different types of predictions
3. **Implement alternative memory systems**: Replace `update_memory()` with different plasticity rules
4. **Test ablations**: Remove specific pathways by modifying the relevant method
