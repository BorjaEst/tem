# torch_tem.data — API Reference

Synthetic graph-world environments, policy generation, walk sampling, and training data generation for the Temporal Experience Model (TEM).

## Overview

The data submodule provides a complete synthetic navigation environment system for TEM training and inference examples. It follows Protocol-based design principles for modularity and testability.

**Key capabilities:**

- Graph-world environment definition and validation
- Policy generation (random, distance-based, Q-learning, mixed)
- Walk trajectory sampling with goal-switching behavior
- Shiny object (reward location) configuration
- PyTorch Lightning DataModule with infinite on-the-fly generation
- Synthetic pattern generation for memory training and testing

**Design principles:**

- Pydantic v2 models for validation and serialization
- Protocol-based contracts for loose coupling
- On-the-fly generation (no pre-caching) for infinite training data
- Curriculum learning support via epoch-dependent policy mixing

---

## Environment System

### `torch_tem.data.environment`

Graph-world environment structure with discrete locations, actions, and transitions.

#### `Action`

```python
class Action(BaseModel):
    id: int                    # Action identifier (>= 0)
    probability: float         # Selection probability [0.0, 1.0]
    transition: List[float]    # Probability distribution over next states
```

**Purpose:** Represents a single action available at a location with its probability of selection and transition distribution to next states.

**Validation:**

- `transition` must sum to 1.0 ± 1e-6
- All probabilities must be non-negative

**Example:**

```python
action = Action(
    id=0,
    probability=0.25,
    transition=[0.9, 0.05, 0.05, 0.0]  # Mostly to state 0
)
```

---

#### `Location`

```python
class Location(BaseModel):
    id: int                      # Location identifier (>= 0)
    observation: int             # Observation ID at this location (>= 0)
    actions: List[Action]        # Available actions from this location
    shiny: Optional[bool]        # Whether location has shiny object
```

**Purpose:** Single location in the environment graph with observation, available actions, and optional shiny object marker.

**Validation:**

- `actions` list must be non-empty
- Action probabilities must sum to 1.0

**Example:**

```python
location = Location(
    id=5,
    observation=2,
    actions=[action1, action2, action3, action4],
    shiny=False
)
```

---

#### `Environment`

```python
class Environment:
    def __init__(
        self,
        env_spec: Union[str, Dict],
        randomize_observations: bool = False
    )
```

**Purpose:** Graph-world environment with locations and transitions. Manages environment structure loaded from JSON or generated programmatically.

**Attributes:**

- `n_locations: int` — Number of discrete locations
- `n_observations: int` — Number of unique sensory observations
- `n_actions: int` — Number of available actions per location
- `adjacency: List[List[float]]` — Adjacency matrix
- `locations: List[Location]` — List of Location Pydantic models

**Parameters:**

- `env_spec` — Path to JSON file or environment dictionary with keys: `'adjacency'`, `'locations'`, `'n_actions'`, `'n_locations'`, `'n_observations'`
- `randomize_observations` — Shuffle observation assignments after loading

**Methods:**

##### `randomize_observations()`

```python
def randomize_observations(self) -> None
```

Randomly shuffle observation assignments across locations.

##### `shortest_paths()`

```python
def shortest_paths(self) -> np.ndarray
```

Compute all-pairs shortest path distances using Dijkstra's algorithm.

**Returns:** `[n_locations, n_locations]` distance matrix (cached)

##### `validate()`

```python
def validate(self) -> bool
```

Comprehensive environment validation:

- Location count matches specification
- All location IDs are unique and sequential
- All observations are in valid range
- All actions have valid transitions
- Adjacency matrix is square and matches location count
- Graph is connected (no isolated components)

**Raises:** `ValueError` if validation fails

**Example:**

```python
# Load from JSON
env = Environment("envs/10x10.json")

# Load from dict
env_dict = {
    "n_locations": 16,
    "n_observations": 10,
    "n_actions": 4,
    "adjacency": [[...]],
    "locations": [...]
}
env = Environment(env_dict, randomize_observations=True)

# Query structure
distances = env.shortest_paths()
env.validate()
```

---

## Policy Generation

### `torch_tem.data.policies`

Action policy generation for navigation and goal-directed behavior.

#### `PolicyGenerator`

```python
class PolicyGenerator:
    def __init__(self, environment: Environment)
```

**Purpose:** Generates action policies for environment navigation with support for multiple policy types for different training phases.

**Policy types:**

- Random exploration (uniform over valid actions)
- Distance-based goal policies (softmax over graph distances)
- Q-learning policies (value iteration toward reward locations)
- Policy mixing for curriculum learning

**Methods:**

##### `random_policy()`

```python
def random_policy(self) -> List[Location]
```

Generate uniform distribution over valid actions at each location.

**Returns:** Locations with uniform action probabilities

##### `distance_policy()`

```python
def distance_policy(
    self,
    goal_locations: Union[int, List[int]],
    beta: float = 1.0
) -> List[Location]
```

Softmax policy based on graph distance to goals.

Computes policy that favors actions leading toward goal locations using softmax over negative distances. **Faster than Q-learning** but ignores transition probabilities.

**Parameters:**

- `goal_locations` — Target location(s) to approach
- `beta` — Inverse temperature for softmax (higher = more deterministic)

**Returns:** Locations with distance-based action probabilities

**Algorithm:**

1. Compute shortest path distances to all goals
2. For each location and action, calculate minimum distance to any goal
3. Apply softmax over negative distances: `π(a|s) ∝ exp(-β · d(s,a))`
4. Disable self-actions at goal locations

##### `q_learning_policy()`

```python
def q_learning_policy(
    self,
    goal_locations: Union[int, List[int]],
    gamma: float = 0.9,
    beta: float = 1.0,
    n_iterations: int = 100
) -> List[Location]
```

Q-learned policy via value iteration.

Computes optimal policy toward reward locations using value iteration on the Bellman equation. **Accounts for transition probabilities** (more accurate than distance policy).

**Parameters:**

- `goal_locations` — Reward location(s)
- `gamma` — Discount factor for future rewards [0.0, 1.0]
- `beta` — Softmax temperature
- `n_iterations` — Value iteration steps (default 100)

**Returns:** Locations with Q-learned action probabilities

**Algorithm (Bellman equation):**

```
Q(s, a) ← r(s) + γ · Σ_s' P(s'|s,a) · max_a' Q(s', a')
π(a|s) = softmax_β(Q(s, :))
```

##### `mix_policies()`

```python
def mix_policies(
    self,
    policies: List[List[Location]],
    weights: List[float]
) -> List[Location]
```

Weighted mixture of multiple policies for curriculum learning.

**Parameters:**

- `policies` — List of policy sets to mix
- `weights` — Mixture weights (must sum to 1.0)

**Returns:** Mixed policy with weighted action probabilities

**Example:**

```python
env = Environment("envs/5x5.json")
policy_gen = PolicyGenerator(env)

# Random exploration
random_pol = policy_gen.random_policy()

# Goal-directed navigation
goal_pol = policy_gen.distance_policy(goal_locations=[10, 15], beta=2.0)

# Optimal policy with Q-learning
q_pol = policy_gen.q_learning_policy(goal_locations=[10], gamma=0.95, beta=1.5)

# Curriculum: 70% goal-directed, 30% random
mixed_pol = policy_gen.mix_policies([goal_pol, random_pol], weights=[0.7, 0.3])
```

---

## Walk Generation

### `torch_tem.data.walks`

Walk sequence generation and batching for training and evaluation.

#### `Walk`

```python
class Walk(BaseModel):
    observations: Tensor      # [walk_length, n_observations] one-hot tensors
    actions: Tensor           # [walk_length] action indices
    locations: Tensor         # [walk_length] location IDs (for analysis)
    shiny_markers: Optional[Tensor]  # [walk_length] boolean shiny flags
```

**Purpose:** Single walk trajectory through the environment.

**Methods:**

##### `__len__()`

Returns walk length (number of steps).

---

#### `WalkGenerator`

```python
class WalkGenerator:
    def __init__(
        self,
        environment: Environment,
        repeat_bias: float = 2.0
    )
```

**Purpose:** Generates walk sequences from environment and policy with repeat-action bias for straight-line movement.

**Parameters:**

- `environment` — Environment to generate walks in
- `repeat_bias` — Multiplicative bias for repeating previous action (encourages straight walks)

**Methods:**

##### `generate_walk()`

```python
def generate_walk(
    self,
    walk_length: int,
    policy: Optional[List[Location]] = None
) -> Walk
```

Generate single walk from optional policy.

**Parameters:**

- `walk_length` — Number of steps
- `policy` — Optional location-specific policies (uses environment default if None)

**Returns:** Walk trajectory

**Algorithm:**

1. Start at random location
2. For each step:
   - Apply repeat bias to previous action
   - Sample action from (biased) policy
   - Sample transition to next location
   - Record observation, action, location

##### `generate_walks()`

```python
def generate_walks(
    self,
    n_walks: int,
    walk_length: int,
    policy: Optional[List[Location]] = None
) -> List[Walk]
```

Generate multiple independent walks.

##### `generate_shiny_walk()`

```python
def generate_shiny_walk(
    self,
    walk_length: int,
    shiny_locations: List[int],
    shiny_policies: List[List[Location]],
    returns: int
) -> Walk
```

Generate walk with automatic goal switching.

Agent approaches shiny objects sequentially, switching goals after reaching each object and lingering for `returns` steps.

**Parameters:**

- `walk_length` — Total walk steps
- `shiny_locations` — List of shiny object locations
- `shiny_policies` — Goal-directed policies for each shiny object
- `returns` — Steps to linger at each shiny before switching goals

**Returns:** Walk with goal-switching behavior and shiny markers

**Algorithm:**

1. Pick random initial shiny target
2. Navigate toward current target using its policy
3. Upon reaching target:
   - Linger for `returns` steps
   - Switch to next random shiny target
4. Repeat until walk_length steps complete

##### `generate_shiny_walks()`

```python
def generate_shiny_walks(
    self,
    n_walks: int,
    walk_length: int,
    shiny_locations: List[int],
    shiny_policies: List[List[Location]],
    returns: int
) -> List[Walk]
```

Generate multiple shiny walks.

##### `batch_walks()`

```python
def batch_walks(
    self,
    walks: List[Walk]
) -> Tuple[Tensor, Tensor, Tensor]
```

Collate walks into batched tensors.

**Returns:** `(observations, actions, locations)` with batch dimension

**Example:**

```python
env = Environment("envs/5x5.json")
policy_gen = PolicyGenerator(env)
walk_gen = WalkGenerator(env, repeat_bias=2.0)

# Generate exploration walks
policy = policy_gen.random_policy()
walks = walk_gen.generate_walks(n_walks=32, walk_length=100, policy=policy)

# Generate goal-switching walks
shiny_locs = [5, 10, 15]
shiny_pols = [policy_gen.distance_policy(loc) for loc in shiny_locs]
shiny_walks = walk_gen.generate_shiny_walks(
    n_walks=32,
    walk_length=100,
    shiny_locations=shiny_locs,
    shiny_policies=shiny_pols,
    returns=15
)

# Batch for training
obs_batch, act_batch, loc_batch = walk_gen.batch_walks(walks)
```

---

## Shiny Objects (Reward Locations)

### `torch_tem.data.shiny`

Shiny object configuration and placement for goal-directed behavior.

#### `ShinyConfig`

```python
class ShinyConfig(BaseModel):
    n: int                     # Number of shiny objects (> 0)
    returns: int               # Steps to linger at shiny object (> 0)
    gamma: float = 0.9         # Discount factor for Q-learning [0.0, 1.0]
    beta: float = 1.0          # Softmax temperature (> 0.0)
    min_separation: float = 0.3  # Minimum distance ratio [0.0, 1.0]
```

**Purpose:** Configuration for shiny object environments. Shiny objects are special reward locations that drive goal-directed behavior with automatic goal switching during walks.

**Validation:**

- `n > 1` and `min_separation > 0.9` is infeasible (cannot place objects)

**Example:**

```python
shiny_config = ShinyConfig(
    n=3,
    returns=15,
    gamma=0.9,
    beta=1.5,
    min_separation=0.4
)
```

---

#### `ShinyEnvironmentBuilder`

```python
class ShinyEnvironmentBuilder:
    def __init__(
        self,
        environment: Environment,
        shiny_config: ShinyConfig,
        policy_generator: PolicyGenerator
    )
```

**Purpose:** Augments environments with shiny object placement. Handles shiny object placement with minimum separation constraints, observation deduplication, and shiny-directed policy generation.

**Methods:**

##### `place_shiny_objects()`

```python
def place_shiny_objects(self) -> List[int]
```

Place shiny objects with minimum separation constraint.

Uses graph distances to ensure shiny objects are sufficiently spread out. Sampling is rejection-based: repeatedly sample locations until separation constraint is satisfied.

**Returns:** Location IDs designated as shiny

**Raises:** `RuntimeError` if unable to place after 1000 attempts

**Algorithm:**

1. For each shiny object:
   - Sample random location
   - Check graph distance to all existing shiny locations
   - Accept if `min_distance >= max_distance * min_separation`
   - Retry up to 1000 times

##### `mark_environment()`

```python
def mark_environment(
    self,
    shiny_locations: List[int]
) -> Environment
```

Update environment with shiny markers and deduplicate observations.

Marks shiny locations and ensures shiny observations don't appear at non-shiny locations (makes shiny objects distinctive).

**Parameters:**

- `shiny_locations` — List of location IDs to mark as shiny

**Returns:** Updated environment with shiny markers

**Algorithm:**

1. Identify shiny observations
2. Replace shiny observations at non-shiny locations with non-shiny alternatives
3. Mark shiny locations with `shiny=True`

##### `generate_shiny_policies()`

```python
def generate_shiny_policies(
    self,
    shiny_locations: List[int]
) -> List[List[Location]]
```

Generate goal-directed policy for each shiny object.

**Returns:** One distance-based policy per shiny object for goal switching

**Example:**

```python
env = Environment("envs/10x10.json")
policy_gen = PolicyGenerator(env)
shiny_config = ShinyConfig(n=3, returns=15, min_separation=0.3)

builder = ShinyEnvironmentBuilder(env, shiny_config, policy_gen)

# Place and configure
shiny_locs = builder.place_shiny_objects()
env = builder.mark_environment(shiny_locs)
shiny_pols = builder.generate_shiny_policies(shiny_locs)
```

---

## PyTorch Lightning DataModule

### `torch_tem.data.datamodule`

PyTorch Lightning DataModule for TEM training with infinite on-the-fly generation.

#### `InfiniteWalkDataset`

```python
class InfiniteWalkDataset(IterableDataset):
    def __init__(self, datamodule: "TEMDataModule")
```

**Purpose:** Infinite stream of walks for training. Generates walks on-the-fly without pre-caching, enabling infinite training data with curriculum learning support.

**Methods:**

##### `__iter__()`

Yield walks indefinitely using datamodule's generation methods.

---

#### `TEMDataModule`

```python
class TEMDataModule(L.LightningDataModule):
    def __init__(
        self,
        env_spec: Union[str, Dict, Environment],
        batch_size: int,
        walk_length: int,
        shiny_config: Optional[ShinyConfig] = None,
        randomize_observations: bool = False,
        repeat_bias: float = 2.0,
        curriculum_schedule: Optional[Callable[[int], Dict]] = None,
        num_workers: int = 0
    )
```

**Purpose:** PyTorch Lightning DataModule for TEM training. Generates infinite stream of walks without pre-caching. Supports curriculum learning via epoch-dependent policy mixing. Integrates environment, policy, and walk generation.

**Parameters:**

- `env_spec` — Environment specification (path, dict, or Environment instance)
- `batch_size` — Walks per batch
- `walk_length` — Steps per walk
- `shiny_config` — Optional shiny object configuration
- `randomize_observations` — Shuffle observation assignments
- `repeat_bias` — Action repeat bias for straight-line movement
- `curriculum_schedule` — Optional `epoch -> policy_params` mapping function
- `num_workers` — Number of dataloader workers

**Attributes:**

- `env: Environment` — Built environment
- `policy_gen: PolicyGenerator` — Policy generator
- `walk_gen: WalkGenerator` — Walk generator
- `shiny_locations: Optional[List[int]]` — Shiny object locations (if enabled)
- `shiny_policies: Optional[List[List[Location]]]` — Shiny policies (if enabled)

**Methods:**

##### `setup()`

```python
def setup(self, stage: Optional[str] = None) -> None
```

Prepare data for training/validation/testing. No setup needed for on-the-fly generation.

##### `train_dataloader()`

```python
def train_dataloader(self) -> DataLoader
```

Infinite walk generation for training.

**Returns:** DataLoader with infinite walk stream

##### `val_dataloader()`

```python
def val_dataloader(self) -> DataLoader
```

Fixed validation set for consistent metrics.

Generates a fixed set of walks (10 batches) for validation to ensure consistent metric tracking across epochs.

**Returns:** DataLoader with fixed validation walks

##### `generate_batch()`

```python
def generate_batch(self, epoch: int = 0) -> Tuple[Tensor, Tensor, Tensor]
```

Generate single batch of walks with epoch-dependent policy.

**Parameters:**

- `epoch` — Current training epoch (for curriculum schedule)

**Returns:** `(observations, actions, locations)` batched tensors

**Curriculum Learning:**

The `curriculum_schedule` parameter accepts a callable that maps epoch numbers to policy parameters:

```python
def curriculum(epoch: int) -> Dict:
    """Progressive curriculum: random → distance → Q-learning."""
    if epoch < 10:
        return {"type": "random"}
    elif epoch < 30:
        return {"type": "distance", "beta": 1.0 + epoch * 0.05}
    else:
        return {"type": "q_learning", "gamma": 0.9, "beta": 2.0}

datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    curriculum_schedule=curriculum
)
```

**Example:**

```python
# Basic usage
datamodule = TEMDataModule(
    env_spec="envs/5x5.json",
    batch_size=32,
    walk_length=100,
    repeat_bias=2.0
)

# With shiny objects
shiny_config = ShinyConfig(n=3, returns=15, min_separation=0.3)
datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    shiny_config=shiny_config
)

# With curriculum learning
def curriculum(epoch):
    if epoch < 20:
        return {"type": "random"}
    else:
        return {"type": "distance", "goal_locations": [25, 50], "beta": 1.5}

datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    curriculum_schedule=curriculum
)

# Use with Lightning Trainer
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

---

## Synthetic Pattern Generation

### `torch_tem.data.synthetic`

Synthetic grid cell activity generation for testing and examples.

#### `SyntheticGridParams` (Protocol)

```python
class SyntheticGridParams(Protocol):
    walk_length: int
    n_g_calculated: List[int]
    f_initial_extended: List[float]
```

**Purpose:** Protocol for synthetic grid cell generation configuration. Any object implementing these attributes can be used (e.g., `Parameters` model).

---

#### `SyntheticGridGenerator`

```python
class SyntheticGridGenerator:
    def __init__(
        self,
        params: SyntheticGridParams,
        batch_size: int = 1,
        time_scale: float = 10.0,
        harmonic_weight: float = 0.3,
        noise_scale: float = 0.2
    )
```

**Purpose:** Generates synthetic grid cell activity patterns. Creates oscillating patterns with frequency-dependent dynamics to simulate grid cell responses during spatial navigation. Useful for testing inference components without requiring full TEM training.

**Pattern characteristics:**

- Primary oscillation at specified frequency
- Secondary harmonic for richer dynamics
- Gaussian noise for biological realism
- Random phase offsets for variation across cells

**Parameters:**

- `params` — Configuration providing `walk_length`, `n_g_calculated`, `f_initial_extended`
- `batch_size` — Batch size for generation
- `time_scale` — Time scaling factor for oscillations
- `harmonic_weight` — Weight for second harmonic component [0.0, 1.0]
- `noise_scale` — Scale of Gaussian noise added to patterns

**Methods:**

##### `generate()`

```python
def generate(self) -> List[Tensor]
```

Generate synthetic grid cell activity patterns.

**Returns:** List of `[T, B, n_g[f]]` tensors with synthetic grid patterns (one per frequency)

**Algorithm:**

```python
t = linspace(0, time_scale * frequency, walk_length)
phases = random_phases(batch_size, n_g[f])
pattern = sin(t + phases) + harmonic_weight * sin(2*t + phases/2)
pattern += noise_scale * gaussian_noise()
```

##### `generate_batch()`

```python
def generate_batch(self) -> List[Tensor]
```

Convenience method for generating one batch (same as `generate()`).

**Example:**

```python
from pydantic import BaseModel

class Config(BaseModel):
    walk_length: int = 50
    n_g_calculated: List[int] = [12, 10, 8]
    f_initial_extended: List[float] = [0.1, 0.3, 0.9]

config = Config()
generator = SyntheticGridGenerator(config, batch_size=4)
g_history = generator.generate()

for f, g_f in enumerate(g_history):
    print(f"Frequency {f}: shape={g_f.shape}")
    # Output: Frequency 0: shape=torch.Size([50, 4, 12])
```

---

### `torch_tem.data.patterns`

Pattern generation for memory training and generation examples.

#### `PatternGeneratorParams` (Protocol)

```python
class PatternGeneratorParams(Protocol):
    n_g_calculated: List[int]
    n_p_calculated: List[int]
    n_x_c: int
```

**Purpose:** Protocol for pattern generation configuration. Components implementing this protocol provide the necessary parameters for generating synthetic training patterns for memory networks.

---

#### `PlaceCellPatternGenerator`

```python
class PlaceCellPatternGenerator:
    def __init__(
        self,
        params: PatternGeneratorParams,
        sparsity: float = 0.1,
        noise_scale: float = 0.2
    )
```

**Purpose:** Generates synthetic place cell activity patterns for memory training. Creates realistic hippocampal place cell patterns with sparse activation, smooth probability distributions, and biological variability.

**Pattern characteristics:**

- Sparse activation (few cells active at once)
- Smooth probability distributions
- Biological variability via noise
- Optional location-specific structure

**Parameters:**

- `params` — Configuration providing dimensions
- `sparsity` — Fraction of active cells [0.0, 1.0]
- `noise_scale` — Scale of Gaussian noise

**Methods:**

##### `generate()`

```python
def generate(self, batch_size: int) -> Tensor
```

Generate synthetic place cell patterns.

**Returns:** `[B, n_p_total]` tensor with place cell activity

##### `generate_sequence()`

```python
def generate_sequence(
    self,
    n_steps: int,
    batch_size: int,
    temporal_smoothness: float = 0.7
) -> Tensor
```

Generate temporally smooth sequence of place cell patterns.

**Parameters:**

- `n_steps` — Sequence length
- `batch_size` — Batch size
- `temporal_smoothness` — Smoothing factor [0.0, 1.0] (higher = smoother)

**Returns:** `[T, B, n_p_total]` tensor

---

#### `GridCellPatternGenerator`

```python
class GridCellPatternGenerator:
    def __init__(
        self,
        params: PatternGeneratorParams,
        noise_scale: float = 0.1
    )
```

**Purpose:** Generates synthetic grid cell activity patterns for memory training. Creates realistic entorhinal grid cell patterns with frequency-specific dimensions and normalized probability distributions.

**Methods:**

##### `generate()`

```python
def generate(self, batch_size: int) -> List[Tensor]
```

Generate synthetic grid cell patterns.

**Returns:** List of `[B, n_g[f]]` tensors (one per frequency)

##### `generate_sequence()`

```python
def generate_sequence(
    self,
    n_steps: int,
    batch_size: int,
    temporal_smoothness: float = 0.8
) -> List[Tensor]
```

Generate temporally smooth sequences.

**Returns:** List of `[T, B, n_g[f]]` tensors

---

#### `PairedPatternGenerator`

```python
class PairedPatternGenerator:
    def __init__(
        self,
        params: PatternGeneratorParams,
        correlation: float = 0.3,
        place_sparsity: float = 0.1,
        noise_scale: float = 0.15
    )
```

**Purpose:** Generates paired (g, p) patterns for memory training. Creates associated grid cell and place cell patterns with realistic structure and optional correlation.

**Parameters:**

- `params` — Configuration
- `correlation` — Correlation between g and p patterns [0.0, 1.0]
- `place_sparsity` — Place cell sparsity
- `noise_scale` — Noise scale

**Methods:**

##### `generate()`

```python
def generate(
    self,
    batch_size: int
) -> Tuple[List[Tensor], Tensor]
```

Generate paired (g, p) patterns.

**Returns:** `(g_patterns, p_patterns)` where:

- `g_patterns` is list of `[B, n_g[f]]` tensors
- `p_patterns` is `[B, n_p_total]` tensor

##### `generate_sequence()`

```python
def generate_sequence(
    self,
    n_steps: int,
    batch_size: int,
    temporal_smoothness: float = 0.7
) -> Tuple[List[Tensor], Tensor]
```

Generate temporally smooth paired sequences.

**Returns:** `(g_sequence, p_sequence)` with temporal dimension

**Example:**

```python
from pydantic import BaseModel

class Config(BaseModel):
    n_g_calculated: List[int] = [12, 10, 8]
    n_p_calculated: List[int] = [120, 100, 80]
    n_x_c: int = 10

config = Config()

# Generate paired patterns for Hebbian learning
paired_gen = PairedPatternGenerator(config, correlation=0.4)
g_batch, p_batch = paired_gen.generate(batch_size=32)

# Use for memory training
# M[f] = hebbian_update(M[f], g_batch[f], p_batch)
```

---

## Module Exports

### `torch_tem.data.__init__`

**Exported classes:**

**Environment system:**

- `Environment`
- `Location`
- `Action`

**Policy generation:**

- `PolicyGenerator`

**Shiny objects:**

- `ShinyConfig`
- `ShinyEnvironmentBuilder`

**Walk generation:**

- `Walk`
- `WalkGenerator`

**DataModule:**

- `TEMDataModule`
- `InfiniteWalkDataset`

**Synthetic generation:**

- `SyntheticGridGenerator`
- `PlaceCellPatternGenerator`
- `GridCellPatternGenerator`
- `PairedPatternGenerator`

**Usage:**

```python
from torch_tem.data import (
    Environment,
    PolicyGenerator,
    WalkGenerator,
    ShinyConfig,
    TEMDataModule,
    SyntheticGridGenerator,
)
```

---

## Common Workflows

### 1. Basic Training Setup

```python
from torch_tem.data import Environment, TEMDataModule

# Setup environment and datamodule
datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    repeat_bias=2.0
)

# Use with Lightning
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

### 2. Custom Policy for Inference

```python
from torch_tem.data import Environment, PolicyGenerator, WalkGenerator

# Load environment
env = Environment("envs/5x5.json")

# Generate goal-directed policy
policy_gen = PolicyGenerator(env)
policy = policy_gen.distance_policy(goal_locations=[10, 15], beta=2.0)

# Generate walks
walk_gen = WalkGenerator(env)
walks = walk_gen.generate_walks(n_walks=10, walk_length=50, policy=policy)
```

### 3. Shiny Object Training

```python
from torch_tem.data import TEMDataModule, ShinyConfig

# Configure shiny objects
shiny_config = ShinyConfig(
    n=3,
    returns=15,
    gamma=0.9,
    beta=1.5,
    min_separation=0.4
)

# Create datamodule with automatic goal switching
datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    shiny_config=shiny_config
)
```

### 4. Curriculum Learning

```python
from torch_tem.data import TEMDataModule

def curriculum_schedule(epoch: int) -> Dict:
    """Progressive difficulty: random → goal-directed → optimal."""
    if epoch < 10:
        return {"type": "random"}
    elif epoch < 30:
        # Gradually increase determinism
        beta = 0.5 + (epoch - 10) * 0.075
        return {"type": "distance", "goal_locations": [25], "beta": beta}
    else:
        return {"type": "q_learning", "gamma": 0.95, "beta": 2.0}

datamodule = TEMDataModule(
    env_spec="envs/10x10.json",
    batch_size=32,
    walk_length=100,
    curriculum_schedule=curriculum_schedule
)
```

### 5. Testing with Synthetic Patterns

```python
from torch_tem.data import SyntheticGridGenerator
from torch_tem.config import Parameters

# Use real config
params = Parameters(
    walk_length=50,
    n_g_subsampled=[12, 10, 8],
    f_initial=[0.99, 0.3, 0.09]
)

# Generate synthetic grid activity
generator = SyntheticGridGenerator(params, batch_size=4)
g_history = generator.generate()

# Test inference components without full training
# inference_module.process(g_history)
```

### 6. Memory Training Patterns

```python
from torch_tem.data import PairedPatternGenerator
from torch_tem.config import Parameters

params = Parameters(
    n_g_subsampled=[12, 10, 8],
    n_x_c=10
)

# Generate training patterns
pattern_gen = PairedPatternGenerator(params, correlation=0.3)
g_batch, p_batch = pattern_gen.generate(batch_size=100)

# Train Hebbian memory
# memory.update(g_batch, p_batch)
```

---

## Design Notes

### Protocol-Based Architecture

All generator classes accept Protocol-typed parameters instead of concrete `Parameters` class. This enables:

- **Testing:** Use lightweight fakes implementing only required attributes
- **Modularity:** Components depend on minimal interfaces
- **Type safety:** Structural typing ensures compatibility

Example:

```python
from pydantic import BaseModel

class TestConfig(BaseModel):
    walk_length: int = 50
    n_g_calculated: List[int] = [12, 10]
    f_initial_extended: List[float] = [0.1, 0.3]

# Works with SyntheticGridGenerator (implements SyntheticGridParams)
generator = SyntheticGridGenerator(TestConfig())
```

### On-the-Fly Generation

The DataModule generates walks **on-demand** during training:

- **Memory efficient:** No pre-cached datasets
- **Infinite data:** Never repeat exact trajectories
- **Curriculum learning:** Policy adapts dynamically per epoch
- **Fast iteration:** No disk I/O overhead

### Validation and Guarantees

All Pydantic models provide runtime validation:

- `Action.transition` sums to 1.0
- `Environment.validate()` ensures graph connectivity
- `ShinyConfig` checks placement feasibility
- Policy probabilities are normalized

### Extensibility

**Add new policy types:**

```python
class PolicyGenerator:
    def custom_policy(self, **kwargs) -> List[Location]:
        # Implement custom action probability computation
        pass
```

**Add new environments:**

```python
# JSON format
{
    "n_locations": 25,
    "n_observations": 15,
    "n_actions": 4,
    "adjacency": [[...]],
    "locations": [
        {"id": 0, "observation": 3, "actions": [...]}
    ]
}
```

---

## See Also

- **`torch_tem.config.parameters`** — Full `Parameters` model satisfying all Protocols
- **`torch_tem.config.facets`** — Complete Protocol definitions
- **`torch_tem.model.tem`** — TEM orchestrator using data components
- **`examples/`** — Complete usage examples

---

**Last updated:** 2025-11-02  
**Submodule version:** Compatible with torch_tem design.md
