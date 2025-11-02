---
post_title: "torch_tem Data API Reference"
author1: "Borja Est"
post_slug: "docs-data-api"
microsoft_alias: "borja"
featured_image: ""
categories: ["documentation"]
tags: ["torch_tem", "api", "pytorch", "tem", "data"]
ai_note: "Generated with assistance from an AI; reviewed for accuracy."
summary: "API documentation for the data submodule: environments, policies, walks, shiny objects, synthetic and training pattern generators, and the Lightning data module."
post_date: "2025-11-02"
---

## Scope and conventions

- Module: `src/torch_tem/data`
- Audience: contributors and advanced users wiring data generation for TEM
- Shapes: `[T, B, ·]` denotes time-major sequences (time, then batch)
- Protocols: constructors may accept Protocols from `torch_tem.config.facets`
- All Pydantic models use v2 with type validation; tensors are PyTorch unless noted

## Imports

Prefer public package exports:

```python
from torch_tem.data import (
  Environment, Location, Action,
  PolicyGenerator,
  ShinyConfig, ShinyEnvironmentBuilder,
  Walk, WalkGenerator,
  TEMDataModule, InfiniteWalkDataset,
  SyntheticGridGenerator,
  PlaceCellPatternGenerator, GridCellPatternGenerator, PairedPatternGenerator,
)
```

## Environment and graph world

### Action (Pydantic)

- File: `src/torch_tem/data/environment.py`
- Fields:
  - `id: int` — non-negative action identifier
  - `probability: float` — selection probability in `[0, 1]`
  - `transition: List[float]` — distribution over next states (size `n_locations`)
- Validators:
  - `transition` must be a valid probability distribution (non-negative, sums to 1)

### Location (Pydantic)

- Fields:
  - `id: int` — location identifier
  - `observation: int` — observation ID present at this location
  - `actions: List[Action]` — available actions (non-empty)
  - `shiny: Optional[bool]` — marks shiny objects if present
- Validators:
  - `actions` list must be non-empty

### Environment

- Purpose: Load, validate, and query a graph-world environment
- Constructor: `Environment(env_spec: Union[str, Dict], randomize_observations: bool = False)`
  - `env_spec` accepts a JSON filepath or a dictionary with keys:
    `adjacency`, `locations`, `n_actions`, `n_locations`, `n_observations`
  - If `randomize_observations=True`, observations are shuffled after load
- Attributes:
  - `n_locations: int`, `n_observations: int`, `n_actions: int`
  - `adjacency: List[List[int]] | np.ndarray`
  - `locations: List[Location]`
- Methods:
  - `randomize_observations() -> None` — reshuffle observation IDs across locations
  - `shortest_paths() -> np.ndarray` — all-pairs shortest path matrix (cached)
  - `validate() -> bool` — structural checks (dimensions, connectivity, probabilities)

Example:

```python
env = Environment("envs/5x5.json", randomize_observations=False)
env.validate()
D = env.shortest_paths()  # [n_locations, n_locations]
```

## Policy generation

### PolicyGenerator

- File: `src/torch_tem/data/policies.py`
- Purpose: Produce location-conditional action policies for exploration and goals
- Constructor: `PolicyGenerator(environment: Environment)`
- Methods:
  - `random_policy() -> List[Location]`
    - Uniform over valid actions at each location
  - `distance_policy(goal_locations: Union[int, List[int]], beta: float = 1.0)
  -> List[Location]`
    - Softmax over negative graph distance to goals; disables self-actions at goals
  - `q_learning_policy(goal_locations: Union[int, List[int]],
  gamma: float = 0.9, beta: float = 1.0, n_iterations: int = 100)
  -> List[Location]`
    - Value iteration over transitions; softmax over Q-values to form policy
  - `mix_policies(policies: List[List[Location]], weights: List[float])
  -> List[Location]`
    - Weighted mixture for curriculum learning

Example:

```python
goal = 7
policies = PolicyGenerator(env)
pi = policies.distance_policy(goal, beta=2.0)
```

## Walks and batching

### Walk (Pydantic)

- File: `src/torch_tem/data/walks.py`
- Represents a trajectory through the environment
- Fields and shapes:
  - `observations: Tensor` — `[T, n_observations]` one-hot per step
  - `actions: Tensor` — `[T]` action indices
  - `locations: Tensor` — `[T]` location IDs
  - `shiny_markers: Optional[Tensor]` — `[T]` boolean; present if any shiny encountered
- `__len__()` returns `T` (number of steps)

### WalkGenerator

- Purpose: Sample walks using environment transitions and optional policies
- Constructor: `WalkGenerator(environment: Environment, repeat_bias: float = 2.0)`
  - `repeat_bias > 1` encourages repeating the previous action (straighter paths)
- Methods:
  - `generate_walk(walk_length: int, policy: Optional[List[Location]] = None) -> Walk`
    - If `policy=None`, uses environment action probabilities
  - `generate_walks(n_walks: int, walk_length: int, policy: Optional[List[Location]] = None)
  -> List[Walk]`
  - `generate_shiny_walk(walk_length: int, shiny_locations: List[int],
  shiny_policies: List[List[Location]], returns: int) -> Walk`
    - Sequentially approaches shiny objects; lingers `returns` steps per find
  - `generate_shiny_walks(n_walks: int, walk_length: int, shiny_locations: List[int],
  shiny_policies: List[List[Location]], returns: int) -> List[Walk]`
  - `batch_walks(walks: List[Walk]) -> Tuple[Tensor, Tensor, Tensor]`
    - Collates a list of walks into time-major tensors

Example:

```python
gen = WalkGenerator(env, repeat_bias=2.0)
walks = gen.generate_walks(n_walks=32, walk_length=50, policy=None)
obs, act, loc = gen.batch_walks(walks)  # obs: [T, B, n_obs], act: [T, B], loc: [T, B]
```

## Shiny objects (goals)

### ShinyConfig (Pydantic)

- File: `src/torch_tem/data/shiny.py`
- Fields:
  - `n: int` — number of shiny objects (> 0)
  - `returns: int` — steps to linger at a shiny before switching
  - `gamma: float` — discount factor for Q-learning in policy generation
  - `beta: float` — softmax temperature for policy extraction
  - `min_separation: float` — required graph-distance ratio between shinies
- Validation:
  - Ensures feasible separation for given `n` and `min_separation`

### ShinyEnvironmentBuilder

- Constructor: `ShinyEnvironmentBuilder(environment: Environment, shiny_config: ShinyConfig, policy_generator: PolicyGenerator)`
- Methods:
  - `place_shiny_objects() -> List[int]`
    - Rejection samples shiny locations with min separation (via shortest paths)
  - `mark_environment(shiny_locations: List[int]) -> Environment`
    - Marks shiny flags and deduplicates shiny observations across locations
  - `generate_shiny_policies(shiny_locations: List[int]) -> List[List[Location]]`
    - One distance-based policy per shiny for goal switching

Example:

```python
builder = ShinyEnvironmentBuilder(env, ShinyConfig(n=2, returns=15), PolicyGenerator(env))
shiny_locs = builder.place_shiny_objects()
env_marked = builder.mark_environment(shiny_locs)
shiny_policies = builder.generate_shiny_policies(shiny_locs)
```

## Lightning DataModule

### InfiniteWalkDataset (IterableDataset)

- File: `src/torch_tem/data/datamodule.py`
- Purpose: Infinite on-the-fly walk generation for streaming training
- Iteration: yields `Walk` objects indefinitely; supports epoch-aware policy via parent

### TEMDataModule (PyTorch Lightning)

- Purpose: Unified environment, policy, and walk generation with optional curriculum
- Constructor:
  - `TEMDataModule(
   env_spec: Union[str, Dict, Environment],
   batch_size: int,
   walk_length: int,
   shiny_config: Optional[ShinyConfig] = None,
   randomize_observations: bool = False,
   repeat_bias: float = 2.0,
   curriculum_schedule: Optional[Callable[[int], Dict]] = None,
   num_workers: int = 0,
 )`
- Behavior:
  - Builds/validates `Environment`; wires `PolicyGenerator` and `WalkGenerator`
  - Optional shiny setup via `ShinyEnvironmentBuilder`
  - Training: infinite stream through `InfiniteWalkDataset`
  - Validation: fixed number of walks (e.g., 10 batches) for consistent metrics
- Key methods:
  - `setup(stage: Optional[str] = None) -> None`
  - `train_dataloader() -> DataLoader` — infinite dataset, time-major collation
  - `val_dataloader() -> DataLoader` — fixed sample; single-process for reproducibility
  - `_collate_walks(walks: List[Walk]) -> Tuple[Tensor, Tensor, Tensor]`
    - Returns `(observations[T, B, n_obs], actions[T, B], locations[T, B])`
  - `_get_current_policy(epoch: int) -> Optional[List[Location]]`
    - Optional curriculum schedule hook
  - `generate_batch(epoch: int = 0) -> Tuple[Tensor, Tensor, Tensor]`
    - Convenience for generating one batch outside Lightning

Example:

```python
dm = TEMDataModule(env_spec="envs/5x5.json", batch_size=32, walk_length=50)
train_loader = dm.train_dataloader()
obs, act, loc = next(iter(train_loader))
```

## Synthetic grid patterns (testing/examples)

### SyntheticGridParams (Protocol)

- File: `src/torch_tem/data/synthetic.py`
- Required attributes:
  - `walk_length: int`
  - `n_g_calculated: List[int]`
  - `f_initial_extended: List[float]`

### SyntheticGridGenerator

- Purpose: Generate oscillatory, noisy grid-like activity per frequency module
- Constructor: `SyntheticGridGenerator(params: SyntheticGridParams, batch_size: int = 1, time_scale: float = 10.0, harmonic_weight: float = 0.3, noise_scale: float = 0.2)`
- Methods:
  - `generate() -> List[Tensor]` — returns `g_history[f]: [T, B, n_g[f]]`
  - `generate_batch() -> List[Tensor]` — alias for a single batch

## Training pattern generators (memory examples)

### PatternGeneratorParams (Protocol)

- File: `src/torch_tem/data/patterns.py`
- Required attributes:
  - `n_g_calculated: List[int]`
  - `n_p_calculated: List[int]`
  - `n_x_c: int`

### PlaceCellPatternGenerator

- Purpose: Produce sparse, smooth place cell activity for training Hebbian memory
- Constructor: `PlaceCellPatternGenerator(params: PatternGeneratorParams, sparsity: float = 0.1, noise_scale: float = 0.2)`
- Methods:
  - `generate(batch_size: int) -> Tensor` — place patterns `[B, sum(n_p_calculated)]`
  - `generate_sequence(n_steps: int, batch_size: int, temporal_smoothness: float = 0.7)
  -> Tensor` — time-sequenced place patterns `[T, B, ·]`

### GridCellPatternGenerator

- Purpose: Produce frequency-specific grid-like activity with phase diversity
- Constructor: `GridCellPatternGenerator(params: PatternGeneratorParams, noise_scale: float = 0.1)`
- Methods:
  - `generate(batch_size: int) -> List[Tensor]` — `g[f]: [B, n_g[f]]`
  - `generate_sequence(n_steps: int, batch_size: int, temporal_smoothness: float = 0.8)
  -> List[Tensor]` — `g_history[f]: [T, B, n_g[f]]`

### PairedPatternGenerator

- Purpose: Generate paired `(g, p)` data with optional correlation
- Constructor: `PairedPatternGenerator(params: PatternGeneratorParams, correlation: float = 0.3, place_sparsity: float = 0.1, noise_scale: float = 0.15)`
- Methods:
  - `generate(batch_size: int) -> Tuple[List[Tensor], Tensor]`
  - `generate_sequence(n_steps: int, batch_size: int, temporal_smoothness: float = 0.7)
  -> Tuple[List[Tensor], Tensor]`

## Public exports

From `src/torch_tem/data/__init__.py`:

- Classes/constructors:
  - `Environment`, `Location`, `Action`
  - `PolicyGenerator`
  - `ShinyConfig`, `ShinyEnvironmentBuilder`
  - `Walk`, `WalkGenerator`
  - `TEMDataModule`, `InfiniteWalkDataset`
  - `SyntheticGridGenerator`
  - `PlaceCellPatternGenerator`, `GridCellPatternGenerator`, `PairedPatternGenerator`

## Minimal end-to-end example

```python
# 1) Environment
env = Environment("envs/5x5.json")
policies = PolicyGenerator(env)
policy = policies.random_policy()

# 2) Walk sampling
wg = WalkGenerator(env, repeat_bias=2.0)
walks = wg.generate_walks(n_walks=16, walk_length=50, policy=policy)
obs, act, loc = wg.batch_walks(walks)

# 3) Lightning streaming
dm = TEMDataModule(env_spec=env, batch_size=32, walk_length=50)
train_loader = dm.train_dataloader()
obs_t, act_t, loc_t = next(iter(train_loader))
```

## Source index

- `src/torch_tem/data/environment.py`
- `src/torch_tem/data/policies.py`
- `src/torch_tem/data/walks.py`
- `src/torch_tem/data/shiny.py`
- `src/torch_tem/data/datamodule.py`
- `src/torch_tem/data/synthetic.py`
- `src/torch_tem/data/patterns.py`
- `src/torch_tem/data/__init__.py`
