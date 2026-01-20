"""Type definitions for the Tolman-Eichenbaum Machine.

This module defines the core types used throughout the TEM implementation,
grounded in the theoretical framework of the Tolman-Eichenbaum Machine.
These types abstract implementation details and provide clear semantics for
multi-scale spatial representations, Hebbian memory, and belief propagation.

Architecture Note
-----------------
This module is **dependency-free** within torch_tem:
    - No imports from other torch_tem modules
    - No Pydantic models (those live in settings.py)
    - Pure type definitions using only standard library, torch, and dataclasses
    - Provides foundation for settings.py and other modules

Type Hierarchy
--------------
Mathematical Primitives:
    - Vector: 1D or 2D tensor representing a single code or batch of codes
    - Matrix: 2D tensor representing connection weights or transformations

Multi-Scale Representations:
    - MultiScaleCode: Hierarchical representation across frequency modules
    - AbstractLocation: Abstract spatial code (g) in factorized representation
    - GroundedLocation: Grounded place cell code (p) from memory retrieval

Sensory Input:
    - Observation: Ground-truth sensory observation (o) as single tensor

Memory Structures:
    - HebbianMemory: Attractor network connection weights
    - MemoryState: Complete memory state (supports single/dual memory)

Data Flow:
    - StepInput: Inputs for a single timestep iteration
    - TEMState: Outputs from a single timestep iteration
    - Trajectory: Sequence of inputs forming a walk or episode
    - Losses: All loss components from a single timestep

Theory References
-----------------
The type system follows the TEM architecture:
    1. Sensory observations (o) → Sensory inference
    2. Abstract locations (g) → Multi-scale grid codes
    3. Grounded locations (p) → Place cell activations
    4. Hebbian memory (M) → Attractor dynamics
    5. Generative/Inference loops → Belief propagation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias

from torch import Tensor

# =============================================================================
# Mathematical Primitives
# =============================================================================

Vector = Tensor
"""A 1D or batched 2D tensor representing a neural population code.

Shape conventions:
    - Unbatched: (n_cells,)
    - Batched: (batch_size, n_cells)
"""

Matrix = Tensor
"""A 2D tensor representing connection weights or transformations.

Shape conventions:
    - Hebbian weights: (n_cells_pre, n_cells_post)
    - Projection matrix: (input_dim, output_dim)
"""


# =============================================================================
# Multi-Scale Representations
# =============================================================================

MultiScaleCode = List[Vector]
"""Hierarchical representation across multiple frequency modules.

In TEM, both abstract locations (g) and grounded locations (p) are
represented as multi-scale codes, where each frequency module operates
at a different spatial scale.

Structure:
    - Length: n_freq (number of frequency modules)
    - Each element: Vector of shape (batch_size, n_cells_f)
    
Theory:
    Multi-scale representations allow the model to capture both local
    precision (high frequencies) and global structure (low frequencies)
    in a factorized manner, similar to multi-scale grid cells in the
    entorhinal cortex.
"""

AbstractLocation = MultiScaleCode
"""Abstract spatial representation (g) from transition dynamics.

The abstract location encodes position in a factorized, multi-scale
representation. It is updated through path integration and refined
through sensory inference.

Theory:
    Corresponds to grid cell populations in medial entorhinal cortex,
    organized into modules with different spatial frequencies.
"""


GroundedLocation = MultiScaleCode
"""Grounded place cell representation (p) from memory retrieval.

The grounded location is retrieved from Hebbian memory using the abstract
location as a query. It represents discrete place cell activations that
are tied to specific environmental features.

Theory:
    Corresponds to hippocampal place cells that encode discrete locations
    with sensory associations. The mapping from abstract to grounded
    locations is learned through Hebbian plasticity.
"""


Location: TypeAlias = GroundedLocation | AbstractLocation
"""Generic location code, either grounded (p) or abstract (g)."""


@dataclass(frozen=True)
class Transition:
    """Probabilistic abstract location estimate with uncertainty quantification.

    Represents a Gaussian estimate of abstract location across multiple
    frequency modules. Each frequency has independent mean and uncertainty.

    Attributes:
        mean: Predicted abstract location per frequency [List of (B, n_g[f])]
        uncertainty: Prediction uncertainty (sigma) per frequency [List of (B, n_g[f])]

    Theory:
        The transition model produces Gaussian estimates for path integration.
        Uncertainty is used to weight fusion with memory and sensory cues
        via precision (1/σ²) weighting. See torch_tem.utils.fusion for
        fusion algorithms.
    """

    mean: Location
    uncertainty: Optional[MultiScaleCode]


Observation = Tensor
"""Ground-truth sensory observation (o) from the environment.

The sensory observation represents the actual sensory input received
at a given timestep, distinct from model-generated sensory predictions.
Observations are provided as single tensors (typically one-hot or encoded)
and are processed internally into multi-scale representations.

Shape:
    - Batched: (batch_size, n_observations)
    - One-hot encoding: Each row has a single 1.0 at the observation index

Theory:
    Raw observations drive the inference pathway, providing the sensory
    evidence that the model uses to infer location and update beliefs.
    Distinguished from SensoryPrediction which is a multi-scale structure
    with both values and logits generated by the model.
"""

Action = Tensor
"""Ground-truth action signal (a) associated with a timestep.

The action represents the agent's motor command (or discrete action index)
that drives state transitions in the environment and therefore informs TEM's
transition/path-integration dynamics.

Shape:
    - Time-major batched (common in DataModule): (T, B)
    - Batched per-step: (batch_size,)
    - Unbatched per-step: () or (1,)

Recommended dtype:
    - Discrete actions: integer type (e.g., torch.long)
    - Continuous actions (if used): float type (e.g., torch.float32)

Theory:
    In TEM, actions are used by the transition model to predict the next
    abstract location (g) via path integration. They are conceptually distinct
    from observations (o), which provide sensory evidence for inference.
"""

Location = Tensor
"""Ground-truth environment location label (ℓ) for supervision and evaluation.

This represents the environment-provided notion of "true" location (e.g., a
grid cell index in a discrete maze). It is primarily used for auxiliary
supervision, diagnostics, and plotting, and is not required for TEM's core
generative/inference loops.

Shape:
    - Time-major batched (common in DataModule): (T, B)
    - Batched per-step: (batch_size,)
    - Unbatched per-step: () or (1,)

Recommended dtype:
    - Discrete location indices: integer type (e.g., torch.long)

Notes:
    The dataloader may zero-out location labels when they are not requested
    (see `WalkBatch`), so consumers should treat this signal as optional.
"""

# =============================================================================
# Memory Structures
# =============================================================================

HebbianMemory = List[Matrix]
"""Attractor network connection weights for memory storage.

Structure:
    - Single memory: [M_gen]
    - Dual memory: [M_gen, M_inf]
    
Where:
    - M_gen: Generative memory (g → p pathway for generation)
    - M_inf: Inference memory (o → p pathway for inference, optional)

Shape:
    Each matrix: [batch_size, sum(n_p), sum(n_p)]

Theory:
    Hebbian plasticity M = λ·M + η·(p_inf + p_gen_gi) ⊗ (p_inf - p_gen_gi)
    creates associative connections between abstract and grounded
    representations, enabling both memory-guided generation and inference.
    The dual-memory architecture allows separate optimization of the
    inference and generative pathways.
"""

MemoryState = HebbianMemory
"""Complete memory state at a given timestep.

Alias for HebbianMemory to clarify temporal context when passing
memory state between iterations or for serialization/checkpointing.
"""


# =============================================================================
# Data Flow
# =============================================================================


@dataclass(frozen=True)
class SensoryPrediction:
    """Composed sensory observation prediction with values and logits.

    Attributes:
        values: Predicted sensory observation (probabilities or activations)
        logits: Pre-softmax logits corresponding to the prediction

    Theory:
        The generative pathway produces sensory predictions from latent
        representations. Both the final values and their logits are retained
        for loss calculation (e.g., cross-entropy from logits).
    """

    values: MultiScaleCode
    logits: MultiScaleCode


@dataclass(frozen=True)
class LocationInference:
    """Composed latent location prediction with abstract and grounded codes.

    Attributes:
        abstract: Abstract location code (g) representing position in a
                 factorized multi-scale representation
        grounded: Grounded location code (p) representing discrete place cell
                 activations tied to environmental features

    Theory:
        The inference pathway produces both abstract (grid-like) and grounded
        (place-like) representations. These are used together for memory
        interaction and sensory generation.
    """

    abstract: AbstractLocation
    grounded: GroundedLocation


@dataclass
class StepInput:
    """Input data for a single TEM iteration.

    Attributes:
        observation: Sensory observation (o) for the current timestep
        action: Action taken at the previous timestep (or None for initial step)
        location_info: Environment metadata (e.g., shiny object locations)

    Theory:
        The model receives sensory observations and action information,
        which drive both the transition model (path integration) and
        sensory inference.
    """

    observation: MultiScaleCode
    action: Optional[int]
    location_info: Dict[str, Any]


@dataclass
class Trajectory:
    """A sequence of inputs forming a walk or episode.

    Attributes:
        steps: Sequence of StepInput objects
        metadata: Optional trajectory-level metadata (e.g., environment ID)

    Theory:
        A trajectory represents a continuous sequence of observations and
        actions through an environment. The model processes trajectories
        sequentially to build spatial representations and memories.
    """

    steps: Sequence[StepInput]
    metadata: Optional[Dict[str, Any]] = None

    def __len__(self) -> int:
        """Return the number of steps in the trajectory."""
        return len(self.steps)

    def __getitem__(self, index: int) -> StepInput:
        """Get a specific step from the trajectory."""
        return self.steps[index]

    def __iter__(self):
        """Iterate over steps in the trajectory."""
        return iter(self.steps)


# =============================================================================
# Batch Processing
# =============================================================================

WalkSample: TypeAlias = Tuple[Vector, Vector, Vector]
"""Single unbatched walk sample.

This is the item-level return type used by the map-style walk dataset.

Tuple elements:
    observations: Float tensor of shape (T, n_o).
    actions: Integer tensor of shape (T,).
    locations: Integer tensor of shape (T,).

Notes:
    - The TEM DataModule collates a list of `WalkSample` into a time-major `WalkBatch`.
    - The time dimension is always first (time-major), which simplifies truncated BPTT.
"""

WalkBatch: TypeAlias = Tuple[Observation, Action, Location]
"""Single time-major batch of walks.

Tuple elements:
    observations: Float tensor of shape (T, B, n_o).
    actions: Integer tensor of shape (T, B).
    locations: Integer tensor of shape (T, B).

Notes:
    `locations` is auxiliary and may be zeroed out by the dataloader's collate
    function when location labels are not required.
"""


BatchedCode = Vector
"""A batched multi-scale code with shape (batch_size, n_cells).

Note: Individual MultiScaleCode elements are already batched.
This type emphasizes that batching occurs at the Vector level.
"""

BatchedMemory = Matrix
"""A batched memory matrix with shape (batch_size, n_cells_pre, n_cells_post).

Note: In practice, memory is typically shared across a batch rather than
per-sample, so this represents the global memory state.
"""


# ....
Reduction: TypeAlias = Literal["none", "sum", "mean"]
Scalar: TypeAlias = int | float | Tensor
