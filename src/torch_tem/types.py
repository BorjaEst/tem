"""Type definitions for the Tolman-Eichenbaum Machine.

This module defines the core types used throughout the TEM implementation,
grounded in the theoretical framework of the Tolman-Eichenbaum Machine.
These types abstract implementation details and provide clear semantics for
multi-scale spatial representations, Hebbian memory, and belief propagation.

Type Hierarchy
--------------
Mathematical Primitives:
    - Vector: 1D or 2D tensor representing a single code or batch of codes
    - Matrix: 2D tensor representing connection weights or transformations

Multi-Scale Representations:
    - MultiScaleCode: Hierarchical representation across frequency modules
    - AbstractLocation: Abstract spatial code (g) in factorized representation
    - GroundedLocation: Grounded place cell code (p) from memory retrieval

Memory Structures:
    - HebbianMemory: Attractor network connection weights
    - MemoryState: Complete memory state (supports single/dual memory)

Data Flow:
    - StepInput: Inputs for a single timestep iteration
    - StepOutput: Outputs from a single timestep iteration
    - Trajectory: Sequence of inputs forming a walk or episode
    - Losses: All loss components from a single timestep

Theory References
-----------------
The type system follows the TEM architecture:
    1. Sensory observations (x) → Sensory inference
    2. Abstract locations (g) → Multi-scale grid codes
    3. Grounded locations (p) → Place cell activations
    4. Hebbian memory (M) → Attractor dynamics
    5. Generative/Inference loops → Belief propagation
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
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

# =============================================================================
# Memory Structures
# =============================================================================

HebbianMemory = List[Matrix]
"""Attractor network connection weights for memory storage.

Structure:
    - Single memory: [M_gen]
    - Dual memory: [M_gen, M_inf]
    
Where:
    - M_gen: Generative memory (p → x pathway)
    - M_inf: Inference memory (x → p pathway, optional)

Theory:
    Hebbian learning (η·p·g^T - κ·M) creates associative connections
    between abstract and grounded representations, enabling both
    memory-guided generation and inference.
"""

MemoryState = HebbianMemory
"""Complete memory state at a given timestep.

Alias for HebbianMemory to clarify temporal context when passing
memory state between iterations.
"""

# =============================================================================
# Loss Components
# =============================================================================


@dataclass(frozen=True)
class Losses:
    """All loss components from a single TEM iteration.

    Attributes:
        L_p_g: Consistency loss between inferred and generated grounded locations
        L_p_x: Consistency loss between sensory-inferred and total-inferred grounded locations
        L_x_gen: Reconstruction loss for observation from generated abstract location
        L_x_g: Reconstruction loss for observation from inferred abstract location
        L_x_p: Reconstruction loss for observation from inferred grounded location
        L_g: Consistency loss between inferred and generated abstract locations
        L_reg_g: L2 regularization on abstract location codes
        L_reg_p: L1 regularization on grounded location codes

    Theory:
        The loss function balances multiple objectives:
        1. Consistency between generative and inference pathways
        2. Accurate sensory reconstruction
        3. Sparsity constraints on representations
    """

    L_p_g: Tensor  # ||p_inf - p_gen||²
    L_p_x: Tensor  # ||p_inf - p_x||²
    L_x_gen: Tensor  # CE(x, x_gen)
    L_x_g: Tensor  # CE(x, x_g)
    L_x_p: Tensor  # CE(x, x_p)
    L_g: Tensor  # ||g_inf - g_gen||²
    L_reg_g: Tensor  # ||g||²
    L_reg_p: Tensor  # ||p||₁

    def total(self, weights: Optional[List[float]] = None) -> Tensor:
        """Compute weighted sum of all loss components.

        Parameters:
            weights: Optional list of 8 weights for each loss component.
                    If None, uses equal weighting.

        Returns:
            Total scalar loss tensor.
        """
        if weights is None:
            weights = [1.0] * 8

        losses = self.as_list()
        return sum(w * L for w, L in zip(weights, losses))


# =============================================================================
# Data Flow
# =============================================================================


@dataclass
class StepInput:
    """Input data for a single TEM iteration.

    Attributes:
        observation: Sensory observation (x) for the current timestep
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
class StepOutput:
    """Output data from a single TEM iteration.

    Attributes:
        belief: Current belief state (abstract location g)
        prediction: Predicted sensory observation
        losses: All loss components for this timestep
        memory: Updated memory state
        generated: Optional generated pathway outputs
        inferred: Optional inference pathway outputs

    Theory:
        The model produces beliefs about current location, predictions
        about sensory input, and updated memory associations. Both
        generative and inference pathways contribute to the final state.
    """

    belief: AbstractLocation
    prediction: MultiScaleCode
    losses: Losses
    memory: MemoryState

    # Optional detailed outputs
    generated: Optional[Dict[str, Any]] = None
    inferred: Optional[Dict[str, Any]] = None


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
# Transition Dynamics
# =============================================================================


@dataclass
class TransitionParams:
    """Parameters for the transition model (path integration).

    Attributes:
        mean: Mean of the predicted abstract location
        uncertainty: Uncertainty (sigma) of the prediction

    Theory:
        The transition model predicts the next abstract location based on
        the previous location and action. The uncertainty quantifies
        prediction confidence and is used to weight inference and generation.
    """

    mean: AbstractLocation
    uncertainty: AbstractLocation


TransitionOutput = Tuple[AbstractLocation, TransitionParams]
"""Output from transition model: (g_gen, (g_gen_mu, sigma_gen))."""

# =============================================================================
# Batch Processing
# =============================================================================

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

# =============================================================================
# Type Guards and Utilities
# =============================================================================


def is_valid_multi_scale_code(code: Any, n_freq: int, batch_size: Optional[int] = None) -> bool:
    """Validate that a value is a well-formed MultiScaleCode.

    Parameters:
        code: Value to validate
        n_freq: Expected number of frequency modules
        batch_size: Optional expected batch size

    Returns:
        True if code is a valid MultiScaleCode with correct structure
    """
    if not isinstance(code, list):
        return False

    if len(code) != n_freq:
        return False

    for vector in code:
        if not isinstance(vector, torch.Tensor):
            return False
        if vector.ndim not in (1, 2):
            return False
        if batch_size is not None and vector.ndim == 2 and vector.shape[0] != batch_size:
            return False

    return True


def is_valid_hebbian_memory(memory: Any, n_freq: int) -> bool:
    """Validate that a value is a well-formed HebbianMemory.

    Parameters:
        memory: Value to validate
        n_freq: Expected number of frequency modules

    Returns:
        True if memory is a valid HebbianMemory (1 or 2 matrices)
    """
    if not isinstance(memory, list):
        return False

    if len(memory) not in (1, 2):
        return False

    for matrix in memory:
        if not isinstance(matrix, torch.Tensor):
            return False
        if matrix.ndim != 2:
            return False

    return True


__all__ = [
    # Primitives
    "Vector",
    "Matrix",
    # Multi-scale representations
    "MultiScaleCode",
    "AbstractLocation",
    "GroundedLocation",
    # Memory
    "HebbianMemory",
    "MemoryState",
    # Losses
    "Losses",
    # Data flow
    "StepInput",
    "StepOutput",
    "Trajectory",
    # Transition
    "TransitionParams",
    "TransitionOutput",
    # Batch processing
    "BatchedCode",
    "BatchedMemory",
    # Utilities
    "is_valid_multi_scale_code",
    "is_valid_hebbian_memory",
]
