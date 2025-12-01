"""PyTorch implementation of the Tolman-Eichenbaum Machine (TEM)."""

from torch_tem.types import (  # Primitives; Multi-scale representations; Memory; Losses; Data flow; Transition; Batch processing; Utilities
    AbstractLocation,
    BatchedCode,
    BatchedMemory,
    GroundedLocation,
    HebbianMemory,
    Losses,
    Matrix,
    MemoryState,
    MultiScaleCode,
    StepInput,
    TEMState,
    Trajectory,
    TransitionOutput,
    TransitionParams,
    Vector,
)

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
    "TEMState",
    "Trajectory",
    # Transition
    "TransitionParams",
    "TransitionOutput",
    # Batch processing
    "BatchedCode",
    "BatchedMemory",
]
