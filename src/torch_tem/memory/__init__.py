"""Memory system for torch_tem package."""

from typing import List

from torch import Tensor

from .attractor import AttractorDynamics
from .storage import MemoryStorage

Matrix = Tensor
"""A 2D tensor representing connection weights or transformations."""

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

__all__ = [
    "MemoryStorage",
    "AttractorDynamics",
    "MemoryState",
    "HebbianMemory",
]
