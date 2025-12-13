"""Hippocampus (HPC) module: Memory storage and retrieval for TEM.

This module implements the hippocampal memory system responsible for storing
and retrieving grounded location representations through Hebbian plasticity
and attractor dynamics.

Components:
    Memory: High-level interface combining storage and retrieval
    MemoryStorage: Hebbian memory matrices with plasticity updates
    AttractorDynamics: Iterative pattern completion for memory retrieval

Theory:
    The hippocampus maintains associative memory between abstract locations (g)
    and grounded locations (p) through Hebbian learning. Attractor dynamics
    enable pattern completion, allowing partial cues to retrieve full memories.

    The dual-memory architecture (M_gen/M_inf) supports bidirectional inference:
    - M_gen: Grid → Place mapping for generative pathway
    - M_inf: Sensory → Place mapping for inference pathway (optional)
"""

from .attractor import AttractorDynamics
from .memory import Memory
from .storage import MemoryStorage

__all__ = [
    "Memory",
    "MemoryStorage",
    "AttractorDynamics",
]
