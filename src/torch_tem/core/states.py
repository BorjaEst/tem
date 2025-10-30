#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State containers for TEM model components.
Uses dataclasses to provide clear interfaces between modules.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from torch import Tensor


@dataclass
class SensoryState:
    """Container for sensory processing outputs."""

    raw: Tensor  # Raw one-hot observation [batch, n_x]
    compressed: Tensor  # Compressed observation [batch, n_x_c]
    filtered: List[Tensor]  # Temporally filtered per frequency
    normalized: List[Tensor]  # Normalized per frequency
    memory_ready: List[Tensor]  # Prepared for memory input


@dataclass
class AbstractLocationState:
    """State container for abstract location (grid cells)."""

    mu: List[Tensor]  # Mean per frequency module
    sigma: List[Tensor]  # Uncertainty per frequency
    sources: List[str] = field(default_factory=list)  # Which sources contributed
    metadata: dict = field(default_factory=dict)


@dataclass
class GroundedLocationState:
    """State container for grounded location (place cells)."""

    p: List[Tensor]  # Place cell activations per frequency
    sigma: Optional[List[Tensor]] = None  # Uncertainty (if computed)


@dataclass
class ObservationDistribution:
    """Container for generated observation distribution."""

    logits: Tensor  # Raw logits [batch, n_x]
    probabilities: Tensor  # Softmax probabilities [batch, n_x]

    def sample(self) -> Tensor:
        """Sample from categorical distribution."""
        return torch.multinomial(self.probabilities, 1)

    def mode(self) -> Tensor:
        """Return most likely observation."""
        return torch.argmax(self.probabilities, dim=-1)


@dataclass
class EpisodeState:
    """Complete state for an episode."""

    memory_gen: Tensor  # Generative memory matrix
    memory_inf: Optional[Tensor]  # Inference memory matrix (if separate)
    g_prev: List[Tensor]  # Previous abstract location
    x_prev: List[Tensor]  # Previous sensory state
    step: int  # Current step number
    metadata: dict = field(default_factory=dict)


@dataclass
class IterationStates:
    """All internal states from one iteration."""

    sensory: SensoryState
    abstract_inf: AbstractLocationState
    abstract_gen: AbstractLocationState
    grounded_inf: GroundedLocationState
    grounded_gen: List[Tensor]


@dataclass
class LossComponents:
    """Individual loss components."""

    L_p_g: Tensor  # Grounded location: inferred vs generated
    L_p_x: Tensor  # Grounded location: inferred vs from sensory
    L_x_gen: Tensor  # Observation: generated from transition
    L_x_g: Tensor  # Observation: generated from inferred g
    L_x_p: Tensor  # Observation: generated from inferred p
    L_g: Tensor  # Abstract location: inferred vs generated
    L_reg_g: Tensor  # Regularization: L2 norm of g
    L_reg_p: Tensor  # Regularization: L1 norm of p

    def as_list(self) -> List[Tensor]:
        """Return losses as list in standard order."""
        return [self.L_p_g, self.L_p_x, self.L_x_gen, self.L_x_g, self.L_x_p, self.L_g, self.L_reg_g, self.L_reg_p]

    @property
    def total(self) -> Tensor:
        """Sum of all loss components."""
        return sum(self.as_list())


@dataclass
class IterationResult:
    """Complete result from one iteration."""

    losses: LossComponents
    states: IterationStates
    observations: List[ObservationDistribution]
    next_state: EpisodeState
