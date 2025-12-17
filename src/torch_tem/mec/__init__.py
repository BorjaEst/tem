from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..types import AbstractLocation, GroundedLocation, MultiScaleCode
from . import abstract, projection, transition
from .transition import Transition


class MECParams(abstract.AbstractLocParams, transition.TransitionParams):
    """Protocol for MEC model initialization parameters.

    Combines parameters needed for abstract location inference and transitions.

    Attributes:
        n_g: Abstract location dimensions per frequency module.
        n_f: Number of frequency modules.
        batch_size: Batch size for state initialization.
    """

    n_g: List[int]
    n_f: int
    batch_size: int


@dataclass
class MECState:
    """State container for MEC pathway.

    Attributes:
        transition_stats: Predicted abstract location from transition model.
        abstract_location: Fused abstract location after inference.
        projection: Projected abstract location to hippocampal input space.
    """

    transition_stats: Transition
    abstract_location: AbstractLocation
    projection: MultiScaleCode

    def detach(self) -> "MECState":
        """Detach all tensors from the computation graph."""
        return MECState(
            transition_stats=Transition(
                mean=[m.detach() for m in self.transition_stats.mean],
                uncertainty=[u.detach() for u in self.transition_stats.uncertainty],
            ),
            abstract_location=[x.detach() for x in self.abstract_location],
            projection=[x.detach() for x in self.projection],
        )


class MECModel(nn.Module):
    """Medial Entorhinal Cortex (MEC) pathway for spatial navigation.

    Implements abstract location processing through:
    - Transition prediction (path integration)
    - Abstract location inference (fusion with memory and landmarks)
    - Projection to hippocampal input space
    """

    def __init__(self, params: MECParams):
        """Initialize MEC model.

        Args:
            params: Configuration with n_g, n_f, and all submodule parameters.
        """
        super().__init__()
        self.transition = transition.TransitionModel(params)  # Transition model module
        self.abstract = abstract.AbstractLocInference(params)  # Abstract location inference module
        self.projection = projection.Projection(params)  # Projection head for MEC pathway
        self.batch_size = params.batch_size

    @property
    def n_g(self) -> List[int]:
        """Abstract location dimensions per frequency module."""
        return self.transition.n_g

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return self.transition.n_f

    def init_state(self, device: torch.device) -> MECState:
        """Initialize MEC state with zeros.

        Args:
            device: Device for tensor allocation.

        Returns:
            Initial MECState with zero-initialized abstract locations.
        """
        g_gen = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g_ = self.projection(g)  # Project to hippocampal input
        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)

    def forward(self, p_x: Optional[GroundedLocation], locations: List[Dict], a: Optional[Tensor], state: MECState) -> MECState:
        """Forward pass through MEC pathway.

        Args:
            p_x: Hippocampal pattern from sensory retrieval (inference mode) or None (generative mode)
            locations: Environment descriptors for landmark cues
            a: Action taken
            state: Previous MEC state

        Returns:
            Updated MEC state with new abstract location
        """
        g_gen = self.transition(state.abstract_location, a)  # State transition: g → g (predict next abstract location)
        g = self.abstract(g_gen, p_x, locations)  # Infer entorhinal (abstract location)
        g_ = self.projection(g)  # Project to hippocampal input: g → g_
        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)


__all__ = ["MECParams", "MECState", "MECModel"]
