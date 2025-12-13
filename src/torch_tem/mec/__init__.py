from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..types import AbstractLocation, GroundedLocation, MultiScaleCode
from . import abstract, projection, transition


class MECParams(abstract.AbstractLocParams, transition.TransitionParams):
    """ """

    n_g: List[int]
    n_f: int
    batch_size: int


@dataclass
class MECState:
    """ """

    transition_stats: transition
    abstract_location: AbstractLocation
    projection: MultiScaleCode


class MECModel(nn.Module):
    """ """

    def __init__(self, params: MECParams):
        """ """
        super().__init__()
        self.transition = transition.TransitionModel(params)  # Transition model module
        self.abstract = abstract.AbstractLocInference(params)  # Abstract location inference module
        self.projection = projection.Projection(params)  # Projection head for MEC pathway
        self.batch_size = params.batch_size

    @property
    def n_g(self) -> List[int]:
        """ """
        return self.transition.n_g

    @property
    def n_f(self) -> int:
        """ """
        return self.transition.n_f

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
        g_gen = self.transition(a, state.abstract_location)  # State transition: g → g (predict next abstract location)
        g = self.abstract(g_gen, p_x, locations)  # Infer entorhinal (abstract location)
        g_ = self.projection(g)  # Project to hippocampal input: g → g_
        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)

    def init_state(self, device: torch.device) -> MECState:
        """ """
        g_gen = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        return MECState(transition_stats=g_gen, abstract_location=g)


__all__ = ["MECParams", "MECState", "MECModel"]
