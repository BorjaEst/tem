from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor, nn

from ..types import AbstractLocation, GroundedLocation
from . import abstract, transition


class MECParams(abstract.AbstractLocParams, transition.TransitionParams):
    """ """

    n_g: List[int]
    n_f: int
    batch_size: int


@dataclass
class MECState:
    """ """

    transition_stats: AbstractLocation
    abstract_location: AbstractLocation


class MECModel(nn.Module):
    """ """

    def __init__(self, params: MECParams):
        """ """
        super().__init__()
        self.transition = transition.TransitionModel(params)  # Transition model module
        self.abstract = abstract.AbstractLocInference(params)  # Abstract location inference module
        self.batch_size = params.batch_size

    @property
    def n_g(self) -> List[int]:
        """ """
        return self.transition.n_g

    @property
    def n_f(self) -> int:
        """ """
        return self.transition.n_f

    def forward(self, p: GroundedLocation, locations: List[Dict], a: Optional[Tensor], state: MECState) -> MECState:
        """ """
        g_gen = self.transition(a, state.abstract_location)  # State transition: g → g (predict next abstract location)
        g = self.abstract(g_gen, p, locations)  # Infer entorhinal (abstract location)
        return MECState(transition_stats=g_gen, abstract_location=g)

    def init_state(self, device: torch.device) -> MECState:
        """ """
        g_gen = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        return MECState(transition_stats=g_gen, abstract_location=g)


__all__ = ["MECParams", "MECState", "MECModel"]
