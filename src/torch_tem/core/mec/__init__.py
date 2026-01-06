from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import MECSettings


@dataclass
class MECState:
    """State container for TEM model components."""

    g: List[Tensor]  # Abstract grid cell features


class MECModel(nn.Module):

    def __init__(self, n_a: int, n_g: List[int], settings: MECSettings, f_init: Optional[List[float]] = None):
        super(MECModel, self).__init__()

        # Store hyperparameters
        self._n_a = n_a  # Number of actions
        self._n_g = n_g  # Number of grid cell features per frequency module
        self.settings = settings

        # TODO: Initialize MEC-specific parameters here if needed

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return len(self._n_g)

    @property
    def n_in(self) -> int:
        """Dimensionality of compressed features."""
        return self._n_a

    @property
    def n_out(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._n_g

    def forward(self, a: Tensor, state: MECState) -> MECState:
        raise NotImplementedError("MEC forward pass not yet implemented.")
