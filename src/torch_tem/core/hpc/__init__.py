from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import HPCSettings
from torch_tem.types import Matrix


@dataclass
class HPCState:
    p: List[Tensor]  # Multi-frequency filtered features
    memory: List[Matrix]  # Memory matrices


class HPCModel(nn.Module):

    def __init__(self, shape: List[int], settings: HPCSettings):
        super().__init__()
        self._settings = settings

        # Store hyperparameters
        self._shape = shape

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._shape

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return len(self.shape)

    def forward(self, *, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        raise NotImplementedError("HPCModel forward pass not yet implemented.")
