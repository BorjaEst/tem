""" """

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import Matrix, MultiScaleCode, Observation

from .. import utils


class EncoderParams(Protocol):
    n_x: int
    n_x_c: int

    @property
    def two_hot_table(self) -> List[Tensor]:
        """Property for two-hot encoding table."""
        ...


class Encoder(nn.Module):
    def __init__(self, params: EncoderParams):
        super().__init__()
        self.register_buffer("two_hot_table", torch.stack(params.two_hot_list))

    def forward(self, x: Observation) -> Tensor:
        indices = torch.argmax(x, dim=1)  # Extract active observation index [B]
        return self.two_hot_table[indices]  # Batch lookup [B, n_x_c]
