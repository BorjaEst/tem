from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from ..types import MultiScaleCode, Observation
from . import processor, sensory


class LECParams(sensory.EncoderParams, processor.ProcessorParams):
    """ """

    n_x_c: List[int]
    n_f: int
    batch_size: int


@dataclass(frozen=True)
class LECState:
    """ """

    compressed_observation: MultiScaleCode
    filtered_observation: MultiScaleCode


class LECModel(nn.Module):
    """ """

    def __init__(self, params: LECParams):
        """ """
        super().__init__()
        self.encoder = sensory.Encoder(params)  # Sensory encoder module: x → x_c
        self.processor = processor.Processor(params)  # Sensory processor module: x
        self.batch_size = params.batch_size

    @property
    def n_x_c(self) -> List[int]:
        """ """
        return self.encoder.config.n_x_c

    @property
    def n_f(self) -> int:
        """ """
        return self.encoder.config.n_f

    def forward(self, x: Observation, state: LECState) -> LECState:
        """ """
        x_c = self.encoder(x)  # Compress sensory observation: x → x_c (one-hot to two-hot)
        x_f = self.processor(x_c, state.filtered_observation)  # Temporally filter sensorium: x_c → x_f
        return LECState(compressed_observation=x_c, filtered_observation=x_f)

    def init_state(self, device: torch.device) -> LECState:
        """ """
        x_c = [torch.zeros((self.batch_size, self.n_x_c[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        x_f = [torch.zeros((self.batch_size, self.n_x_c[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        return LECState(compressed_observation=x_c, filtered_observation=x_f)


__all__ = ["LECParams", "LECState", "LECModel"]
