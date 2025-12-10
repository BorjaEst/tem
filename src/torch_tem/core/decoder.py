"""Observation decoder for torch_tem package.

This module implements the final generative step from grounded locations (place cells)
to sensory observations. It completes the generative pathway: g → p → x, allowing
the model to predict what sensory input should be experienced given an internal
spatial representation.

The ObservationDecoder performs p→x decoding using learned sensory processing
parameters (w_x, b_x) and tiling matrices (W_tile).
"""

from typing import List, Protocol, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.types import GroundedLocation, Observation, SensoryPrediction


class DecoderParams(Protocol):
    n_x: int
    n_x_c: int


class Decoder(nn.Module):

    def __init__(self, params: DecoderParams, W_tile: List[Tensor]):
        super().__init__()

        # Validate W_tile shape
        if len(W_tile) == 0:
            raise ValueError("W_tile must contain at least one matrix")
        if (actual_cols := W_tile[0].shape[1]) != (expected_cols := params.n_x_c):
            raise ValueError(f"W_tile[0] has incorrect shape: expected {W_tile[0].shape[0]}x{expected_cols}, got {W_tile[0].shape[0]}x{actual_cols}")

        self.w_x = nn.Parameter(torch.ones(1, params.n_x_c))
        self.b_x = nn.Parameter(torch.zeros(1, params.n_x_c))
        self.register_buffer("W_tile_0", W_tile[0])

        activation: Tuple = (torch.nn.functional.elu, None)
        hidden_dim = 20 * params.n_x_c
        self.mlp_decoder = MLP(params.n_x_c, params.n_x, activation, hidden_dim, bias=(True, True))

    def forward(self, p: List[Tensor]) -> SensoryPrediction:
        x_proj = torch.matmul(p[0], self.W_tile_0.t())
        x = self.w_x * x_proj + self.b_x
        x_logits = self.mlp_decoder(x)
        x_probs = torch.nn.functional.softmax(x_logits, dim=-1)
        return SensoryPrediction(values=[x_probs], logits=[x_logits])
