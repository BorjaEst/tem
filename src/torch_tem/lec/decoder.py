"""Observation decoder for torch_tem package.

This module implements the final generative step from grounded locations (place cells)
to sensory observations. It completes the generative pathway: g → p → x, allowing
the model to predict what sensory input should be experienced given an internal
spatial representation.

The Decoder performs p→x decoding using learned sensory processing
parameters (w_x, b_x) and tiling matrices (W_tile).
"""

from typing import List, Protocol, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.types import GroundedLocation, Observation, SensoryPrediction


class DecoderParams(Protocol):
    """Protocol defining required parameters for Decoder initialization.

    Attributes:
        n_x: Number of sensory observation neurons (output dimension)
        n_x_c: Number of compressed sensory neurons (int or List[int])
    """

    n_x: int
    n_x_c: Union[int, List[int]]


class Decoder(nn.Module):
    """Decodes grounded location (place cells) to sensory predictions.

    W_tile matrices are managed by the parent LECModel and passed to forward()
    to ensure consistency and proper device management.

    Args:
        params: Configuration with n_x, n_x_c
    """

    def __init__(self, params: DecoderParams):
        super().__init__()
        # Extract n_x_c - use first element if it's a list
        n_x_c = params.n_x_c[0] if isinstance(params.n_x_c, list) else params.n_x_c
        self.n_x_c = n_x_c
        self.w_x = nn.Parameter(torch.ones(1, n_x_c))
        self.b_x = nn.Parameter(torch.zeros(1, n_x_c))

        activation: Tuple = (torch.nn.functional.elu, None)
        hidden_dim = 20 * n_x_c
        self.mlp_decoder = MLP(n_x_c, params.n_x, activation, hidden_dim, bias=(True, True))

    def forward(self, p: List[Tensor], W_tile_0: Tensor) -> SensoryPrediction:
        """Decode place cells to sensory prediction.

        Args:
            p: Grounded location (place cells) as List[n_f] of (batch, n_p[f])
            W_tile_0: Tiling matrix for frequency 0 with shape (n_x_c, n_p[0])

        Returns:
            SensoryPrediction with observation probabilities and logits
        """
        x_proj = torch.matmul(p[0], W_tile_0.t())
        x = self.w_x * x_proj + self.b_x
        x_logits = self.mlp_decoder(x)
        x_probs = torch.nn.functional.softmax(x_logits, dim=-1)
        return SensoryPrediction(values=[x_probs], logits=[x_logits])
