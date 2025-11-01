"""Observation decoder for torch_tem package."""

from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..config.facets import DecoderParams
from .mlp import MLP


class ObservationDecoder(nn.Module):
    """Decodes grounded location into observation predictions.

    Uses MLP to map from grounded location to observation space,
    outputting both softmax probabilities and raw logits.
    """

    def __init__(self, params: DecoderParams):
        """Initialize observation decoder.

        Args:
            params: Configuration satisfying DecoderParams protocol
        """
        super().__init__()
        self.n_x = params.n_x
        self.n_x_c = params.n_x_c
        self.n_x_f = params.n_x_f_calculated

        # MLP for decoding: from compressed sensory to full observation
        # Only use first (highest frequency) module's grounded location
        self.mlp_decoder = MLP(
            in_dim=self.n_x_f[0],  # Input from highest frequency
            out_dim=self.n_x,  # Output full observation
            activation=(torch.nn.functional.elu, None),
            hidden_dim=None,  # Will use mean of in/out
            bias=(True, True),
        )

    def forward(self, p: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Decode grounded location to observation.

        Args:
            p: List of grounded locations per frequency.
               Uses only p[0] (highest frequency).

        Returns:
            x_probs: [B, n_x] softmax probabilities over observations
            x_logits: [B, n_x] raw logits (for loss computation)
        """
        # Use only highest frequency grounded location
        # First, we need to reduce p to sensory dimensions
        # Take first n_x_f[0] dimensions as they correspond to sensory
        p_sensory = p[0][:, : self.n_x_f[0]]

        # Decode through MLP
        x_logits = self.mlp_decoder(p_sensory)

        # Apply softmax for probabilities
        x_probs = torch.nn.functional.softmax(x_logits, dim=-1)

        return x_probs, x_logits
