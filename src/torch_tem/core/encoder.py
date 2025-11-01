"""Sensory encoder for torch_tem package."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ..config.facets import EncoderParams


class SensoryEncoder(nn.Module):
    """Encodes sensory observations into compressed representation.

    Compresses one-hot sensory observations into two-hot (or learned) representation.
    Uses a lookup table for two-hot encoding.
    """

    def __init__(self, params: EncoderParams):
        """Initialize sensory encoder.

        Args:
            params: Configuration satisfying EncoderParams protocol
        """
        super().__init__()
        self.n_x = params.n_x
        self.n_x_c = params.n_x_c
        self.two_hot_table = params.two_hot_table_calculated

    def forward(self, x: Tensor) -> Tensor:
        """Encode sensory observation to compressed representation.

        Args:
            x: [B, n_x] one-hot sensory observation

        Returns:
            x_c: [B, n_x_c] compressed representation (two-hot)
        """
        # Get indices of active observation
        indices = torch.argmax(x, dim=1)

        # Stack two-hot table into a single tensor for efficient lookup
        # Shape: [n_codes, n_x_c]
        two_hot_tensor = torch.stack(self.two_hot_table).to(x.device)

        # Use indices to select appropriate two-hot codes
        # Shape: [B, n_x_c]
        x_c = two_hot_tensor[indices]

        return x_c
