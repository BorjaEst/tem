"""LEC Sensory Encoder: Two-hot compression of high-dimensional observations.

The Encoder implements the first stage of the Lateral Entorhinal Cortex (LEC)
sensory processing pathway. It compresses one-hot encoded observations into a
lower-dimensional two-hot representation using a pre-computed lookup table.

Two-hot encoding provides:
- Dimensionality reduction: n_x → n_x_c (typically 45 → 10)
- Distributed representation: Each observation encoded by exactly 2 active units
- Smooth transitions: Similar observations share one active unit
- Efficient learning: Sparse activations enable faster credit assignment

Architecture:
    Input: One-hot observations [B, n_x]
    Processing: Lookup table mapping observation index → two-hot code
    Output: Two-hot compressed sensory [B, n_x_c]
"""

from typing import List, Protocol, Union

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import Observation


class EncoderParams(Protocol):
    """Protocol defining required parameters for Encoder initialization.

    Attributes:
        two_hot_table: Lookup table mapping observation indices to two-hot codes
        n_x_c: Number of compressed sensory neurons (int or List[int])
    """

    n_x_c: Union[int, List[int]]

    @property
    def two_hot_table(self) -> List[Tensor]:
        """Two-hot encoding lookup table.

        Returns:
            List of n_x tensors, each of shape [n_x_c], representing the
            two-hot code for each possible observation index.
        """
        ...


class Encoder(nn.Module):
    """LEC Sensory Encoder using two-hot compression.

    Compresses high-dimensional one-hot observations into a lower-dimensional
    two-hot representation. The two-hot encoding uses exactly 2 active units
    per observation, providing a sparse distributed code that facilitates
    efficient learning and generalization.

    The lookup table is pre-computed using a systematic enumeration of all
    possible 2-hot codes (combinations of 2 active units from n_x_c dimensions).

    Args:
        params: Configuration object implementing EncoderParams protocol.
                Must provide n_x, n_x_c, and two_hot_table property.

    Attributes:
        two_hot_table: Registered buffer of shape [n_x, n_x_c] containing
                      the pre-computed two-hot codes for each observation.

    Example:
        >>> from torch_tem.config import ModelConfig
        >>> config = ModelConfig(n_x=25, n_x_c=8)
        >>> encoder = Encoder(config)
        >>> x = torch.zeros(4, 25)  # Batch of 4 one-hot observations
        >>> x[0, 5] = 1.0  # Observation index 5
        >>> x_c = encoder(x)  # [4, 8] two-hot compressed
        >>> (x_c[0].sum() == 2.0)  # Exactly 2 active units
        True
    """

    def __init__(self, params: EncoderParams):
        super().__init__()
        self.n_x_c = params.n_x_c if isinstance(params.n_x_c, list) else [params.n_x_c]
        self.register_buffer("two_hot_table", torch.stack(params.two_hot_table))

    def forward(self, x: Observation) -> Tensor:
        """Encode one-hot observations to two-hot compressed representation.

        Args:
            x: One-hot encoded observations of shape [B, n_x], where exactly
               one element per batch item is 1.0 and all others are 0.0.

        Returns:
            Two-hot compressed sensory of shape [B, n_x_c], where exactly
            two elements per batch item are 1.0 and all others are 0.0.
        """
        indices = torch.argmax(x, dim=1)  # Extract active observation index [B]
        return self.two_hot_table[indices]  # Batch lookup [B, n_x_c]
