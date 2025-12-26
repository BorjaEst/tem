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

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem import utils
from torch_tem.types import Observation


class EncoderConfig(BaseModel):
    """Encoder configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    n_hot: int = Field(default=2, ge=2, frozen=True, description="Number of active units in two-hot encoding (fixed at 2)")


class Encoder(nn.Module):
    """LEC Sensory Encoder using n-hot compression.

    Compresses high-dimensional one-hot observations into a lower-dimensional
    n-hot representation. The n-hot encoding uses exactly n active units
    per observation (default n=2), providing a sparse distributed code that
    facilitates efficient learning and generalization.

    The lookup table is deterministically generated using a systematic
    enumeration of all possible n-hot codes (combinations of n active units
    from n_x_c dimensions).

    Args:
        n_x: Number of sensory observation neurons
        n_x_c: Compressed sensory dimension
        config: Encoder configuration parameters (n_hot strategy)
    """

    def __init__(self, n_x: int, n_x_c: int, config: EncoderConfig):
        """Initialize encoder with two-hot lookup table.

        Args:
            n_x: Number of sensory observation neurons
            n_x_c: Compressed sensory dimension
            config: Encoder configuration (n_hot parameter)
        """
        super().__init__()
        self._config = config

        # Pre-compute two-hot encoding lookup table
        encoding_table = utils.create_encoding_table(n_x, n_x_c, config.n_hot)
        self.register_buffer("encoding_table", torch.stack(encoding_table))

    @property
    def n_x(self) -> int:
        """Number of sensory observation neurons x."""
        return self.encoding_table.size(0)

    @property
    def n_x_c(self) -> int:
        """Number of compressed sensory neurons x_c."""
        return self.encoding_table.size(1)

    def forward(self, x: Observation) -> Tensor:
        """Encode one-hot observations to n-hot compressed representation.

        Args:
            x: One-hot encoded observations of shape [B, n_x], where exactly
               one element per batch item is 1.0 and all others are 0.0.

        Returns:
            N-hot compressed sensory of shape [B, n_x_c], where exactly
            n_hot elements per batch item are 1.0 and all others are 0.0.
        """
        indices = torch.argmax(x, dim=1)  # Extract active observation index [B]
        return self.encoding_table[indices]  # Batch lookup [B, n_x_c]


__all__ = ["Encoder", "EncoderConfig"]
