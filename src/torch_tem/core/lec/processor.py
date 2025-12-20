"""LEC Sensory Processor: Multi-frequency temporal filtering.

The Processor implements the second stage of the Lateral Entorhinal Cortex (LEC)
sensory processing pathway. It applies frequency-specific exponential smoothing
to compressed sensory inputs, creating multiple temporally-filtered views at
different timescales.

Multi-frequency filtering provides:
- Temporal credit assignment: Different frequencies capture different timescales
- Stability: Smoothing reduces noise and improves generalization
- Hierarchical representation: Lower frequencies = more temporal context
- Learnable dynamics: Each frequency has a learnable decay rate (alpha)

Architecture:
    Input: Compressed sensory [B, n_x_c]
    Processing: Per-frequency exponential moving average
    Output: Multi-frequency filtered sensory List[n_f] of [B, n_x_c]

    For each frequency f:
        x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

    where alpha[f] is a learnable decay rate in (0, 1).

    Note: Normalization (f_n) is applied later in the Projection module.
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import MultiScaleCode


class ProcessorParams(Protocol):
    """Protocol defining required parameters for Processor initialization.

    Attributes:
        n_f: Number of frequency channels (temporal filtering scales)
        f_extended: Initial frequency values for each channel [0, 1]
                   Higher values = higher spatial/temporal frequency = less smoothing
    """

    n_f: int
    f_extended: List[float]


class Processor(nn.Module):
    """LEC Sensory Processor with multi-frequency temporal filtering.

    Applies frequency-specific exponential moving averages to compressed sensory
    inputs, creating n_f parallel views at different temporal smoothing scales.
    Each frequency channel has a learnable decay rate (alpha) that controls the
    balance between new input and previous state.

    The filtering operation for each frequency f is:
        x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

    Normalization is applied later in the Projection module via f_n().

    Args:
        params: Configuration object implementing ProcessorParams protocol.
                Must provide n_f and f_extended.

    Attributes:
        n_f: Number of frequency channels
        f_tensor: Base frequency values (used to initialize alpha parameters)
        alpha_logit: Learnable decay rates in logit space for each frequency

    Example:
        >>> from torch_tem.config import ModelConfig
        >>> config = ModelConfig(n_x_c=8, f_initial=[0.9, 0.5, 0.2])
        >>> processor = Processor(config)
        >>> x_c = torch.randn(4, 8)  # Compressed sensory [B, n_x_c]
        >>> x_prev = [torch.zeros(4, 8) for _ in range(3)]  # Previous state
        >>> x_f = processor(x_c, x_prev)  # List[3] of [4, 8]
        >>> len(x_f)
        3
        >>> x_f[0].shape
        torch.Size([4, 8])
    """

    def __init__(self, params: ProcessorParams):
        super().__init__()
        self.n_f = params.n_f

        # Initialize learnable decay rates from base frequencies
        # Use logit space to ensure alpha stays in (0, 1) after sigmoid
        self.f_tensor = torch.tensor(params.f_extended, dtype=torch.float)
        logits = torch.logit(self.f_tensor)
        self.alpha_logit = nn.ParameterList([nn.Parameter(logits[i : i + 1]) for i in range(self.n_f)])

    def filter_temporal(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        """Apply exponential smoothing at each frequency channel.

        For each frequency f, computes:
            x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

        where alpha[f] = sigmoid(alpha_logit[f]) ∈ (0, 1).

        Args:
            x_c: Compressed sensory input [B, n_x_c]
            x_prev: Previous filtered state List[n_f] of [B, n_x_c]

        Returns:
            Filtered sensory (before normalization) List[n_f] of [B, n_x_c]
        """
        alpha = [torch.sigmoid(self.alpha_logit[f]) for f in range(self.n_f)]
        return [alpha[f] * x_c + (1 - alpha[f]) * x_prev[f] for f in range(self.n_f)]

    def forward(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        """Process compressed sensory through temporal filtering.

        Applies exponential smoothing at each frequency:
            x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

        Args:
            x_c: Compressed sensory input [B, n_x_c]
            x_prev: Previous filtered state List[n_f] of [B, n_x_c]

        Returns:
            Multi-frequency filtered sensory List[n_f] of [B, n_x_c]
            (Normalization applied later in Projection module)
        """
        return self.filter_temporal(x_c, x_prev)
