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

from typing import List

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.types import MultiScaleCode


class ProcessorConfig(BaseModel):
    """Processor configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # Learnable parameters initialization
    learn_alpha: bool = Field(default=True, description="If True, alpha decay rates require gradients; if False, frozen")


class Processor(nn.Module):
    """LEC Sensory Processor with multi-frequency temporal filtering.

    Applies frequency-specific exponential moving averages to compressed sensory
    inputs, creating n_f parallel views at different temporal smoothing scales.
    Each frequency channel has a learnable decay rate (alpha) that controls the
    balance between new input and previous state.

    The filtering operation for each frequency f is:
        x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

    Normalization is applied later in the Projection module.

    Args:
        f_initial: Initial frequency values for alpha initialization
        config: Processor configuration parameters
    """

    def __init__(self, f_initial: List[float], config: ProcessorConfig):
        """Initialize processor with learnable temporal filtering.

        Args:
            f_initial: Initial frequency values for each channel
            config: Processor configuration (learning control)
        """
        super().__init__()
        self._config = config

        # Initialize learnable decay rates from base frequencies
        # Use logit space to ensure alpha stays in (0, 1) after sigmoid
        alpha_init = torch.tensor(f_initial, dtype=torch.float)
        alpha_logit = torch.logit(alpha_init)

        # Create learnable parameters for each frequency
        # Always create as parameters, control learning via requires_grad
        p = [nn.Parameter(alpha_logit[i : i + 1], requires_grad=config.learn_alpha) for i in range(self.n_f)]
        self._alpha_logit = nn.ParameterList(p)

    @property
    def n_f(self) -> int:
        """Number of frequency channels."""
        return len(self._alpha_logit)

    def set_alpha_learning(self, learn: bool) -> None:
        """Set learning state and synchronize immediately.

        Convenience method that combines config mutation and sync.

        Args:
            learn: If True, enable gradients; if False, freeze parameters
        """
        self._config.learn_alpha = learn
        for param in self._alpha_logit:
            param.requires_grad_(learn)

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
        alpha = [torch.sigmoid(alpha_f) for alpha_f in self._alpha_logit]
        return [alpha_f * x_c + (1 - alpha_f) * x_prev[f] for f, alpha_f in enumerate(alpha)]

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


__all__ = ["Processor", "ProcessorConfig"]
