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
            f_initial: Initial frequency values for each channel.
            config: Processor configuration (learning control).
        """
        super().__init__()
        self._config = config

        # Initialize learnable decay rates from base frequencies
        # Use logit space to ensure alpha stays in (0, 1) after sigmoid
        alpha_init = torch.tensor(f_initial, dtype=torch.float)
        alpha_logit = torch.logit(alpha_init)

        # Create learnable parameters for each frequency
        # Always create as parameters, control learning via requires_grad
        p = [nn.Parameter(alpha_logit[i : i + 1], requires_grad=config.learn_alpha) for i in range(len(f_initial))]
        self._alpha_logit = nn.ParameterList(p)

    @property
    def n_f(self) -> int:
        """Number of frequency channels."""
        return len(self._alpha_logit)

    def set_alpha_learning(self, learn: bool) -> None:
        """Set learning state and synchronize immediately.

        Convenience method that combines config mutation and sync.

        Args:
            learn: If True, enable gradients; if False, freeze parameters.
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
            x_c: Compressed sensory input [B, n_x_c].
            x_prev: Previous filtered state List[n_f] of [B, n_x_c].

        Returns:
            Filtered sensory (before normalization) List[n_f] of [B, n_x_c].
        """
        alpha = [torch.sigmoid(alpha_f) for alpha_f in self._alpha_logit]
        return [alpha_f * x_c + (1 - alpha_f) * x_prev[f] for f, alpha_f in enumerate(alpha)]

    def forward(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        """Process compressed sensory through temporal filtering.

        Applies exponential smoothing at each frequency:
            x_f[f] = alpha[f] * x_c + (1 - alpha[f]) * x_prev[f]

        Args:
            x_c: Compressed sensory input [B, n_x_c].
            x_prev: Previous filtered state List[n_f] of [B, n_x_c].

        Returns:
            Multi-frequency filtered sensory List[n_f] of [B, n_x_c].
            Normalization applied later in Projection module.
        """
        return self.filter_temporal(x_c, x_prev)


__all__ = ["Processor", "ProcessorConfig"]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Processor usage example: Multi-frequency temporal filtering.

    Demonstrates how the processor applies exponential smoothing at different
    frequencies to create multiple temporally-filtered views of sensory input.
    """
    print("=" * 80)
    print("Processor Example - Multi-Frequency Temporal Filtering")
    print("=" * 80)

    # Configuration
    f_initial = [0.8, 0.5, 0.3]  # Decay rates per frequency
    n_x_c = 10  # Compressed sensory dimension
    batch_size = 4
    n_steps = 5

    print(f"\nConfiguration:")
    print(f"  Frequencies: {len(f_initial)}")
    print(f"  Initial alpha values: {f_initial}")
    print(f"  Compressed dimension: {n_x_c}")
    print(f"  Batch size: {batch_size}")
    print(f"  Time steps: {n_steps}")

    # Create processor
    config = ProcessorConfig(learn_alpha=True)
    processor = Processor(f_initial, config)
    print(f"\n✓ Processor initialized (n_f={processor.n_f})")

    # Initialize previous state
    x_prev = [torch.zeros(batch_size, n_x_c) for _ in range(processor.n_f)]
    print(f"✓ Initial state: {[x.shape for x in x_prev]}")

    # Simulate temporal sequence
    print(f"\nTemporal filtering over {n_steps} steps:")
    for t in range(n_steps):
        # New compressed sensory input (random walk)
        x_c = torch.randn(batch_size, n_x_c) * 0.1
        if t > 0:
            x_c = x_c + x_prev[0] * 0.5  # Correlation with previous

        # Filter at all frequencies
        with torch.no_grad():
            x_f = processor(x_c, x_prev)

        # Show statistics
        print(f"  Step {t}: ", end="")
        for f in range(processor.n_f):
            change = (x_f[f] - x_prev[f]).abs().mean().item()
            print(f"freq_{f}_change={change:.4f} ", end="")
        print()

        # Update state
        x_prev = x_f

    print(f"\n✓ Final filtered representations:")
    for f in range(processor.n_f):
        print(f"  Frequency {f}: mean={x_prev[f].mean():.4f}, std={x_prev[f].std():.4f}")

    print("\n" + "=" * 80)
