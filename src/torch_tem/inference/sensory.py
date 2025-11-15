"""Sensory processing utilities for the Tolman-Eichenbaum Machine (TEM).

This module implements a lightweight sensory pre-processor used by TEM
components before inference and learning. It provides two core steps:

1) Temporal filtering (exponential smoothing) per frequency channel
     For each frequency f in a bank of smoothing frequencies, we compute:

             x_f[f] = f * x_c + (1 - f) * x_prev[f]

     where `x_c` is the current compressed sensory vector and `x_prev[f]` is the
     previously filtered vector for that frequency. This creates a set of
     lagged/low-pass views of the same input, which can improve temporal
     credit assignment and stability in downstream modules.

2) Normalization with learnable affine parameters
     Each filtered vector is transformed by element-wise weights and biases and
     then L2-normalized to unit length:

             x_norm = (w_x * x + b_x) / ||w_x * x + b_x||_2

Shapes
------
- Batch size: B
- Input compressed sensory: [B, n_x_c]
- Number of frequencies: n_f
- Output (list): length n_f, each tensor of shape [B, n_x_c]

Parameters
----------
`SensoryProcessor` is configured via an object matching
``torch_tem.config.facets.SensoryProcessorParams``. Only the following fields
are required by this implementation:

- ``n_f_calculated``: int, number of frequencies (length of the filter bank)
- ``n_x_c``: int, feature dimension of the compressed sensory vector
- ``n_x_f_calculated``: int, optional downstream convenience (unused here)
- ``f_initial_extended``: Sequence[float], frequency values in (0, 1]

Notes
-----
- Frequencies close to 1.0 emphasize the current input (short memory), while
    smaller frequencies emphasize historical inputs (longer memory).
- Normalization is performed per-frequency to keep the behavior consistent with
    the original implementation and to prevent scale drift across channels.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.config.facets import SensoryProcessorParams


class SensoryProcessor(nn.Module):
    """Temporal filtering and normalization of sensory input.

    Applies exponential smoothing per frequency module:
    x_f[freq] = freq * x_c + (1 - freq) * x_prev[freq]

    Then applies L2 normalization with learnable weights and biases.
    """

    def __init__(self, params: SensoryProcessorParams):
        super().__init__()

        self.n_f = params.n_f_calculated  # Number of frequencies (filter channels)
        self.n_x_c = params.n_x_c  # Feature dimension of compressed sensory input
        self.n_x_f = params.n_x_f_calculated  # Calculated size for downstream components
        self.f_initial = params.f_initial_extended  # Frequency values in (0, 1]

        # Learnable normalization parameters
        # Element-wise affine transform prior to L2 normalization
        self.w_x = nn.Parameter(torch.ones(1, self.n_x_c))  # [1, n_x_c]
        self.b_x = nn.Parameter(torch.zeros(1, self.n_x_c))  # [1, n_x_c]

    def filter_temporal(self, x_c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        """Apply exponential smoothing per frequency.

        x_f[freq] = freq * x_c + (1 - freq) * x_prev[freq]

        Args:
            x_c: Current compressed sensory [B, n_x_c]
            x_prev: Previous filtered sensory [n_f] of [B, n_x_c]

        Returns:
            x_f: Filtered sensory [n_f] of [B, n_x_c]
        """
        x_f = []
        for f in range(self.n_f):
            # Scalar frequency weight for this channel
            freq = self.f_initial[f]
            # EMA-like update combining current input with previous filtered state
            x_f.append(freq * x_c + (1 - freq) * x_prev[f])
        return x_f

    def normalize(self, x_f: List[Tensor]) -> List[Tensor]:
        """L2 normalize with learnable weights.

        x_norm = (w_x * x + b_x) / ||w_x * x + b_x||_2

        Applied per frequency for consistency with original implementation.

        Args:
            x_f: Filtered sensory [n_f] of [B, n_x_c]

        Returns:
            x_normalized: Normalized sensory [n_f] of [B, n_x_c]
        """
        x_normalized = []
        for f in range(self.n_f):
            # Apply element-wise affine transform prior to normalization
            x_weighted = self.w_x * x_f[f] + self.b_x
            # Stabilized L2 normalization (add small epsilon to avoid divide-by-zero)
            x_norm = x_weighted / (torch.norm(x_weighted, dim=1, keepdim=True) + 1e-8)
            x_normalized.append(x_norm)
        return x_normalized

    def forward(self, x_c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        """Process sensory: filter and normalize.

        Args:
            x_c: Current compressed sensory
            x_prev: Previous filtered sensory

        Returns:
            x_processed: Filtered and normalized sensory
        """
        # 1) Exponential smoothing for each frequency channel
        x_f = self.filter_temporal(x_c, x_prev)
        # 2) Per-channel L2 normalization with learnable affine transform
        x_normalized = self.normalize(x_f)
        return x_normalized


if __name__ == "__main__":
    # Minimal usage example (smoke test)
    # This example creates a small frequency bank and runs a forward pass.
    import types

    B, n_x_c, n_f = 2, 4, 3  # batch, feature dimension, number of frequencies

    # Build a lightweight params stub with the required fields
    params = types.SimpleNamespace(
        n_f_calculated=n_f,
        n_x_c=n_x_c,
        n_x_f_calculated=n_f * n_x_c,  # not used by this module, but often handy downstream
        f_initial_extended=[0.1, 0.5, 0.9],
    )

    proc = SensoryProcessor(params)

    # Current compressed sensory input and previous filtered states
    x_c = torch.randn(B, n_x_c)
    x_prev = [torch.zeros(B, n_x_c) for _ in range(n_f)]

    x_out = proc(x_c, x_prev)

    print(f"Frequencies: {n_f}")
    print(f"Input shape: {tuple(x_c.shape)}")
    print(f"Output list length: {len(x_out)}; each shape: {tuple(x_out[0].shape)}")
