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
``torch_tem.config.facets.ProcessorParams``. Only the following fields
are required by this implementation:

- ``n_f``: int, number of frequencies (length of the filter bank)
- ``n_x_c``: int, feature dimension of the compressed sensory vector
- ``n_x_f``: int, optional downstream convenience (unused here)
- ``f_extended``: Sequence[float], frequency values in (0, 1]

Notes
-----
- Frequencies close to 1.0 emphasize the current input (short memory), while
    smaller frequencies emphasize historical inputs (longer memory).
- Normalization is performed per-frequency to keep the behavior consistent with
    the original implementation and to prevent scale drift across channels.
"""

from typing import List, Protocol

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import Matrix, MultiScaleCode, SensoryObservation


class ProcessorParams(Protocol):
    """Minimal interface for SensoryProcessor.

    Dependencies: n_f, n_x_c, n_x_f, f_extended
    Complexity: Low (4 parameters)
    """

    n_x_c: int
    n_f: int
    n_x_f: List[int]
    f_extended: List[float]


class SensoryProcessor(nn.Module):
    """Temporal filtering and normalization of sensory input.

    Applies exponential smoothing per frequency module:
    x_f[freq] = freq * x_c + (1 - freq) * x_prev[freq]

    Then applies L2 normalization with learnable weights and biases.
    """

    def __init__(self, params: ProcessorParams):
        super().__init__()

        self.n_f = params.n_f  # Number of frequencies (filter channels)
        self.n_x_c = params.n_x_c  # Feature dimension of compressed sensory input
        self.n_x_f = params.n_x_f  # Calculated size for downstream components
        self.f_initial = params.f_extended  # Frequency values in (0, 1]

        # Learnable normalization parameters
        # Element-wise affine transform prior to L2 normalization
        self.w_x = nn.Parameter(torch.ones(1, self.n_x_c))  # [1, n_x_c]
        self.b_x = nn.Parameter(torch.zeros(1, self.n_x_c))  # [1, n_x_c]

        # Store LOGIT (inverse sigmoid) as learnable parameter
        # This allows unconstrained optimization while sigmoid maps to (0, 1)
        f_tensor = torch.tensor(self.f_initial, dtype=torch.float)
        logits = torch.logit(f_tensor)
        self.alpha_logit = nn.ParameterList([nn.Parameter(logits[i : i + 1]) for i in range(self.n_f)])

    def filter_temporal(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        """Apply exponential smoothing per frequency.

        x_f[freq] = freq * x_c + (1 - freq) * x_prev[freq]

        Args:
            x_c: Current compressed sensory [B, n_x_c]
            x_prev: Previous filtered sensory [n_f] of [B, n_x_c]

        Returns:
            x_f: Filtered sensory [n_f] of [B, n_x_c]
        """
        alpha = [torch.sigmoid(self.alpha_logit[f]) for f in range(self.n_f)]
        return [alpha[f] * x_c + (1 - alpha[f]) * x_prev[f] for f in range(self.n_f)]

    def normalize(self, x_f: MultiScaleCode) -> MultiScaleCode:
        """L2 normalize with learnable weights.

        Legacy normalization: L2(ReLU(x - mean(x)))
        Note: w_x and b_x are NOT used here, but are stored in this module
        for use by the ObservationDecoder (shared parameters).

        Args:
            x_f: Filtered sensory [n_f] of [B, n_x_c]

        Returns:
            x_normalized: Normalized sensory [n_f] of [B, n_x_c]
        """
        x_normalized = []
        for f in range(self.n_f):
            # Legacy normalization:
            # 1. Center by subtracting scalar mean
            x_centered = x_f[f] - torch.mean(x_f[f])
            # 2. Apply ReLU
            x_relu = torch.relu(x_centered)
            # 3. L2 normalize
            x_norm = torch.nn.functional.normalize(x_relu, p=2, dim=-1)
            x_normalized.append(x_norm)
        return x_normalized

    def forward(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
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


class EncoderParams(Protocol):
    """Minimal interface for SensoryEncoder.

    Dependencies: n_x, n_x_c, two_hot_table
    Complexity: Low (3 parameters)
    """

    n_x: int
    n_x_c: int


class SensoryEncoder(nn.Module):
    """Encodes one-hot observations to two-hot compressed representation via lookup table.

    Parameter-free compression using pre-computed two-hot codes. Each code has exactly
    two active bits, enabling efficient outer products and Hebbian learning downstream.

    Attributes:
        n_x: Number of unique observations
        n_x_c: Compressed dimension (two-hot code length)
        two_hot_table: List[Tensor] of [n_x_c] codes with 2 active elements each

    Args:
        params: EncoderParams with n_x, n_x_c, two_hot_table
        two_hot_table: List[Tensor] of [n_x_c] codes with 2 active elements each

    Example:
        >>> params = SimpleNamespace(n_x=10, n_x_c=5,
        ...     two_hot_table=create_two_hot_table(10, 5))
        >>> encoder = SensoryEncoder(params)
        >>> x = torch.zeros(2, 10); x[0, 0] = 1.0; x[1, 5] = 1.0
        >>> x_c = encoder(x)  # Shape: [2, 5], each row has 2 active bits
    """

    def __init__(self, params: EncoderParams, two_hot_table: List[Matrix]):
        """Initialize encoder with two-hot lookup table."""
        super().__init__()
        self.n_x = params.n_x
        self.n_x_c = params.n_x_c
        self.two_hot_table = two_hot_table

    def forward(self, x: SensoryObservation) -> Tensor:
        """Encode one-hot observation [B, n_x] to two-hot [B, n_x_c].

        Args:
            x: One-hot tensor, each row has single 1.0 at observation index

        Returns:
            x_c: Two-hot codes, each row has exactly two 1.0 values
        """
        indices = torch.argmax(x, dim=1)  # Extract active observation index [B]
        two_hot_tensor = torch.stack(self.two_hot_table).to(x.device)  # [n_x, n_x_c]
        x_c = two_hot_tensor[indices]  # Batch lookup [B, n_x_c]
        return x_c


class ProjectionParams(Protocol):
    """Minimal interface for SensoryProjection.

    Dependencies: n_f, n_x_f
    Complexity: Low (2 parameters)
    """

    n_f: int
    n_x_f: List[int]


class SensoryProjection(nn.Module):
    """Projects sensory input to p-space via learnable tiling transformation.

    In TEM, sensory observations (x) must be transformed to match the dimensionality
    of the grounded location space (p) for Hebbian memory operations. This module
    applies a frequency-specific tiling matrix (W_tile) with learnable gating weights
    to prepare sensory input for outer product computation with abstract locations.

    Architecture:
        - Per-frequency tiling matrices (W_tile): Fixed transformation matrices
        - Per-frequency gate weights (w_p): Learnable scalars controlling contribution
        - Sigmoid activation: Ensures 0-1 gating range

    Forward Pass:
        x_[f] = sigmoid(w_p[f]) * (x_normalized[f] @ W_tile[f])

    Args:
        params: Configuration providing n_f, n_x_f, and W_tile matrices
        W_tile: Fixed tiling matrices [n_x_f[f] x n_p[f] for f in n_f]

    Attributes:
        n_f: Number of frequency modules
        n_x_f: Sensory dimensions per frequency [n_x_f[f] for f in n_f]
        W_tile: Fixed tiling matrices [n_x_f[f] x n_p[f] for f in n_f]
        w_p: Learnable gate weights [n_f learnable scalars]

    Shape:
        Input: List of [B, n_x_f[f]] tensors (one per frequency)
        Output: List of [B, n_p[f]] tensors (one per frequency)
    """

    def __init__(self, params: ProjectionParams, W_tile: List[Matrix]):
        """Initialize sensory projection with tiling matrices and gate weights."""
        super().__init__()
        self.n_f = params.n_f
        self.n_x_f = params.n_x_f
        self.W_tile = W_tile

        # Initialize learnable gate weights (one per frequency module)
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def forward(self, x_normalized: MultiScaleCode) -> MultiScaleCode:
        """Transform normalized sensory input to p-space representation.

        Args:
            x_normalized: Temporally filtered sensory input per frequency
                         List of [B, n_x_f[f]] tensors

        Returns:
            List of [B, n_p[f]] tensors ready for memory indexing
        """
        x_ = []
        for f in range(self.n_f):
            # Gate sensory input with learnable weight (sigmoid ensures [0,1])
            gate = torch.sigmoid(self.w_p[f])
            # Apply tiling transformation to match p-space dimensions
            W_tile_f = self.W_tile[f].to(x_normalized[f].device)
            x_f = gate * torch.matmul(x_normalized[f], W_tile_f)
            x_.append(x_f)

        return x_


if __name__ == "__main__":
    # Minimal usage example (smoke test)
    # This example creates a small frequency bank and runs a forward pass.
    import types

    B, n_x_c, n_f = 2, 4, 3  # batch, feature dimension, number of frequencies

    # Build a lightweight params stub with the required fields
    params = types.SimpleNamespace(
        n_f=n_f,
        n_x_c=n_x_c,
        n_x_f=n_f * n_x_c,  # not used by this module, but often handy downstream
        f_extended=[0.1, 0.5, 0.9],
    )

    proc = SensoryProcessor(params)

    # Current compressed sensory input and previous filtered states
    x_c = torch.randn(B, n_x_c)
    x_prev = [torch.zeros(B, n_x_c) for _ in range(n_f)]

    x_out = proc(x_c, x_prev)

    print(f"Frequencies: {n_f}")
    print(f"Input shape: {tuple(x_c.shape)}")
    print(f"Output list length: {len(x_out)}; each shape: {tuple(x_out[0].shape)}")
