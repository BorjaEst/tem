"""Sensory projection module for TEM.

Transforms normalized sensory observations to p-space representation for Hebbian
memory indexing. Part of the inference pipeline when use_p_inf=True.
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor


class SensoryProjectionParams(Protocol):
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

    def __init__(self, params: SensoryProjectionParams, W_tile: List[Tensor]):
        """Initialize sensory projection with tiling matrices and gate weights."""
        super().__init__()
        self.n_f = params.n_f
        self.n_x_f = params.n_x_f
        self.W_tile = W_tile

        # Initialize learnable gate weights (one per frequency module)
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def forward(self, x_normalized: List[Tensor]) -> List[Tensor]:
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
    """Minimal example demonstrating SensoryProjection usage."""
    from pydantic import BaseModel, ConfigDict

    # Define minimal config satisfying SensoryProjectionParams protocol
    class ExampleConfig(BaseModel):
        """Minimal configuration for example."""

        model_config = ConfigDict(arbitrary_types_allowed=True)

        n_f: int = 3  # Number of frequency modules
        n_x_f: list[int] = [10, 8, 6]  # Sensory dimensions per frequency

        @property
        def n_f(self) -> int:
            return self.n_f

        @property
        def n_x_f(self) -> list[int]:
            return self.n_x_f

    # Initialize module
    config = ExampleConfig()
    W_tile = [
        torch.randn(10, 15),  # [n_x_f[0], n_p[0]]
        torch.randn(8, 12),  # [n_x_f[1], n_p[1]]
        torch.randn(6, 9),  # [n_x_f[2], n_p[2]]
    ]
    projection = SensoryProjection(config, W_tile)

    # Create example input (batch_size=2, temporally filtered sensory observations)
    x_normalized = [
        torch.randn(2, 10),  # [B, n_x_f[0]]
        torch.randn(2, 8),  # [B, n_x_f[1]]
        torch.randn(2, 6),  # [B, n_x_f[2]]
    ]

    # Forward pass
    x_projected = projection(x_normalized)

    # Display results
    print("SensoryProjection Example")
    print("=" * 50)
    print(f"Number of frequency modules: {config.n_f}")
    print(f"Input dimensions per frequency: {config.n_x_f}")
    print(f"Output dimensions per frequency: {[W.shape[1] for W in config.W_tile]}")
    print()
    print("Gate weights (before sigmoid):")
    for f, w in enumerate(projection.w_p):
        print(f"  Frequency {f}: {w.item():.4f} → sigmoid: {torch.sigmoid(w).item():.4f}")
    print()
    print("Input/Output shapes:")
    for f in range(config.n_f):
        print(f"  Frequency {f}: {x_normalized[f].shape} → {x_projected[f].shape}")
