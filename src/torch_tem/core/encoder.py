"""Sensory encoder for the Tolman-Eichenbaum Machine (TEM).

Compresses one-hot sensory observations (n_x dims) into two-hot codes (n_x_c dims)
via lookup table. Two-hot encoding maintains sparsity while reducing dimensionality:
C(n_x_c, 2) = n_x_c * (n_x_c - 1) / 2 unique codes possible.

Example (n_x_c=5): [0,0,0,1,1], [0,0,1,0,1], [0,0,1,1,0], ...

The compressed x_c feeds into SensoryProcessor (temporal filtering),
GroundedLocationInference (g ⊗ x_c → p), and ObservationDecoder (reconstruction).

Reference: Whittington et al. (2020). Cell, 183(5), 1249-1263.
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor


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

    def __init__(self, params: EncoderParams, two_hot_table: List[Tensor]):
        """Initialize encoder with two-hot lookup table."""
        super().__init__()
        self.n_x = params.n_x
        self.n_x_c = params.n_x_c
        self.two_hot_table = two_hot_table

    def forward(self, x: Tensor) -> Tensor:
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


if __name__ == "__main__":
    """Demo: Two-hot encoding properties and TEM integration.

    Run: python -m src.torch_tem.core.encoder
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import types

    from torch_tem.utils import create_two_hot_table

    print("=" * 80)
    print("SensoryEncoder Example - Tolman-Eichenbaum Machine (TEM)")
    print("=" * 80)

    # Configuration
    n_x, n_x_c, batch_size = 10, 5, 4

    print(f"\nConfig: n_x={n_x}, n_x_c={n_x_c}, capacity=C({n_x_c},2)={n_x_c*(n_x_c-1)//2}")

    # Create encoder
    two_hot_table = create_two_hot_table(n_x=n_x, n_x_c=n_x_c)
    params = types.SimpleNamespace(n_x=n_x, n_x_c=n_x_c, two_hot_table=two_hot_table)
    encoder = SensoryEncoder(params)

    # Show encoding table
    print(f"\nTwo-hot codes (first 5):")
    for i in range(min(5, n_x)):
        code = encoder.two_hot_table[i]
        active = torch.where(code == 1.0)[0].tolist()
        print(f"  Obs {i}: {code.tolist()} (bits {active})")

    # Encode batch
    x = torch.zeros(batch_size, n_x)
    obs_indices = [0, 3, 7, 9]
    for batch_idx, obs_idx in enumerate(obs_indices):
        x[batch_idx, obs_idx] = 1.0

    with torch.no_grad():
        x_c = encoder(x)

    print(f"\nInput: {tuple(x.shape)}, observations {obs_indices}")
    print(f"Output: {tuple(x_c.shape)}")

    # Verify properties
    num_active = x_c.sum(dim=1)
    print(f"\n✓ Two-hot property: {(num_active == 2.0).all().item()} (active={num_active.tolist()})")
    print(f"✓ Compression: {n_x/n_x_c:.1f}x")

    # Show encodings
    print(f"\nEncodings:")
    for i, obs_idx in enumerate(obs_indices):
        active = torch.where(x_c[i] == 1.0)[0].tolist()
        print(f"  Obs {obs_idx} → bits {active}")

    # Hamming distances
    print(f"\nHamming distances:")
    for i in range(len(obs_indices)):
        for j in range(i + 1, len(obs_indices)):
            dist = (x_c[i] != x_c[j]).sum().item()
            print(f"  Obs {obs_indices[i]} ↔ {obs_indices[j]}: {dist}")

    # TEM pipeline
    print("\n" + "=" * 80)
    print("TEM Pipeline:")
    print("  1. SensoryEncoder: x[B,n_x] → x_c[B,n_x_c] (two-hot compression)")
    print("  2. SensoryProcessor: x_c → x_f (temporal filtering per frequency)")
    print("  3. GroundedLocationInference: g ⊗ x_f → p (place cells)")
    print("  4. MemoryStorage: Hebbian M = λM + η·outer(p,p)")
    print("  5. ObservationDecoder: p → x̂ (reconstruction)")
    print("=" * 80)
