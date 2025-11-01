"""Grounded location inference for the Tolman-Eichenbaum Machine (TEM).

Computes hippocampal-like place cell representations by combining abstract location
(grid cells) with sensory information via outer product: p = g ⊗ x

The outer product creates conjunctive codes that bind spatial location with sensory
context, analogous to how hippocampal place cells encode location-specific patterns.

Operation per frequency module f:
    p[f] = g[f] ⊗ x[f] = [g[0]·x, g[1]·x, ..., g[n_g-1]·x]

Implementation uses Kronecker product matrices (W_repeat, W_tile) for efficient
batched computation: p = (g @ W_repeat) ⊙ (x @ W_tile)

Reference: Whittington et al. (2020). Cell, 183(5), 1249-1263.
"""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.config.facets import GroundedInferenceParams


class GroundedLocationInference(nn.Module):
    """Infers grounded location p via outer product of abstract location g and sensory x.

    Creates hippocampal place-like representations by binding grid cell patterns (g)
    with sensory context (x). The outer product p[f] = g[f] ⊗ x[f] produces conjunctive
    codes where each element represents a specific (location, observation) combination.

    Attributes:
        n_f: Number of frequency modules
        n_p: List[int] of grounded location dimensions per frequency [n_g[f] * n_x_c]
        w_p: Learnable weights for sensory contribution per frequency
        W_repeat_{f}: Matrices for expanding g to outer product dimension
        W_tile_{f}: Matrices for expanding x to outer product dimension

    Args:
        params: GroundedInferenceParams with n_f_calculated, n_p_calculated,
                W_repeat_calculated, W_tile_calculated

    Example:
        >>> params = SimpleNamespace(n_f_calculated=2, n_p_calculated=[30, 24],
        ...     W_repeat_calculated=[torch.randn(10, 30), torch.randn(8, 24)],
        ...     W_tile_calculated=[torch.randn(3, 30), torch.randn(3, 24)])
        >>> grounded = GroundedLocationInference(params)
        >>> g = [torch.randn(4, 10), torch.randn(4, 8)]  # batch=4
        >>> x = [torch.randn(4, 3), torch.randn(4, 3)]
        >>> p = grounded(g, x)  # Returns list of [4, 30] and [4, 24]
    """

    def __init__(self, params: GroundedInferenceParams):
        """Initialize with Kronecker product matrices for efficient outer product computation."""
        super().__init__()
        self.n_f = params.n_f_calculated
        self.n_p = params.n_p_calculated

        # Register W_repeat and W_tile as buffers (not trainable)
        W_repeat = params.W_repeat_calculated
        W_tile = params.W_tile_calculated

        for f in range(self.n_f):
            self.register_buffer(f"W_repeat_{f}", W_repeat[f])
            self.register_buffer(f"W_tile_{f}", W_tile[f])

        # Learnable weights control sensory vs spatial dominance per frequency
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def forward(self, g_downsampled: List[Tensor], x_filtered: List[Tensor]) -> List[Tensor]:
        """Compute grounded location via outer product p = g ⊗ x per frequency.

        Args:
            g_downsampled: Downsampled abstract location [n_f] of [B, n_g_subsampled[f]]
            x_filtered: Temporally filtered sensory [n_f] of [B, n_x_c]

        Returns:
            p: Grounded location [n_f] of [B, n_p[f]], where n_p[f] = n_g[f] * n_x_c

        Mathematical operation per frequency f:
            g_expanded = g @ W_repeat  -> [B, n_p[f]]  (repeat g for each x dimension)
            x_expanded = x @ W_tile    -> [B, n_p[f]]  (tile x for each g dimension)
            p = w_p[f] * (g_expanded ⊙ x_expanded)    (element-wise product)
        """
        p = []
        for f in range(self.n_f):
            W_repeat = getattr(self, f"W_repeat_{f}")  # [n_g_sub[f], n_p[f]]
            W_tile = getattr(self, f"W_tile_{f}")  # [n_x_c, n_p[f]]

            # Expand g and x to outer product space via matrix multiplication
            g_repeated = torch.matmul(g_downsampled[f], W_repeat)  # [B, n_p[f]]
            x_tiled = torch.matmul(x_filtered[f], W_tile)  # [B, n_p[f]]

            # Outer product via element-wise multiplication
            p_f = g_repeated * x_tiled  # [B, n_p[f]]

            # Apply learnable weighting
            p_f = self.w_p[f] * p_f

            p.append(p_f)

        return p


if __name__ == "__main__":
    """Grounded location inference via outer product."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import types

    from torch_tem.utils import create_W_repeat, create_W_tile

    print("=" * 80)
    print("GroundedLocationInference Example - TEM Place Cell Formation")
    print("=" * 80)

    # Configuration: 2 frequency modules with different scales
    n_f = 2
    n_g_sub = [10, 6]  # Grid cell dimensions per frequency (downsampled)
    n_x_c = 4  # Sensory dimension (compressed)
    n_p = [40, 24]  # Place cell dimensions: n_g[f] * n_x_c
    batch_size = 3

    print(f"\nConfig:")
    print(f"  Frequency modules: {n_f}")
    print(f"  Grid cells (g): {n_g_sub} per frequency")
    print(f"  Sensory dim (x): {n_x_c}")
    print(f"  Place cells (p): {n_p} = [g[f] * x_c] per frequency")

    # Create Kronecker product matrices for outer product
    W_repeat = create_W_repeat(n_g_sub, [n_x_c] * n_f)
    W_tile = create_W_tile(n_g_sub, [n_x_c] * n_f)

    print(f"\nKronecker matrices:")
    for f in range(n_f):
        print(f"  Freq {f}: W_repeat {tuple(W_repeat[f].shape)}, W_tile {tuple(W_tile[f].shape)}")

    # Create inference module
    params = types.SimpleNamespace(n_f_calculated=n_f, n_p_calculated=n_p, W_repeat_calculated=W_repeat, W_tile_calculated=W_tile)

    grounded = GroundedLocationInference(params)
    print(f"\nModule initialized with {n_f} frequency modules")
    print(f"Learnable weights w_p: {[f'{w.item():.2f}' for w in grounded.w_p]}")

    # Simulate abstract location (grid cells) and sensory input
    g_downsampled = [torch.randn(batch_size, n_g_sub[f]) for f in range(n_f)]
    x_filtered = [torch.randn(batch_size, n_x_c) for f in range(n_f)]

    print(f"\nInputs:")
    print(f"  g_downsampled shapes: {[tuple(g.shape) for g in g_downsampled]}")
    print(f"  x_filtered shapes: {[tuple(x.shape) for x in x_filtered]}")

    # Compute grounded location (place cells)
    with torch.no_grad():
        p = grounded(g_downsampled, x_filtered)

    print(f"\nOutputs (grounded location p):")
    print(f"  p shapes: {[tuple(p_f.shape) for p_f in p]}")

    # Verify outer product structure
    print(f"\n✓ Verification:")
    for f in range(n_f):
        expected_dim = n_g_sub[f] * n_x_c
        actual_dim = p[f].shape[1]
        print(f"  Freq {f}: Expected dim={expected_dim}, Actual dim={actual_dim}, Match={expected_dim == actual_dim}")

    # Demonstrate conjunctive coding property
    print(f"\nConjunctive Coding Demo (Freq 0):")
    print(f"  Each place cell p[i,j] binds grid cell g[i] with sensory feature x[j]")

    # Manually compute outer product for first batch item, first frequency
    g_manual = g_downsampled[0][0:1]  # [1, n_g_sub[0]]
    x_manual = x_filtered[0][0:1]  # [1, n_x_c]

    # Compute via module
    p_module = p[0][0]  # [n_p[0]]

    # Verify structure: p should have blocks corresponding to g[i] * x for each i
    print(f"  Sample p[0,0:4] (g[0] * x): {p_module[0:4].detach().numpy()}")
    print(f"  Sample p[0,4:8] (g[1] * x): {p_module[4:8].detach().numpy()}")

    # Show activation statistics
    print(f"\nActivation Statistics:")
    for f in range(n_f):
        mean_val = p[f].mean().item()
        std_val = p[f].std().item()
        print(f"  Freq {f}: mean={mean_val:.4f}, std={std_val:.4f}")

    # TEM integration
    print("\n" + "=" * 80)
    print("TEM Integration:")
    print("  1. TransitionModel: a → g (predict abstract location from action)")
    print("  2. ProjectionHead: g → g_downsampled (downsample for memory indexing)")
    print("  3. SensoryProcessor: x_c → x_filtered (temporal filtering)")
    print("  4. GroundedLocationInference: g ⊗ x → p (bind location & sensory)")
    print("  5. MemoryStorage: Store p via Hebbian M = λM + η·outer(p,p)")
    print("  6. AttractorDynamics: Retrieve p_gen = M^T @ p (memory recall)")
    print("=" * 80)
