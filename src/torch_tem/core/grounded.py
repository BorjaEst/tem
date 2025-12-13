"""Grounded location inference for the Tolman-Eichenbaum Machine (TEM).

Computes hippocampal-like place cell representations by combining abstract location
(grid cells) with sensory information via outer product: p = g ⊗ x

The outer product creates conjunctive codes that bind spatial location with sensory
context, analogous to how hippocampal place cells encode location-specific patterns.

Theory:
    The outer product g ⊗ x is implemented via Kronecker product matrices:
    - g_expanded = g @ W_repeat (done by Projection.forward())
    - x_expanded = x @ W_tile (done by SensoryProjection)
    - p = (g_expanded ⊙ x_expanded) weighted and activated

    This module performs ONLY the final element-wise multiplication and activation,
    as the expansion is already done by upstream modules.

Reference: Whittington et al. (2020). Cell, 183(5), 1249-1263.
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import GroundedLocation, MultiScaleCode


class GroundedLocParams(Protocol):
    """Minimal interface for GroundedLocInference.

    Dependencies: n_f, n_p
    Complexity: Low (2 parameters)

    Note: W_repeat and W_tile are no longer needed here as expansion
          is handled by Projection and SensoryProjection respectively.
    """

    n_f: int
    n_p: List[int]


class GroundedLocInference(nn.Module):
    """Infers grounded location p via element-wise product of expanded inputs.

    Creates hippocampal place-like representations by binding ALREADY-EXPANDED
    grid cell patterns (g_) with sensory context (x_). The inputs are expected
    to have already been projected to place cell dimensions.

    Architecture Flow:
        Upstream:
            1. Projection.forward(g) → g_ [B, n_p[f]] (downsample + W_repeat expansion)
            2. SensoryProjection(x_f) → x_ [B, n_p[f]] (W_tile expansion + w_p gating)
        This module:
            3. p = activation(g_ ⊙ x_) [B, n_p[f]] (element-wise product only)

    Attributes:
        n_f: Number of frequency modules
        n_p: List[int] of grounded location dimensions per frequency

    Args:
        params: GroundedLocParams with n_f, n_p

    Example:
        >>> params = SimpleNamespace(n_f=2, n_p=[96, 80])
        >>> grounded = GroundedLocInference(params)
        >>> g_expanded = [torch.randn(4, 96), torch.randn(4, 80)]  # Already expanded
        >>> x_expanded = [torch.randn(4, 96), torch.randn(4, 80)]  # Already expanded
        >>> p = grounded(g_expanded, x_expanded)  # Element-wise product + activation
    """

    def __init__(self, params: GroundedLocParams):
        """Initialize with learnable weights for sensory-spatial balance."""
        super().__init__()
        self.n_f = params.n_f
        self.n_p = params.n_p

    def forward(self, g_expanded: MultiScaleCode, x_expanded: MultiScaleCode) -> GroundedLocation:
        """Compute grounded location via element-wise product of expanded inputs.

        Args:
            g_expanded: ALREADY expanded abstract location [n_f] of [B, n_p[f]]
                       (via Projection.forward() which does downsample + W_repeat expansion)
            x_expanded: ALREADY expanded AND gated sensory [n_f] of [B, n_p[f]]
                       (via SensoryProjection which applies W_tile expansion + w_p gating)

        Returns:
            p: Grounded location [n_f] of [B, n_p[f]]

        Mathematical operation per frequency f:
            p[f] = leaky_relu(clamp(g_expanded[f] ⊙ x_expanded[f], -1, 1))

        Theory:
            The outer product structure g ⊗ x = (g @ W_repeat) ⊙ (x @ W_tile) is
            computed upstream. This module performs only the final binding via
            element-wise multiplication, matching the legacy inf_p implementation:
                mu_p = f_p(g_[f] * x_[f])  # Element-wise product with activation
        """
        p = []
        for f in range(self.n_f):
            # Element-wise product (Hadamard)
            # x_expanded already has w_p gating from SensoryProjection
            p_f = g_expanded[f] * x_expanded[f]

            # Apply activation: leaky_relu(clamp(x, -1, 1))
            # Matches legacy f_p activation
            p_f = torch.nn.functional.leaky_relu(torch.clamp(p_f, min=-1.0, max=1.0))

            p.append(p_f)

        return p


if __name__ == "__main__":
    """Grounded location inference via element-wise product."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    import types

    print("=" * 80)
    print("GroundedLocInference Example - TEM Place Cell Formation")
    print("=" * 80)

    # Configuration: 2 frequency modules with different scales
    n_f = 2
    n_p = [96, 80]  # Place cell dimensions (already expanded)
    batch_size = 3

    print(f"\nConfig:")
    print(f"  Frequency modules: {n_f}")
    print(f"  Place cells (p): {n_p} per frequency")
    print(f"  Note: Inputs are ALREADY EXPANDED to place cell dimensions")

    # Create inference module
    params = types.SimpleNamespace(n_f=n_f, n_p=n_p)

    grounded = GroundedLocInference(params)
    print(f"\nModule initialized with {n_f} frequency modules")
    print(f"Note: w_p gating is handled by SensoryProjection upstream")

    # Simulate ALREADY EXPANDED inputs (from Projection and SensoryProjection)
    g_expanded = [torch.randn(batch_size, n_p[f]) for f in range(n_f)]
    x_expanded = [torch.randn(batch_size, n_p[f]) for f in range(n_f)]

    print(f"\nInputs (already expanded):")
    print(f"  g_expanded shapes: {[tuple(g.shape) for g in g_expanded]}")
    print(f"  x_expanded shapes: {[tuple(x.shape) for x in x_expanded]}")

    # Compute grounded location (place cells)
    with torch.no_grad():
        p = grounded(g_expanded, x_expanded)

    print(f"\nOutputs (grounded location p):")
    print(f"  p shapes: {[tuple(p_f.shape) for p_f in p]}")

    # Verify dimensions match
    print(f"\n✓ Verification:")
    for f in range(n_f):
        expected_dim = n_p[f]
        actual_dim = p[f].shape[1]
        print(f"  Freq {f}: Expected dim={expected_dim}, Actual dim={actual_dim}, Match={expected_dim == actual_dim}")

    # Show activation statistics
    print(f"\nActivation Statistics:")
    for f in range(n_f):
        mean_val = p[f].mean().item()
        std_val = p[f].std().item()
        min_val = p[f].min().item()
        max_val = p[f].max().item()
        print(f"  Freq {f}: mean={mean_val:.4f}, std={std_val:.4f}, range=[{min_val:.4f}, {max_val:.4f}]")

    # TEM integration (updated architecture)
    print("\n" + "=" * 80)
    print("TEM Integration (Updated Architecture):")
    print("  1. TransitionModel: a → g (predict abstract location from action)")
    print("  2. lec.Encoder: x → x_c (compress to two-hot)")
    print("  3. lec.Processor: x_c → x_f (temporal filtering)")
    print("  4. SensoryProjection: x_f → x_ (W_tile expansion to n_p)")
    print("  5. Projection: g → g_ (downsample + W_repeat expansion to n_p)")
    print("  6. GroundedLocInference: (g_, x_) → p (element-wise product)")
    print("  7. MemoryStorage: Store p via Hebbian M = λM + η·outer(p,p)")
    print("  8. AttractorDynamics: Retrieve p_gen = M^T @ p (memory recall)")
    print("\nKey Change: Expansion now happens in steps 4-5, not in step 6!")
    print("=" * 80)
