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

import torch
import torch.nn as nn

from torch_tem.types import GroundedLocation, MultiScaleCode


class GroundedLocInference(nn.Module):
    """Infers grounded location p via element-wise product of expanded inputs.

    Creates hippocampal place-like representations by binding grid cell patterns
    with sensory context. Inputs must already be expanded to place cell dimensions
    by upstream Projection and SensoryProjection modules.

    Example:
        >>> grounded = GroundedLocInference()
        >>> g_expanded = [torch.randn(4, 96), torch.randn(4, 80)]
        >>> x_expanded = [torch.randn(4, 96), torch.randn(4, 80)]
        >>> p = grounded(g_expanded, x_expanded)
    """

    def forward(self, g_expanded: MultiScaleCode, x_expanded: MultiScaleCode) -> GroundedLocation:
        """Compute grounded location via element-wise product of expanded inputs.

        Args:
            g_expanded (MultiScaleCode): Expanded abstract location, list of [B, n_p[f]] tensors.
            x_expanded (MultiScaleCode): Expanded and gated sensory, list of [B, n_p[f]] tensors.

        Returns:
            GroundedLocation: Grounded location, list of [B, n_p[f]] tensors.

        Note:
            p[f] = leaky_relu(clamp(g_expanded[f] ⊙ x_expanded[f], -1, 1))
        """
        p = []
        for f in range(len(g_expanded)):
            # Element-wise product and activation
            p_f = g_expanded[f] * x_expanded[f]
            p_f = torch.nn.functional.leaky_relu(torch.clamp(p_f, min=-1.0, max=1.0))
            p.append(p_f)
        return p


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Grounded location inference example: conjunctive coding.

    Demonstrates how grounded locations are formed by binding grid cells
    with sensory input via element-wise product.
    """
    print("=" * 80)
    print("Grounded Location Inference Example - Conjunctive Coding")
    print("=" * 80)

    # Configuration
    n_p = [96, 80, 64]  # Place cells per frequency
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Frequencies: {len(n_p)}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Batch size: {batch_size}")

    # Create grounded location inference module
    grounded = GroundedLocInference()
    print(f"\n✓ Grounded inference module initialized")

    # Simulate expanded inputs (from upstream Projection and SensoryProjection)
    g_expanded = [torch.randn(batch_size, n) for n in n_p]
    x_expanded = [torch.randn(batch_size, n) for n in n_p]

    print(f"✓ Simulated inputs:")
    print(f"  g_expanded (grid cells): {[g.shape for g in g_expanded]}")
    print(f"  x_expanded (sensory): {[x.shape for x in x_expanded]}")

    # Compute grounded location
    with torch.no_grad():
        p = grounded(g_expanded, x_expanded)

    print(f"✓ Grounded location: {[p_f.shape for p_f in p]}")

    # Statistics
    print(f"\nActivation statistics:")
    for f in range(len(n_p)):
        print(f"  Freq {f}: mean={p[f].mean():.3f}, std={p[f].std():.3f}, " f"range=[{p[f].min():.3f}, {p[f].max():.3f}]")

    print("\n" + "=" * 80)
    print("TEM Pipeline:")
    print("  1. Projection: g → g_expanded (Kronecker repeat)")
    print("  2. SensoryProjection: x → x_expanded (Kronecker tile)")
    print("  3. GroundedLocInference: g_expanded ⊙ x_expanded → p")
    print("  4. MemoryStorage: (p_inferred, p_generated) → M (Hebbian)")
    print("  5. AttractorDynamics: (p_query, M) → p_retrieved (recall)")
    print("=" * 80)
