"""Spatial inference through grid cells.

This module implements the spatial component of abstract location inference,
combining path integration predictions with memory-corrected estimates.

Sources combined:
    1. Path integration (g_gen) - from transition model
    2. Memory retrieval (p_x → g_mem) - learned MLP mapping

Uses precision-weighted fusion to combine estimates based on uncertainty.

In generative mode (p_x=None), only path integration is used.
In inference mode (p_x provided), memory enables drift correction.

Typical usage example:
    >>> config = AbstractLocConfig(hidden_multiplier=2)
    >>> abstract = AbstractLocModel(n_g=[48, 40, 32], n_p=[96, 80, 64], config=config)
    >>> g = abstract(g_gen, p_x)
"""

from typing import List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem import utils
from torch_tem.modules.mlp import MLP
from torch_tem.types import AbstractLocation, GroundedLocation, MultiScaleCode, Transition

__all__ = ["AbstractLocConfig", "AbstractLocModel"]


class AbstractLocConfig(BaseModel):
    """Configuration for spatial (grid cell) inference."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # Architecture
    hidden_multiplier: int = Field(default=2, ge=1, description="MLP hidden dimension multiplier: hidden_dim[f] = multiplier * n_g[f]")

    # Legacy parity mode
    use_inverse_projection: bool = Field(default=False, description="If True, apply inverse projection (p @ W_repeat^T) before MLP (legacy mode)")

    # Hyperparameters
    g_mem_std: float = Field(default=0.1, gt=0, description="Memory MLP weight initialization std")
    p2g_scale_offset: float = Field(default=0.0, ge=0, description="Memory uncertainty offset (curriculum scheduling)")
    p2g_sig_val: float = Field(default=10000.0, ge=0, description="Base memory uncertainty magnitude")

    # Behavior
    do_sample: bool = Field(default=False, description="Sample from distributions vs use mean")


class AbstractLocModel(nn.Module):
    """Spatial inference: Fuse path integration + memory for grid cells.

    Combines:
    1. Path integration (g_gen from transition model)
    2. Memory-corrected estimate (p_x → g via learned MLP)

    Uses precision-weighted fusion (Bayesian combination).

    Theory:
        In GENERATIVE mode: p_x = None, only path integration used
        In INFERENCE mode: p_x retrieved from sensory, enables drift correction
    """

    def __init__(self, n_p: List[int], n_g: List[int], config: AbstractLocConfig):
        """Initialize spatial inference.

        Args:
            n_p: Place cell dimensions per frequency
            n_g: Grid cell dimensions per frequency (full resolution)
            config: Spatial inference configuration
        """
        super().__init__()
        self._config = config

        # Memory → Grid MLP (p_x → mu_g_mem or g_downsampled → mu_g_mem)
        self.mlp_mu_g_mem = MLP(in_dim=n_p, out_dim=n_g, hidden_dim=[config.hidden_multiplier * g for g in n_g])

        # Initialize with small random weights (legacy parity)
        weights = self.mlp_mu_g_mem.get_weights(-1)
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * config.g_mem_std for w in weights])

        # Memory uncertainty MLP (retrieval quality → sigma_g_mem)
        # Input: [norm, reconstruction_error] per frequency
        self.mlp_sigma_g_mem = MLP(in_dim=[2] * len(n_g), out_dim=n_g, activation=[torch.tanh, torch.exp], hidden_dim=[config.hidden_multiplier * g for g in n_g])

    def forward(self, g_gen: Transition, x: Optional[GroundedLocation], feedback: float = 0.0) -> AbstractLocation:
        """Infer spatial location from path integration + memory.

        Args:
            g_gen: Path integration prediction (always available)
            x: Retrieved place cell pattern (None = generative mode)

        Returns:
            Fused grid cell location [n_f] of [B, n_g[f]]
        """
        estimates = [g_gen]

        # Add memory correction if available
        if x is not None:
            g_mem = self.memory_estimate(x, error=feedback)
            estimates.append(g_mem)

        # Fuse with precision weighting
        fused = utils.fuse_transitions(estimates)
        return utils.sample_transition(fused) if self._config.do_sample else fused.mean

    def memory_estimate(self, x: MultiScaleCode, error: float) -> Transition:
        """Compute grid location from memory retrieval.

        Args:
            x: Retrieved place cell pattern
            error: Reconstruction error feedback

        Returns:
            Memory-derived grid cell estimate with uncertainty
        """
        # Predict mean: p_x → x^ → mu_g_mem
        mu_g_mem = self.mlp_mu_g_mem(x)

        # Predict uncertainty from retrieval quality
        quality_indicators = self.retrieval_quality(mu_g_mem, error)
        sigma_g_mem_base = self.mlp_sigma_g_mem(quality_indicators)

        # Apply scheduling offset (curriculum learning)
        # Higher offset → lower precision → less memory influence
        sigma_g_mem = [sig + self._config.p2g_scale_offset * self._config.p2g_sig_val for sig in sigma_g_mem_base]

        return Transition(mean=mu_g_mem, uncertainty=sigma_g_mem)

    def retrieval_quality(self, x: GroundedLocation) -> List[Tensor]:
        """Compute quality indicators for memory retrieval.

        Args:
            x: Retrieved place cell pattern

        Returns:
            Quality indicators [n_f] of [B, 2] (norm, reconstruction_error)
        """
        quality_fn = lambda f: [x[f].norm(dim=-1), torch.zeros_like(x[f][:, 0])]
        return [torch.stack(quality_fn(f), dim=-1) for f, _ in enumerate(x)]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Abstract location inference example: Fusing path integration with memory.

    Demonstrates how grid cells are inferred by combining path integration
    predictions with memory-based corrections.
    """
    print("=" * 80)
    print("Abstract Location Inference Example - Grid Cell Fusion")
    print("=" * 80)

    # Configuration
    n_g = [48, 40, 32]  # Grid cells per frequency
    n_p = [96, 80, 64]  # Place cells per frequency
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Grid cells: {n_g}")
    print(f"  Place cells: {n_p}")
    print(f"  Frequencies: {len(n_g)}")
    print(f"  Batch size: {batch_size}")

    # Create configuration and model
    config = AbstractLocConfig(hidden_multiplier=2, do_sample=False)
    abstract = AbstractLocModel(n_g, n_p, config)
    print(f"\n✓ Abstract location model initialized")

    # Create inputs
    # Path integration prediction
    mu_gen = [torch.randn(batch_size, n) for n in n_g]
    sigma_gen = [torch.ones(batch_size, n) * 0.1 for n in n_g]
    g_gen = Transition(mean=mu_gen, uncertainty=sigma_gen)

    # Memory-retrieved place cells
    p_x = [torch.randn(batch_size, n) for n in n_p]

    print(f"\n✓ Inputs created:")
    print(f"  Path integration: {[g.shape for g in g_gen.mean]}")
    print(f"  Memory retrieval: {[p.shape for p in p_x]}")

    # Forward pass
    g = abstract(g_gen, p_x)
    print(f"\n✓ Forward pass complete")
    print(f"  Output shape: {[g_f.shape for g_f in g]}")

    # Generative mode (no memory)
    g_gen_only = abstract(g_gen, None)
    print(f"\n✓ Generative mode (path integration only):")
    print(f"  Output shape: {[g_f.shape for g_f in g_gen_only]}")

    print(f"\n{'=' * 80}")
    print(f"✓ Abstract location inference example complete")
    print(f"{'=' * 80}")
