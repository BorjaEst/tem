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
from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, GroundedLocation, Transition

__all__ = ["AbstractLocConfig", "AbstractLocModel"]


class AbstractLocConfig(BaseModel):
    """Configuration for spatial (grid cell) inference."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # Architecture
    hidden_multiplier: int = Field(default=2, ge=1, description="MLP hidden dimension multiplier: hidden_dim[f] = multiplier * n_g[f]")

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

    def __init__(self, n_g: List[int], n_p: List[int], config: AbstractLocConfig):
        """Initialize spatial inference.

        Args:
            n_g: Grid cell dimensions per frequency
            n_p: Place cell dimensions per frequency
            config: Spatial inference configuration
        """
        super().__init__()
        self._config = config
        self._n_g = n_g
        self._n_f = len(n_g)

        # Memory → Grid MLP (p_x → mu_g_mem)
        # Takes concatenated place cells, outputs per-frequency grid cells
        self.mlp_mu_g_mem = MLP(in_dim=[sum(n_p)] * self.n_f, out_dim=n_g, hidden_dim=[config.hidden_multiplier * g for g in n_g])

        # Initialize with small random weights (legacy parity)
        weights = self.mlp_mu_g_mem.get_weights(-1)
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * config.g_mem_std for w in weights])

        # Memory uncertainty MLP (retrieval quality → sigma_g_mem)
        # Input: [norm, reconstruction_error] per frequency
        self.mlp_sigma_g_mem = MLP(in_dim=[2] * self._n_f, out_dim=n_g, activation=[torch.tanh, torch.exp], hidden_dim=[config.hidden_multiplier * g for g in n_g])

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return self._n_f

    @property
    def n_g(self) -> List[int]:
        """Grid cell dimensions per frequency."""
        return self._n_g

    def forward(self, g_gen: Transition, p_x: Optional[GroundedLocation]) -> AbstractLocation:
        """Infer spatial location from path integration + memory.

        Args:
            g_gen: Path integration prediction (always available)
            p_x: Memory retrieval from sensory (inference mode only)

        Returns:
            Fused grid cell location [n_f] of [B, n_g[f]]
        """
        estimates = [g_gen]

        # Add memory correction if available
        if p_x is not None:
            g_mem = self.memory_estimate(p_x)
            estimates.append(g_mem)

        # Fuse with precision weighting
        fused = utils.fuse_transitions(estimates)
        return utils.sample_transition(fused) if self._config.do_sample else fused.mean

    def memory_estimate(self, p_x: GroundedLocation) -> Transition:
        """Compute grid location from memory retrieval.

        Args:
            p_x: Hippocampal pattern from sensory retrieval

        Returns:
            Memory-derived grid cell estimate with uncertainty
        """
        # Concatenate all frequencies for MLP input
        p_x_concat = torch.cat(p_x, dim=-1)

        # Predict mean: p_x → g_mem
        mu_g_mem = self.mlp_mu_g_mem([p_x_concat] * self._n_f)

        # Predict uncertainty from retrieval quality
        quality_indicators = self.retrieval_quality(p_x)
        sigma_g_mem_base = self.mlp_sigma_g_mem(quality_indicators)

        # Apply scheduling offset (curriculum learning)
        # Higher offset → lower precision → less memory influence
        sigma_g_mem = [sig + self._config.p2g_scale_offset * self._config.p2g_sig_val for sig in sigma_g_mem_base]

        return Transition(mean=mu_g_mem, uncertainty=sigma_g_mem)

    def retrieval_quality(self, p_x: GroundedLocation) -> List[Tensor]:
        """Compute quality indicators for memory retrieval.

        Args:
            p_x: Retrieved place cell pattern

        Returns:
            Quality indicators [n_f] of [B, 2] (norm, reconstruction_error)
        """
        quality_fn = lambda f: [p_x[f].norm(dim=-1), torch.zeros_like(p_x[f][:, 0])]
        return [torch.stack(quality_fn(f), dim=-1) for f in range(self.n_f)]
