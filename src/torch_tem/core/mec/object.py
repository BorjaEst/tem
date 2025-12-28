"""Object vector cells (OVCs) for landmark inference.

This module implements object vector cell inference, which provides
object identity information parallel to spatial location from grid cells.

OVCs respond to specific objects/landmarks regardless of location,
complementing the spatial codes from grid cells.

Supports two modes:
    - Merged mode (frequencies=None): OVCs share grid frequencies
    - Separate mode (frequencies=[...]): OVCs have independent frequency modules

Typical usage example:
    >>> config = ObjectInferenceConfig(n_ovc=[8, 6, 4], frequencies=[0.8, 0.5, 0.3])
    >>> ovc = ObjectInference(n_g=[48, 40, 32], n_g_ovc=[24, 18, 12], config=config)
    >>> g_ovc = ovc(g_gen, locations)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor

from torch_tem import utils
from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, Transition

__all__ = ["ObjectInferenceConfig", "ObjectInference"]


class ObjectInferenceConfig(BaseModel):
    """Configuration for object vector cell inference.

    Supports two modes via the `frequencies` field:
    - Merged mode: frequencies=None → OVCs share grid frequencies
    - Separate mode: frequencies=[...] → OVCs have independent frequency modules
    """

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # OVC dimensions
    n_ovc: List[int] = Field(default_factory=list, description="OVC neuron counts per frequency. Empty list = disabled.")

    # Mode selector (replaces separate_modules boolean)
    frequencies: Optional[List[float]] = Field(
        default=None, description="Frequency values for OVC modules. " "None = merged mode (share grid frequencies). " "List[float] = separate mode (independent OVC modules)."
    )

    # Hyperparameters
    hidden_multiplier: int = Field(default=2, ge=1, description="MLP hidden dimension multiplier: hidden_dim[f] = multiplier * n_ovc[f]")
    g_init_std: float = Field(default=0.5, gt=0, description="Initial OVC state std")


class ObjectInference(nn.Module):
    """Object inference: OVCs from landmark/shiny object cues.

    Supports two modes (determined by config.frequencies):
    1. Merged mode (frequencies=None): OVCs share grid frequencies
       - n_g_ovc indicates portion within each grid module
       - Forward returns [] (handled by AbstractLocModel)

    2. Separate mode (frequencies=[...]): OVCs have independent modules
       - n_g_ovc are full OVC module dimensions
       - Forward returns OVC activations
    """

    def __init__(self, n_g: List[int], n_g_ovc: List[int], config: Optional[ObjectInferenceConfig]):
        """Initialize object inference.

        Args:
            n_g: Total abstract location dimensions (from W_down context)
            n_g_ovc: OVC dimensions - either portion (merged) or full (separate)
            config: OVC configuration (None = disabled)
        """
        super().__init__()
        self._config = config
        self._n_g = n_g
        self._n_g_ovc = n_g_ovc

        if self.enabled:
            n_f_ovc = len(n_g_ovc)

            # Shiny → OVC MLPs (only for enabled separate OVC)
            self.mlp_mu_g_shiny = MLP(in_dim=[1] * n_f_ovc, out_dim=n_g_ovc, hidden_dim=[config.hidden_multiplier * g for g in n_g_ovc])
            self.mlp_sigma_g_shiny = MLP(in_dim=[1] * n_f_ovc, out_dim=n_g_ovc, activation=[torch.tanh, torch.exp], hidden_dim=[config.hidden_multiplier * g for g in n_g_ovc])

            # Learnable priors for OVC initialization
            self.g_init = nn.ParameterList([nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=g, loc=0, scale=config.g_init_std), dtype=torch.float)) for g in n_g_ovc])

            self.logsig_g_init = nn.ParameterList([nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=g, loc=0, scale=config.g_init_std), dtype=torch.float)) for g in n_g_ovc])

    @property
    def enabled(self) -> bool:
        """Check if OVC is enabled in separate mode."""
        return self._config is not None and len(self._n_g_ovc) > 0 and self._config.frequencies is not None  # None = merged mode

    @property
    def is_merged(self) -> bool:
        """Check if OVC is in merged mode (shares grid frequencies)."""
        return self._config is not None and len(self._n_g_ovc) > 0 and self._config.frequencies is None

    @property
    def n_f(self) -> int:
        """Number of OVC frequency modules."""
        return len(self._n_g_ovc) if self.enabled else 0

    @property
    def n_g_ovc(self) -> List[int]:
        """OVC dimensions per frequency."""
        return self._n_g_ovc if self.enabled else []

    def forward(self, g_gen: Transition, locations: List[Dict[str, Any]]) -> AbstractLocation:
        """Infer object location from landmarks/shiny cues.

        Args:
            g_gen: Path integration (fallback if no landmarks)
            locations: Environment descriptors with 'shiny' field

        Returns:
            Object vector cell activations [n_f_ovc] of [B, n_g_ovc[f]]
            Empty list [] if disabled or merged
        """
        if not self.enabled:
            return []  # No OVC or merged mode (handled by AbstractLocModel)

        # Check for shiny objects
        if any(shiny_present := [loc.get("shiny") is not None for loc in locations]):
            # Create binary shiny indicators
            shiny_binary = [[1.0] if s else [0.0] for s in shiny_present]
            shiny_tensor = [torch.tensor(shiny_binary, dtype=torch.float, device=g_gen.mean[0].device) for _ in range(self.n_f)]

            # Compute OVC response to shiny objects
            mu_g = self.mlp_mu_g_shiny(shiny_tensor)
            mu_g = [torch.abs(mu) for mu in mu_g]
            mu_g = self._apply_activation(mu_g)
            sigma_g = self.mlp_sigma_g_shiny(shiny_tensor)

            # Fuse with path integration (only OVC portion)
            # Extract OVC portion from g_gen if it contains both grid + ovc
            n_f_grid = len(g_gen.mean) - self.n_f
            if n_f_grid > 0:
                # g_gen has both grid and ovc
                g_gen_ovc = Transition(mean=g_gen.mean[n_f_grid:], uncertainty=g_gen.uncertainty[n_f_grid:])
            else:
                # g_gen is ovc-only
                g_gen_ovc = g_gen

            g_shiny = Transition(mean=mu_g, uncertainty=sigma_g)
            fused = utils.fuse_transitions([g_gen_ovc, g_shiny])
            return fused.mean
        else:
            # No landmarks, use path integration
            n_f_grid = len(g_gen.mean) - self.n_f
            if n_f_grid > 0:
                return g_gen.mean[n_f_grid:]  # OVC portion
            else:
                return g_gen.mean  # All OVC

    def _apply_activation(self, g: List[Tensor]) -> List[Tensor]:
        """Apply leaky ReLU activation (like place cells).

        Args:
            g: OVC activations before nonlinearity

        Returns:
            Activated OVC responses
        """
        return [torch.nn.functional.leaky_relu(torch.clamp(g_f, min=-1, max=1)) for g_f in g]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Object vector cell inference example.

    Demonstrates OVC inference for landmark/object recognition,
    supporting both merged and separate frequency modes.
    """
    print("=" * 80)
    print("Object Vector Cell (OVC) Inference Example")
    print("=" * 80)

    # Configuration
    n_g = [48, 40, 32]  # Total grid cells
    n_g_ovc = [24, 18, 12]  # OVC dimensions
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Total grid cells: {n_g}")
    print(f"  OVC dimensions: {n_g_ovc}")
    print(f"  Batch size: {batch_size}")

    # Example 1: Separate mode
    print(f"\n{'=' * 80}")
    print(f"Example 1: Separate OVC Mode")
    print(f"{'=' * 80}")

    config_separate = ObjectInferenceConfig(n_ovc=n_g_ovc, frequencies=[0.8, 0.5, 0.3], hidden_multiplier=2)  # Independent frequencies
    ovc_separate = ObjectInference(n_g, n_g_ovc, config_separate)
    print(f"✓ OVC model initialized (separate mode)")

    # Create inputs
    mu_gen = [torch.randn(batch_size, n) for n in n_g]
    sigma_gen = [torch.ones(batch_size, n) * 0.1 for n in n_g]
    g_gen = Transition(mean=mu_gen, uncertainty=sigma_gen)

    locations = [{"shiny": i % 2 == 0} for i in range(batch_size)]

    # Forward pass
    g_ovc = ovc_separate(g_gen, locations)
    print(f"✓ Forward pass complete")
    print(f"  Output shape: {[g.shape for g in g_ovc]}")

    # Example 2: Merged mode
    print(f"\n{'=' * 80}")
    print(f"Example 2: Merged OVC Mode")
    print(f"{'=' * 80}")

    config_merged = ObjectInferenceConfig(n_ovc=n_g_ovc, frequencies=None, hidden_multiplier=2)  # Share grid frequencies
    ovc_merged = ObjectInference(n_g, n_g_ovc, config_merged)
    print(f"✓ OVC model initialized (merged mode)")

    g_ovc_merged = ovc_merged(g_gen, locations)
    print(f"✓ Forward pass complete")
    print(f"  Output: {g_ovc_merged} (handled by AbstractLocModel)")

    print(f"\n{'=' * 80}")
    print(f"✓ OVC inference example complete")
    print(f"{'=' * 80}")
