"""Object vector cells (OVCs) for landmark inference.

This module implements object vector cell inference, which provides
object identity information parallel to spatial location from grid cells.

OVCs respond to specific objects/landmarks regardless of location,
complementing the spatial codes from grid cells.

OVC portions are allocated backwards from the end of n_g_grid:
    - Merged mode (frequencies=[]): OVCs share grid frequencies
    - Separate mode (frequencies=[...]): OVCs have independent frequency modules

Examples:
    # Full merged: n_g=[20,30,40], n_g_ovc=[10,10,10] → n_g_grid=[10,20,30]
    >>> config = ObjectInferenceConfig(n_g_ovc=[10, 10, 10], frequencies=[])
    >>> ovc = ObjectInference(n_g=[20, 30, 40], config=config)

    # Partial merged: n_g=[10,30,40], n_g_ovc=[10,10] → n_g_grid=[10,20,30]
    >>> config = ObjectInferenceConfig(n_g_ovc=[10, 10], frequencies=[])
    >>> ovc = ObjectInference(n_g=[10, 30, 40], config=config)

    # Merged + separate: n_g=[10,30,40,10], n_g_ovc=[10,10,10], f_ovc=[0.1] → n_g_grid=[10,20,30]
    >>> config = ObjectInferenceConfig(n_g_ovc=[10, 10, 10], frequencies=[0.1])
    >>> ovc = ObjectInference(n_g=[10, 30, 40, 10], config=config)
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor

from torch_tem import utils
from torch_tem.modules.mlp import MLP
from torch_tem.types import AbstractLocation, Transition

__all__ = ["ObjectInferenceConfig", "ObjectInference"]


class ObjectInferenceConfig(BaseModel):
    """Configuration for object vector cell inference.

    OVC portions allocated backwards from end of n_g_grid:

    Examples:
        Full merged: n_g=[20,30,40], n_g_ovc=[10,10,10], f_ovc=[] → n_g_grid=[10,20,30]
        Partial merged: n_g=[10,30,40], n_g_ovc=[10,10], f_ovc=[] → n_g_grid=[10,20,30]
        Merged+separate: n_g=[10,30,40,10], n_g_ovc=[10,10,10], f_ovc=[0.1] → n_g_grid=[10,20,30]
        Full separate: n_g=[10,20,30,10], n_g_ovc=[10], f_ovc=[0.1] → n_g_grid=[10,20,30]

    Modes (determined by frequencies):
        Merged: frequencies=[] → OVCs share grid frequencies
        Separate: frequencies=[...] → OVCs have independent modules
    """

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # OVC dimensions (allocated backwards from end of n_g_grid)
    n_g_ovc: List[int] = Field(default_factory=list, description="OVC portions to allocate backwards from grid modules")
    frequencies: List[float] = Field(default_factory=list, description="Frequency values for separate OVC modules. [] = merged mode, [0.1, ...] = separate mode")

    # Hyperparameters
    hidden_multiplier: int = Field(default=2, ge=1, description="MLP hidden dimension multiplier: hidden_dim[f] = multiplier * n_g_ovc[f]")
    g_init_std: float = Field(default=0.5, gt=0, description="Initial OVC state std")


class ObjectInference(nn.Module):
    """Object inference: OVCs from landmark/shiny object cues.

    Always initializes OVC infrastructure (even if empty).
    Returns separate OVC modules if frequencies != [], else returns [].
    """

    def __init__(self, n_g: List[int], config: ObjectInferenceConfig):
        """Initialize object inference.

        Args:
            n_g: Total abstract location dimensions (from W_down context)
            config: OVC configuration (n_g_ovc=[] means no OVC)
        """
        super().__init__()
        self._config = config
        self._n_g = n_g

        # Extract separate OVC dimensions (last n_f_ovc_separate elements)
        # In backward allocation: n_g_ovc=[10,10] with frequencies=[0.25] means
        #   - First [10] is merged into grid module
        #   - Last [10] is separate OVC module
        # MLPs should only be created for separate portion
        n_f_ovc_separate = len(config.frequencies)
        n_g_ovc_all = config.n_g_ovc
        n_g_ovc_separate = n_g_ovc_all[-n_f_ovc_separate:] if n_f_ovc_separate > 0 else []

        # Shiny → OVC MLPs (only for separate modules)
        self.mlp_mu_g_shiny = MLP(in_dim=[1] * n_f_ovc_separate, out_dim=n_g_ovc_separate, hidden_dim=[config.hidden_multiplier * g for g in n_g_ovc_separate])
        self.mlp_sigma_g_shiny = MLP(
            in_dim=[1] * n_f_ovc_separate, out_dim=n_g_ovc_separate, activation=[torch.tanh, torch.exp], hidden_dim=[config.hidden_multiplier * g for g in n_g_ovc_separate]
        )

        # Learnable priors (only for separate modules)
        self.g_init = nn.ParameterList([nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=g, loc=0, scale=config.g_init_std), dtype=torch.float)) for g in n_g_ovc_separate])
        self.logsig_g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=g, loc=0, scale=config.g_init_std), dtype=torch.float)) for g in n_g_ovc_separate]
        )

    @property
    def n_f_ovc(self) -> int:
        """Number of separate OVC modules."""
        return len(self._config.frequencies)

    @property
    def n_g_ovc(self) -> List[int]:
        """OVC dimensions from config."""
        return self._config.n_g_ovc

    def forward(self, g_gen: Transition, locations: List[Dict[str, Any]]) -> AbstractLocation:
        """Infer OVC activations from landmarks.

        Returns:
            Separate OVC modules if frequencies != [], else []
        """
        # Return empty if no separate OVC modules
        if not self._config.frequencies:
            return []

        # Check for shiny objects
        if any(shiny_present := [loc.get("shiny") is not None for loc in locations]):
            # Compute OVC responses
            shiny_binary = [[1.0] if s else [0.0] for s in shiny_present]
            shiny_tensor = [torch.tensor(shiny_binary, dtype=torch.float, device=g_gen.mean[0].device) for _ in range(self.n_f_ovc)]

            mu_g = self.mlp_mu_g_shiny(shiny_tensor)
            mu_g = [torch.abs(mu) for mu in mu_g]
            mu_g = self._apply_activation(mu_g)
            sigma_g = self.mlp_sigma_g_shiny(shiny_tensor)

            # Fuse with path integration (OVC portion only)
            n_f_grid = len(g_gen.mean) - self.n_f_ovc
            g_gen_ovc = Transition(mean=g_gen.mean[n_f_grid:], uncertainty=g_gen.uncertainty[n_f_grid:])

            g_shiny = Transition(mean=mu_g, uncertainty=sigma_g)
            fused = utils.fuse_transitions([g_gen_ovc, g_shiny])
            return fused.mean
        else:
            # No landmarks - return path integration OVC portion
            n_f_grid = len(g_gen.mean) - self.n_f_ovc
            return g_gen.mean[n_f_grid:]

    def _apply_activation(self, g: List[Tensor]) -> List[Tensor]:
        """Apply leaky ReLU activation."""
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
    batch_size = 4

    # Example 1: Full merged
    # n_g=[20,30,40], n_g_ovc=[10,10,10] → n_g_grid=[10,20,30]
    print(f"\n{'=' * 80}")
    print(f"Example 1: Full Merged Mode")
    print(f"{'=' * 80}")
    n_g = [20, 30, 40]
    config_merged = ObjectInferenceConfig(n_g_ovc=[10, 10, 10], frequencies=[], hidden_multiplier=2)
    ovc_merged = ObjectInference(n_g, config_merged)
    print(f"✓ OVC model initialized (merged mode)")
    print(f"  n_g: {n_g}")
    print(f"  n_g_ovc: {config_merged.n_g_ovc}")
    print(f"  Expected n_g_grid: [10, 20, 30]")

    # Example 2: Separate mode
    # n_g=[10,30,40,10], n_g_ovc=[10,10,10], f_ovc=[0.1] → n_g_grid=[10,20,30]
    print(f"\n{'=' * 80}")
    print(f"Example 2: Merged + Separate Mode")
    print(f"{'=' * 80}")
    n_g = [10, 30, 40, 10]
    config_separate = ObjectInferenceConfig(n_g_ovc=[10, 10, 10], frequencies=[0.1], hidden_multiplier=2)
    ovc_separate = ObjectInference(n_g, config_separate)
    print(f"✓ OVC model initialized (separate mode)")
    print(f"  n_g: {n_g}")
    print(f"  n_g_ovc: {config_separate.n_g_ovc}")
    print(f"  f_ovc: {config_separate.frequencies}")
    print(f"  Expected n_g_grid: [10, 20, 30]")

    # Create inputs
    mu_gen = [torch.randn(batch_size, n) for n in n_g]
    sigma_gen = [torch.ones(batch_size, n) * 0.1 for n in n_g]
    g_gen = Transition(mean=mu_gen, uncertainty=sigma_gen)

    locations = [{"shiny": i % 2 == 0} for i in range(batch_size)]

    # Forward pass
    g_ovc = ovc_separate(g_gen, locations)
    print(f"\n✓ Forward pass complete")
    print(f"  Output shape: {[g.shape for g in g_ovc]}")
    print(f"  Separate OVC modules only (merged handled by AbstractLocModel)")

    print(f"\n{'=' * 80}")
    print(f"✓ OVC inference example complete")
    print(f"{'=' * 80}")
