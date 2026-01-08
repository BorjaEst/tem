"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem.core.mec.grid import GridModel, GridSettings
from torch_tem.core.mec.ovc import OVCModel, OVCSettings
from torch_tem.settings import MECSettings
from torch_tem.types import Transition


@dataclass
class MECState:
    """State container for MEC dynamics.

    Attributes:
        g_gen: Ancestral prediction for the generative pathway.
        g_path: Path integration prior (mean, uncertainty) for inference.
        g: Infered abstract location.
    """

    g_gen: List[Tensor]
    g_path: Transition
    g: Optional[List[Tensor]] = None

    def __post_init__(self):
        """Ensure g is always defined (defaults to g_path.mean)."""
        if self.g is None:
            self.g = list(self.g_path.mean)


class MECModel(nn.Module):
    """MEC path integration.

    This module implements the legacy transition model:
    - compute mu_g via action-conditioned transition
    - compute sigma_g via MLP_sigma_g_path
    - sample g if do_sample else take mu_g
    - for shiny environments, generative branch uses non-directional weights
      (legacy: if *any* shiny env exists, g_gen is recomputed for *all* envs)
    """

    def __init__(
        self,
        n_g: List[int],
        n_f_g: int,
        n_f_ovc: int,
        n_a: int,
        settings: Optional[MECSettings] = None,
        f_init: Optional[List[float]] = None,
    ):
        settings = settings or MECSettings()
        super().__init__()

        # Store for backward compatibility with methods that reference self.n_g
        self.n_g = n_g
        self.n_a = n_a

        # Initialize GridModel
        self.grid_model = GridModel(n_a, n_g, settings=settings.grid_cells, f_init=f_init)

        # Create OVCSettings from MECSettings
        # Compute OVC output dims outside OVCModel (legacy-equivalent).
        # - If OVC modules are separate (n_f_ovc > 0), only the OVC modules get shiny outputs.
        # - Otherwise, all modules get shiny outputs.
        separate_ovc = n_f_ovc > 0
        n_ovc = n_g[n_f_g:] if separate_ovc else n_g

        ovc_settings = OVCSettings(separate_ovc=separate_ovc)
        self.ovc_model = OVCModel(n_ovc=n_ovc, settings=ovc_settings)

    # ---------------------------------------------------------------------
    # Public API (compatible with model.py)
    # ---------------------------------------------------------------------

    def forward(self, a: Tensor, state: MECState, locations: list[dict]) -> Tuple[List[Tensor], MECState]:
        """Compute next MEC state from action-driven transition.

        Args:
            a: One-hot encoded actions (B, n_a). With has_static_action=True,
               action 0 (stand still) is encoded as all-zeros.
            state: Current MEC state
            locations: Per-env location dicts (for shiny detection)

        Returns:
            New MECState with updated g_gen and g_path.

        Note:
            Caller is responsible for resetting state.g to g_init at episode boundaries.
            This module always applies transition dynamics from the provided state.
        """
        # Shiny envs use no_direc=True (no action-driven transitions)
        no_direc = [loc.get("shiny") is not None for loc in locations]
        g_gen, transition = self.grid_model(a, state.g, no_direc=no_direc)

        return g_gen, MECState(g_gen=g_gen, g_path=transition)

    # ---------------------------------------------------------------------
    # Exposed attributes expected by TEMModel (legacy API surface)
    # ---------------------------------------------------------------------

    @property
    def g_init(self):
        """Expose grid initialization mean for TEMModel compatibility."""
        return self.grid_model.g_init_mean

    @property
    def logsig_g_init(self):
        """Expose grid initialization log-std for TEMModel compatibility."""
        return self.grid_model.g_init_logstd

    @property
    def MLP_mu_g_shiny(self):
        """Expose OVC shiny mean MLP for TEMModel compatibility."""
        return self.ovc_model.MLP_mu_g_shiny

    @property
    def MLP_sigma_g_shiny(self):
        """Expose OVC shiny sigma MLP for TEMModel compatibility."""
        return self.ovc_model.MLP_sigma_g_shiny

    def g_clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp grid cell activations for stability."""
        return self.grid_model.g_clamp(g)
