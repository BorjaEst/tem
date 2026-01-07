"""Object vector cell (OVC) inference: landmark-based spatial anchoring.

Provides object/landmark identity codes that complement grid cell spatial codes.
OVCs respond to specific objects regardless of location.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import MECSettings
from torch_tem.types import AbstractLocation, Transition


class ObjectInference(nn.Module):
    """Object vector cell inference from landmark cues.

    Handles shiny/landmark detection and fuses object signals with
    path integration for object-associated grid modules.

    Architecture:
    - If n_ovc is empty: no OVC processing
    - If f_ovc is non-empty: separate OVC modules appended
    - If f_ovc is empty: OVC dims merged into tail of grid modules
    """

    def __init__(
        self,
        *,
        n_g_grid: List[int],
        settings: Optional[MECSettings] = None,
    ):
        """Initialize object inference.

        Args:
            n_g_grid: Grid cell dimensions (before OVC allocation).
            settings: MECSettings with n_ovc and f_ovc configuration.
        """
        super().__init__()
        self.settings = settings or MECSettings()

        # Determine OVC structure
        n_ovc = self.settings.n_ovc
        f_ovc = self.settings.f_ovc

        if not n_ovc:
            # No OVC
            self._n_ovc_modules = 0
            self._ovc_dims = []
            return

        if f_ovc:
            # Separate OVC modules
            self._n_ovc_modules = len(f_ovc)
            self._ovc_dims = n_ovc
        else:
            # Merged OVC (no separate modules)
            self._n_ovc_modules = 0
            self._ovc_dims = []
            return

        # Create MLPs for separate OVC modules only
        if self._n_ovc_modules > 0:
            # Shiny indicator → mu_g_ovc
            self.mlp_mu_g_shiny = MLP(
                in_dim=[1] * self._n_ovc_modules,
                out_dim=self._ovc_dims,
                activation=(torch.nn.functional.elu, None),
            )

            # Shiny indicator → sigma_g_ovc
            self.mlp_sigma_g_shiny = MLP(
                in_dim=[1] * self._n_ovc_modules,
                out_dim=self._ovc_dims,
                activation=(torch.nn.functional.elu, torch.nn.functional.softplus),
            )

            # Learnable priors
            self.g_init = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(
                            truncnorm.rvs(-2, 2, loc=0, scale=self.settings.g_init_std, size=self._ovc_dims[f]),
                            dtype=torch.float32,
                        )
                    )
                    for f in range(self._n_ovc_modules)
                ]
            )

            self.logsig_g_init = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.tensor(
                            truncnorm.rvs(-2, 2, loc=0, scale=self.settings.g_init_std, size=self._ovc_dims[f]),
                            dtype=torch.float32,
                        )
                    )
                    for f in range(self._n_ovc_modules)
                ]
            )

    @property
    def has_ovc(self) -> bool:
        """Whether this model has separate OVC modules."""
        return self._n_ovc_modules > 0

    def forward(
        self,
        g_path: Transition,
        locations: List[Dict[str, Any]],
        ovc_module_start: int,
    ) -> Transition:
        """Infer OVC activations and fuse with path integration.

        Args:
            g_path: Path integration for all modules (grid + OVC).
            locations: List of location dicts with optional 'shiny' key.
            ovc_module_start: Index where OVC modules start (n_f_grid).

        Returns:
            Updated transition with OVC fusion applied to OVC modules.
        """
        # No-op if no separate OVC modules
        if not self.has_ovc:
            return g_path

        # Check for shiny objects
        shiny_present = torch.tensor(
            [loc.get("shiny") is not None for loc in locations],
            dtype=torch.bool,
            device=g_path.mean[0].device,
        )

        if not torch.any(shiny_present):
            # No landmarks - return path integration unchanged
            return g_path

        # Compute OVC responses for shiny environments
        batch_size = g_path.mean[0].shape[0]
        shiny_indicator = [shiny_present.float().unsqueeze(-1) for _ in range(self._n_ovc_modules)]

        mu_g_shiny = self.mlp_mu_g_shiny(shiny_indicator)
        sigma_g_shiny = self.mlp_sigma_g_shiny(shiny_indicator)

        # Fuse with path integration for OVC modules only
        mu_fused = list(g_path.mean)
        sigma_fused = list(g_path.uncertainty)

        for i in range(self._n_ovc_modules):
            ovc_idx = ovc_module_start + i

            # Apply fusion only to shiny environments
            mu_path_ovc = g_path.mean[ovc_idx]
            sigma_path_ovc = g_path.uncertainty[ovc_idx]

            # Fuse via inverse-variance weighting
            mu_combined, sigma_combined = utils.inv_var_weight(
                [mu_path_ovc[shiny_present], mu_g_shiny[i][shiny_present]],
                [sigma_path_ovc[shiny_present], sigma_g_shiny[i][shiny_present]],
            )

            # Update only shiny rows
            mu_fused[ovc_idx] = torch.where(
                shiny_present.unsqueeze(-1),
                mu_combined if mu_combined.shape[0] == batch_size else mu_path_ovc,
                mu_path_ovc,
            )
            sigma_fused[ovc_idx] = torch.where(
                shiny_present.unsqueeze(-1),
                sigma_combined if sigma_combined.shape[0] == batch_size else sigma_path_ovc,
                sigma_path_ovc,
            )

        return Transition(mean=mu_fused, uncertainty=sigma_fused)
