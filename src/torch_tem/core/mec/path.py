"""MEC path integration: action-driven grid cell transitions."""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import PathSettings
from torch_tem.types import Transition


class PathIntegrator(nn.Module):
    """Path integration module: action-conditioned transitions with optional no-direction override.

    Implements grid cell path integration via learned action-driven transition matrices.
    Supports "shiny env" semantics where transitions are disabled (D_no_a used instead).
    """

    def __init__(self, n_a: int, n_g: List[int], f_init: List[float], settings: PathSettings):
        super().__init__()
        self._n_g, self._n_freq = n_g, len(n_g)
        self._connections = conn = utils.connections(f_init)
        self._settings = settings
        path_out_sizes = [sum(n_g[fb] for fb in range(self._n_freq) if conn[fa][fb]) * n_g[fa] for fa in range(self._n_freq)]
        hidden_dim = [settings.hidden_dim] * self._n_freq

        # Transition weights (action-conditioned)
        self.MLP_D_a = MLP([n_a] * self._n_freq, path_out_sizes, [torch.tanh, None], hidden_dim, bias=[True, False])
        self.MLP_D_a.set_weights(1, 0.0)

        # Non-directional transition weights (used for shiny generative branch)
        D_no_a_init = [torch.zeros(sum(n_g[f_from] for f_from in range(self._n_freq) if conn[f_to][f_from]) * n_g[f_to]) for f_to in range(self._n_freq)]
        self.D_no_a = nn.ParameterList([nn.Parameter(n) for n in D_no_a_init])

        # Transition uncertainty
        self.MLP_sigma_g_path = MLP(n_g, n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    @property
    def settings(self) -> PathSettings:
        """Path integration settings."""
        return self._settings

    def forward(self, a: Tensor, g_prev: List[Tensor], no_direc_mask: Tensor | None = None) -> Tuple[List[Tensor], Transition]:
        """Execute path integration step.

        Args:
            a: Action tensor (batch, n_a) one-hot encoded
            g_prev: Previous grid cell activations per frequency
            no_direc_mask: Boolean mask (batch,) indicating envs without action-driven transitions

        Returns:
            g_gen: Grid code for generative branch (previous g for shiny envs)
            transition: New path-integrated state (mean, uncertainty)
        """
        mu = self._transition_mean(a, g_prev, no_direc_mask)
        sigma = self._transition_uncertainty(g_prev)

        # Always return distribution (mean + uncertainty); parent decides sampling
        transition = Transition(mean=mu, uncertainty=sigma)

        # Legacy shiny behavior: g_gen uses no-direction transition for shiny envs
        if no_direc_mask is not None and torch.any(no_direc_mask):
            g_gen = self._transition_mean(a, g_prev, no_direc_mask)
        else:
            g_gen = mu

        return g_gen, transition

    def _transition_mean(self, a: Tensor, g: List[Tensor], no_direc_mask: Tensor | None) -> List[Tensor]:
        """Compute mean of transitioned grid cells."""
        mats = self._transition_matrices(a, no_direc_mask)

        # Build input by concatenating connected frequencies
        g_in = [torch.cat([g[f_from] for f_from in range(self._n_freq) if self._connections[f_to][f_from]], dim=1).unsqueeze(1) for f_to in range(self._n_freq)]

        # Apply transition via batch matrix multiply
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        g_next = [g_f + delta_f for g_f, delta_f in zip(g, delta)]

        return g_next

    def _transition_uncertainty(self, g: List[Tensor]) -> List[Tensor]:
        """Compute uncertainty of transition."""
        return self.MLP_sigma_g_path(g)

    def _transition_matrices(self, a: Tensor, no_direc_mask: Tensor | None) -> List[Tensor]:
        """Build per-frequency transition matrices, optionally overriding with D_no_a."""
        d_flat = self.MLP_D_a([a] * self._n_freq)

        if no_direc_mask is not None and torch.any(no_direc_mask):
            for f in range(self._n_freq):
                d_no_a = self.D_no_a[f].unsqueeze(0).expand_as(d_flat[f])
                d_flat[f] = torch.where(no_direc_mask.unsqueeze(1), d_no_a, d_flat[f])

        # Reshape flat vectors into matrices
        mats: List[Tensor] = []
        for f_to in range(self._n_freq):
            in_dim = sum(self._n_g[f] for f in range(self._n_freq) if self._connections[f_to][f])
            mats.append(d_flat[f_to].reshape(-1, in_dim, self._n_g[f_to]))

        return mats
