"""MEC path integration: action-driven grid cell transitions."""

from __future__ import annotations

from typing import List

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

    def __init__(self, n_a: int, mec_shape: List[int], f_init: List[float], settings: PathSettings):
        super().__init__()
        self._mec_shape, self._n_freq = mec_shape, len(mec_shape)
        self._connections = conn = utils.connections(f_init)
        self._conn_indices = [[f_from for f_from in range(self.n_freq) if conn[f_to][f_from]] for f_to in range(self.n_freq)]
        self._in_dims = [sum(mec_shape[f_from] for f_from in self._conn_indices[f_to]) for f_to in range(self.n_freq)]
        self._mat_shape = [(self._in_dims[f_to], mec_shape[f_to]) for f_to in range(self.n_freq)]
        self._settings = settings

        # Transition weights (action-conditioned)
        hidden_dim = [settings.hidden_dim] * self.n_freq
        self.MLP_D_a = MLP([n_a] * self.n_freq, self.shape, [torch.tanh, None], hidden_dim, bias=[True, False])
        self.MLP_D_a.set_weights(1, 0.0)
        self.D_no_a = nn.ParameterList([nn.Parameter(torch.zeros(m)) for m in self._mat_shape])  # Non-directional, per-frequency matrix

        # Transition uncertainty
        self.MLP_sigma_g_path = MLP(mec_shape, mec_shape, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in mec_shape])

    @property
    def settings(self) -> PathSettings:
        """Path integration settings."""
        return self._settings

    @property
    def shape(self) -> List[int]:
        """Flattened transition sizes per frequency module."""
        return [in_dim * out_dim for in_dim, out_dim in self._mat_shape]

    @property
    def n_freq(self) -> int:
        """Number of grid cell frequency modules."""
        return self._n_freq

    def forward(self, a: Tensor, g_prev: List[Tensor], no_direc_mask: Tensor | None = None) -> Transition:
        mu = self.mean(a, g_prev, no_direc_mask)
        sigma = self.MLP_sigma_g_path(g_prev)
        return Transition(mean=mu, uncertainty=sigma)

    def mean(self, a: Tensor, g: List[Tensor], no_direc_mask: Tensor | None) -> List[Tensor]:
        """Compute mean of transitioned grid cells."""
        mats = self._transition_matrices(a, no_direc_mask)

        # Build input by concatenating connected frequencies
        g_in = [torch.cat([g[f_from] for f_from in self._conn_indices[f_to]], dim=1).unsqueeze(1) for f_to in range(self.n_freq)]

        # Apply transition via batch matrix multiply
        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        return [g_f + delta_f for g_f, delta_f in zip(g, delta)]

    def _transition_matrices(self, a: Tensor, no_direc_mask: Tensor | None) -> List[Tensor]:
        """Build per-frequency transition matrices, optionally overriding with D_no_a."""
        d_flat = self.MLP_D_a([a] * self.n_freq)
        mats = [d[f].reshape(-1, *self._mat_shape[f]) for f, d in enumerate(d_flat)]

        if no_direc_mask is not None and torch.any(no_direc_mask):
            # Replace where the no-direction mask is active
            mask = no_direc_mask.view(-1, 1, 1)
            for f_to in range(self.n_freq):
                d_no_a = self.D_no_a[f_to].unsqueeze(0).expand_as(mats[f_to])
                mats[f_to] = torch.where(mask, d_no_a, mats[f_to])

        return mats
