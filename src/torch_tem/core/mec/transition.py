"""Transition module: action-driven path integration for abstract locations.

Simplified wrapper around _TransitionMean and _TransitionStd for standalone usage.
Uses MECSettings and Transition types for clean integration.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor, nn

from torch_tem.settings import MECSettings
from torch_tem.types import AbstractLocation, Transition


class TransitionModel(nn.Module):
    """Action-driven path integration for abstract locations.

    Thin wrapper around MECModel's transition components for:
    - Standalone demos/examples
    - Testing transition dynamics in isolation
    - Legacy API compatibility

    For production use, prefer MECModel directly.
    """

    def __init__(
        self,
        *,
        n_actions: int,
        n_g: List[int],
        f_initial: List[float],
        g_connections: Optional[List[List[bool]]] = None,
        settings: Optional[MECSettings] = None,
    ):
        """Initialize transition model.

        Args:
            n_actions: Number of discrete actions.
            n_g: Abstract location dimensions per frequency module.
            f_initial: Frequency values for hierarchical connectivity.
            g_connections: Optional custom connectivity matrix [n_f][n_f].
            settings: MECSettings (uses defaults if None).
        """
        super().__init__()

        # Import here to avoid circular dependency
        from torch_tem.core.mec import _default_g_connections_from_frequencies, _TransitionMean, _TransitionStd

        self.settings = settings or MECSettings()
        self._n_g = n_g
        self._n_f = len(n_g)

        # Build connectivity
        if g_connections is None:
            g_connections = _default_g_connections_from_frequencies(
                f_out=f_initial,
                n_f_g=self._n_f,
                n_f_ovc=0,
            )

        # Create transition components
        self.transition_mean = _TransitionMean(
            n_a=n_actions,
            n_out=n_g,
            g_connections=g_connections,
            d_hidden_dim=self.settings.d_hidden_dim,
        )
        self.transition_std = _TransitionStd(n_out=n_g)

    def forward(
        self,
        g_prev: AbstractLocation,
        a: Tensor,
        do_step: Optional[Tensor] = None,
        no_direc: Optional[Tensor] = None,
    ) -> Transition:
        """Compute transition from previous state and action.

        Args:
            g_prev: Previous abstract location [n_f] of [B, n_g[f]].
            a: Action tensor [B, n_actions] (one-hot or all-zero for static).
            do_step: Bool tensor [B] (True=continue, False=reset). Auto-detected if None.
            no_direc: Bool tensor [B] (True=use non-directional transition).

        Returns:
            Transition with mean and uncertainty for next abstract location.
        """
        # Auto-detect new walks if not specified
        if do_step is None:
            do_step = a.sum(dim=1) != 0

        # Compute transition
        mu_g = self.transition_mean(
            a=a,
            g_prev=g_prev,
            g_init=list(self.transition_mean.d_no_a),  # Use zeros as init
            do_step=do_step,
            no_direc=no_direc,
        )
        sigma_g = self.transition_std(
            g_prev=g_prev,
            logsig_g_init=[torch.zeros_like(g) for g in g_prev],  # Dummy init
            do_step=do_step,
        )

        # Sample if configured
        if self.settings.do_sample:
            g = [mu_g[f] + sigma_g[f] * torch.randn_like(mu_g[f]) for f in range(self._n_f)]
        else:
            g = mu_g

        return Transition(mean=g, uncertainty=sigma_g)
