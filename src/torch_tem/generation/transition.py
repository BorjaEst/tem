"""Transition model for torch_tem package."""

from typing import List, Protocol, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, Transition


class TransitionParams(Protocol):
    """Minimal interface for TransitionModel.

    Dependencies: n_f, n_g, n_actions, g_connections,
                  do_sample, g_init_std, g_mem_std, d_hidden_dim
    Complexity: Medium (8 parameters)
    """

    n_actions: int
    do_sample: bool
    g_init_std: float
    g_mem_std: float
    d_hidden_dim: int
    n_f: int
    n_g: List[int]


class TransitionModel(nn.Module):
    """Predicts next abstract location from current location and action.

    Handles:
    - Action-based transitions (via MLP_D_a)
    - No-action transitions for shiny environments (via D_no_a)
    - Hierarchical connections between frequency modules
    - Uncertainty estimation (sigma_g)
    """

    def __init__(self, params: TransitionParams, g_connections: List[List[bool]]):
        """Initialize transition model.

        Args:
            params: Configuration satisfying TransitionParams protocol
        """
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g
        self.n_actions = params.n_actions
        self.do_sample = params.do_sample
        self.g_connections = g_connections

        # MLP for action-based transitions
        self.MLP_D_a = MLP(
            in_dim=[self.n_actions] * self.n_f,
            out_dim=[sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]]) * self.n_g[f_to] for f_to in range(self.n_f)],
            activation=[torch.tanh, None],
            hidden_dim=[params.d_hidden_dim] * self.n_f,
            bias=[True, False],
        )
        # Initialize to identity (no change initially)
        self.MLP_D_a.set_weights(1, 0.0)

        # No-action transition weights for shiny environments
        self.D_no_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]]) * self.n_g[f_to])) for f_to in range(self.n_f)]
        )

        # Uncertainty estimation MLPs
        self.MLP_sigma_g_path = MLP(in_dim=[self.n_actions] * self.n_f, out_dim=self.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[params.d_hidden_dim] * self.n_f)

    def transition_with_action(self, g_prev: AbstractLocation, a: Tensor) -> Transition:
        """Compute transition using action: g_t+1 = g_t + D(a) * g_connections.

        Args:
            g_prev: Previous abstract location [n_f] of [B, n_g[f]]
            a: Action tensor [B, n_actions] or [B] (indices)

        Returns:
            Transition: (mu_g, sigma_g) tuple
        """
        # Convert action indices to one-hot if needed
        if a.dim() == 1 or a.shape[1] == 1:
            a_onehot = torch.nn.functional.one_hot(a.squeeze().long(), num_classes=self.n_actions).float()
        else:
            a_onehot = a

        mu_g = []
        # MLP expects list input if it has multiple modules
        D_a = self.MLP_D_a([a_onehot] * self.n_f)  # Replicate action for each frequency

        for f_to in range(self.n_f):
            # Gather inputs from connected frequencies
            g_inputs = torch.cat([g_prev[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1)

            # Reshape transition matrix and apply
            n_inputs = sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]])
            D_f = D_a[f_to].view(-1, n_inputs, self.n_g[f_to])

            # Apply transition: g_new = g_old + transition
            delta_g = torch.bmm(g_inputs.unsqueeze(1), D_f).squeeze(1)
            mu_g.append(g_prev[f_to] + delta_g)

        # Compute uncertainty
        sigma_g = self.MLP_sigma_g_path([a_onehot] * self.n_f)  # Replicate action for each frequency

        return mu_g, sigma_g

    def transition_no_action(self, g_prev: AbstractLocation) -> Transition:
        """Compute transition without action (for shiny environments).

        Args:
            g_prev: Previous abstract location [n_f] of [B, n_g[f]]

        Returns:
            Transition: (mu_g, sigma_g) tuple
        """
        mu_g = []

        for f_to in range(self.n_f):
            g_inputs = torch.cat([g_prev[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1)

            n_inputs = sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]])
            D_f = self.D_no_a[f_to].view(n_inputs, self.n_g[f_to])

            delta_g = torch.matmul(g_inputs, D_f)
            mu_g.append(g_prev[f_to] + delta_g)

        # Fixed low uncertainty for no-action
        sigma_g = [torch.ones_like(g) * 0.1 for g in mu_g]

        return mu_g, sigma_g

    def sample(self, mu_g: AbstractLocation, sigma_g: AbstractLocation) -> AbstractLocation:
        """Sample from transition distribution if do_sample=True.

        Args:
            mu_g: Mean of abstract location
            sigma_g: Standard deviation

        Returns:
            g: Sampled (or mean) abstract location
        """
        if self.do_sample:
            return [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_g, sigma_g)]
        return mu_g

    def forward(self, g_prev: AbstractLocation, a: Tensor, use_action: bool = True) -> Transition:
        """Main forward pass.

        Args:
            g_prev: Previous abstract location
            a: Action (used if use_action=True)
            use_action: Whether to use action or no-action transition

        Returns:
            Transition: (g_gen, sigma_g) tuple
        """
        if use_action:
            mu_g, sigma_g = self.transition_with_action(g_prev, a)
        else:
            mu_g, sigma_g = self.transition_no_action(g_prev)

        g_gen = self.sample(mu_g, sigma_g)
        return g_gen, sigma_g
