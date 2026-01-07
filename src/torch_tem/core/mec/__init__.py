"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import MECSettings
from torch_tem.types import Transition


@dataclass
class MECState:
    """State container for MEC dynamics."""

    g_gen: List[Tensor]  # Ancestral prediction for generative model
    g_path: Transition  # Path integration prior (mean, uncertainty) for inference


class MECModel(nn.Module):
    """MEC path integration with optional OVC support.

    Handles:
    - Action-driven transitions across frequency modules
    - Hierarchical connectivity (low → high frequency)
    - Optional object vector cells (append or merge modes)
    - State initialization and reset semantics

    External dependencies:
    - Projection (g -> g_ for memory) is handled externally

    """

    def __init__(
        self,
        n_p: List[int],
        n_g: List[int],
        n_g_subsampled: List[int],
        n_f: int,
        n_f_g: int,
        n_f_ovc: int,
        n_a: int,
        d_hidden: int,
        g_conn: list[list[bool]],
        g_init_std: float,
        g_mem_std: float,
        separate_ovc: bool,
        do_sample: bool,
        has_static_action: bool,
    ):
        super().__init__()

        # Initial activity of abstract location cells when entering action new environment, like action prior on g. Initialise with truncated normal
        self.g_init = torch.nn.ParameterList([torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=n_g[f], loc=0, scale=g_init_std), dtype=torch.float)) for f in range(n_f)])
        # Log of standard deviation of abstract location cells when entering action new environment; standard deviation of the prior on g. Initialise with truncated normal
        self.logsig_g_init = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=n_g[f], loc=0, scale=g_init_std), dtype=torch.float)) for f in range(n_f)]
        )
        # MLP for transition weights (not in paper, but recommended by James so you can learn about similarities between actions). Size is given by grid connections
        self.MLP_D_a = MLP(
            [n_a for _ in range(n_f)],
            [sum([n_g[f_from] for f_from in range(n_f) if g_conn[f_to][f_from]]) * n_g[f_to] for f_to in range(n_f)],
            activation=[torch.tanh, None],
            hidden_dim=[d_hidden for _ in range(n_f)],
            bias=[True, False],
        )
        # Initialise the hidden to output weights as zero, so initially you simply keep the current abstract location to predict the next abstract location
        self.MLP_D_a.set_weights(1, 0.0)
        # Transition weights without specifying an action for use in generative model with shiny objects
        self.D_no_a = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(sum([n_g[f_from] for f_from in range(n_f) if g_conn[f_to][f_from]]) * n_g[f_to])) for f_to in range(n_f)]
        )
        # MLP for standard deviation of transition sample
        self.MLP_sigma_g_path = MLP(n_g, n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])
        # MLP for standard devation of grounded location from retrieved memory sample
        self.MLP_sigma_p = MLP(n_p, n_p, activation=[torch.tanh, torch.exp])
        # MLP to generate mean of abstract location from downsampled abstract location, obtained by summing grounded location over sensory preferences in inference model
        self.MLP_mu_g_mem = MLP(n_g_subsampled, n_g, hidden_dim=[2 * g for g in n_g])
        # Initialise weights in last layer of MLP_mu_g_mem as truncated normal for each frequency module
        self.MLP_mu_g_mem.set_weights(
            -1,
            [torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=g_mem_std), dtype=torch.float) for f in range(n_f)],
        )
        # MLP to generate standard deviation of abstract location from two measures (generated observation error and inferred abstract location vector norm) of memory quality
        self.MLP_sigma_g_mem = MLP([2 for _ in n_g_subsampled], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])
        # MLP to generate mean of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_mu_g_shiny = MLP(
            [1 for _ in range(n_f_ovc if separate_ovc else n_f)],
            [n_g for n_g in n_g[(n_f_g if separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in n_g[(n_f_g if separate_ovc else 0) :]],
        )
        # MLP to generate standard deviation of abstract location directly from shiny object presence. Outputs to object vector cell modules if they're separated, else to all abstract location modules
        self.MLP_sigma_g_shiny = MLP(
            [1 for _ in range(n_f_ovc if separate_ovc else n_f)],
            [n_g for n_g in n_g[(n_f_g if separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in n_g[(n_f_g if separate_ovc else 0) :]],
            activation=[torch.tanh, torch.exp],
        )

        # Store other hyperparameters
        self.do_sample = do_sample
        self.n_a = n_a
        self.n_g = n_g
        self.n_f = n_f
        self.has_static_action = has_static_action
        self.g_connections = g_conn

    def forward(self, a, mec_state, locations) -> MECState:
        # Transition from previous abstract location to new abstract location using weights specific to action taken for each frequency module
        mu_g = self.f_mu_g_path(a, mec_state.g_path.mean)
        sigma_g = self.f_sigma_g_path(a, mec_state.g_path.mean)
        # Either sample new abstract location g or simply take the mean of distribution in noiseless case.
        g = [mu_g[f] + sigma_g[f] * torch.randn_like(mu_g[f]) if self.do_sample else mu_g[f] for f in range(self.n_f)]
        # But for environments with shiny objects, the transition to the new abstract location shouldn't have access to the action direction in the generative model
        shiny_envs = [location["shiny"] is not None for location in locations]
        # If there are any shiny environments, the abstract locations for the generative model will need to be re-calculated without providing actions for those
        g_gen = self.f_mu_g_path(a, mec_state.g_path.mean, no_direc=shiny_envs) if any(shiny_envs) else g
        # Return: (1) ancestral prediction g_gen for generative pathway, (2) path integration prior for inference
        return MECState(g_gen=g_gen, g_path=Transition(mean=g, uncertainty=sigma_g))

    def f_mu_g_path(self, a, g_prev, no_direc=None):
        # If there are no environments where the transition direction needs to be omitted (e.g. no shiny objects, or in inference model: set to all false
        no_direc = [False for _ in a] if no_direc is None else no_direc
        # Remove all Nones from a: these are walks where there was no previous action, so no step needs to be calculated for those
        a_prev_step = [action if action is not None else 0 for action in a]
        # And also keep track of which walks these valid step actions are for
        a_do_step = [action is not None for action in a]
        # Get device from g_prev
        device = g_prev[0].device
        # Transform list of actions into batch of one-hot row vectors.
        if self.has_static_action:
            # If this world has static actions: whenever action 0 (standing still) appears, the action vector should be all zeros. All other actions should have action 1 in the label-1 entry
            action = torch.zeros((len(a_prev_step), self.n_a), device=device).scatter_(
                1, torch.clamp(torch.tensor(a_prev_step, device=device).unsqueeze(1) - 1, min=0), 1.0 * (torch.tensor(a_prev_step, device=device).unsqueeze(1) > 0)
            )
        else:
            # Without static actions: each action label should become action one-hot vector for that label
            action = torch.zeros((len(a_prev_step), self.n_a), device=device).scatter_(1, torch.tensor(a_prev_step, device=device).unsqueeze(1), 1.0)
        # Get vector of transition weights by feeding actions into MLP
        D_a = self.MLP_D_a([action for _ in range(self.n_f)])
        # Replace transition weights by non-directional transition weights in environments where transition direction needs to be omitted (can set only if any no_direc)
        for f in range(self.n_f):
            D_a[f][no_direc, :] = self.D_no_a[f]
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        D_a = [
            torch.reshape(D_a[f_to], (-1, sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]]), self.n_g[f_to])) for f_to in range(self.n_f)
        ]
        # Select the frequency modules of the previous abstract location that are connected to each frequency module, to
        g_in = [torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1), 1) for f_to in range(self.n_f)]
        # Reshape transition weight vector into transition matrix. The number of rows in the transition matrix is given by the incoming abstract location connections for each frequency module
        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        # Not in the paper, but recommended by James for stability: use inferred code as *difference* in abstract location. Calculate new abstract location from previous abstract location and difference
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        # Not in paper, but recommended by James for stability: clamp activations between -1 and 1
        g_step = self.f_g_clamp(g_step)
        # Build new abstract location from result of transition if there was one, or from prior on abstract location if there wasn't
        return [torch.stack([g_step[f][batch_i, :] if do_step else self.g_init[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.n_f)]

    def f_sigma_g_path(self, a, g_prev):
        # Keep track of which walks these valid step actions are for
        a_do_step = [action != None for action in a]
        # Multi layer perceptron to generate standard deviation from all previous abstract locations, including those that were just initialised and not real previous locations
        from_g = self.MLP_sigma_g_path(g_prev)
        # And take exponent to get prior sigma for the walks that didn't have action previous location
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        # Now select the standard deviation generated from the previous abstract location if there was one, and the prior standard deviation on abstract location otherwise
        return [torch.stack([from_g[f][batch_i, :] if do_step else from_prior[f] for batch_i, do_step in enumerate(a_do_step)]) for f in range(self.n_f)]

    def f_g_clamp(self, g):
        # Calculate activation for abstract location, thresholding between -1 and 1
        activation = [torch.clamp(g_f, min=-1, max=1) for g_f in g]
        return activation
