from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm

from torch_tem import utils
from torch_tem.config import ArchitectureConfig, ModelConfig, StaticMatrices
from torch_tem.core.mlp import MLP
from torch_tem.core.states import AbstractLocationState, GroundedLocationState


class AbstractLocationModule(nn.Module):
    """Manages abstract location (grid cell) inference and generation."""

    def __init__(self, arch_config: ArchitectureConfig, model_config: ModelConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.arch = arch_config
        self.model = model_config
        self.matrices = static_matrices

        # Prior parameters
        self.g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=arch_config.n_g[f], loc=0, scale=arch_config.g_init_std), dtype=torch.float)) for f in range(arch_config.n_f)]
        )

        self.logsig_g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=arch_config.n_g[f], loc=0, scale=arch_config.g_init_std), dtype=torch.float)) for f in range(arch_config.n_f)]
        )

        # Path integration MLPs
        self.MLP_D_a = MLP(
            [arch_config.n_actions for _ in range(arch_config.n_f)],
            [
                sum([arch_config.n_g[f_from] for f_from in range(arch_config.n_f) if static_matrices.g_connections[f_to][f_from]]) * arch_config.n_g[f_to]
                for f_to in range(arch_config.n_f)
            ],
            activation=[torch.tanh, None],
            hidden_dim=[arch_config.d_hidden_dim for _ in range(arch_config.n_f)],
            bias=[True, False],
        )
        self.MLP_D_a.set_weights(1, 0.0)

        self.D_no_a = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(sum([arch_config.n_g[f_from] for f_from in range(arch_config.n_f) if static_matrices.g_connections[f_to][f_from]]) * arch_config.n_g[f_to])
                )
                for f_to in range(arch_config.n_f)
            ]
        )

        self.MLP_sigma_g_path = MLP(arch_config.n_g, arch_config.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in arch_config.n_g])

        # Memory-based inference MLPs
        self.MLP_mu_g_mem = MLP(arch_config.n_g_subsampled_combined, arch_config.n_g, hidden_dim=[2 * g for g in arch_config.n_g])
        self.MLP_mu_g_mem.set_weights(
            -1,
            [
                torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=arch_config.g_mem_std), dtype=torch.float)
                for f in range(arch_config.n_f)
            ],
        )

        self.MLP_sigma_g_mem = MLP(
            [2 for _ in arch_config.n_g_subsampled_combined], arch_config.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in arch_config.n_g]
        )

        # Shiny object MLPs
        self.MLP_mu_g_shiny = MLP(
            [1 for _ in range(arch_config.n_f_ovc if arch_config.separate_ovc else arch_config.n_f)],
            [n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
        )

        self.MLP_sigma_g_shiny = MLP(
            [1 for _ in range(arch_config.n_f_ovc if arch_config.separate_ovc else arch_config.n_f)],
            [n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            activation=[torch.tanh, torch.exp],
        )

    def generate_from_transition(self, g_prev: List[torch.Tensor], actions: List[int], locations: List[dict]) -> AbstractLocationState:
        """Generate abstract location through path integration."""
        mu_g = self._path_integration_mean(g_prev, actions)
        sigma_g = self._path_integration_sigma(g_prev, actions)

        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.model.do_sample else mu_g[f] for f in range(self.arch.n_f)]

        # Handle shiny objects (no directional information)
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if any(shiny_envs):
            mu_g = self._path_integration_mean(g_prev, actions, no_direc=shiny_envs)

        return AbstractLocationState(mu=mu_g, sigma=sigma_g, sources=["path_integration"])

    def infer(self, p_from_memory: Optional[List[torch.Tensor]], g_gen: AbstractLocationState, x: torch.Tensor, locations: List[dict]) -> AbstractLocationState:
        """Infer abstract location from memory and path integration."""
        # Start with path integration
        mu_g = g_gen.mu
        sigma_g = g_gen.sigma
        sources = ["path_integration"]

        # Add memory-based inference if enabled
        if self.model.use_p_inf and p_from_memory is not None:
            mu_g_mem, sigma_g_mem = self._infer_from_memory(p_from_memory, x)
            # Combine via inverse variance weighting
            mu_g, sigma_g = [], []
            for f in range(self.arch.n_f):
                mu, sigma = utils.inv_var_weight([g_gen.mu[f], mu_g_mem[f]], [g_gen.sigma[f], sigma_g_mem[f]])
                mu_g.append(mu)
                sigma_g.append(sigma)
            sources.append("memory")

        # Handle shiny objects
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if any(shiny_envs):
            mu_g, sigma_g = self._incorporate_shiny(mu_g, sigma_g, locations, shiny_envs)
            sources.append("shiny_objects")

        # Sample or take mean
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.model.do_sample else mu_g[f] for f in range(self.arch.n_f)]

        return AbstractLocationState(mu=g, sigma=sigma_g, sources=sources)

    def prepare_for_memory(self, g: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare abstract location for memory operations."""
        downsampled = [torch.matmul(g[f], self.matrices.g_downsample[f]) for f in range(self.arch.n_f)]
        return [torch.matmul(downsampled[f], self.matrices.W_repeat[f]) for f in range(self.arch.n_f)]

    def _path_integration_mean(self, g_prev: List[torch.Tensor], actions: List[int], no_direc: Optional[List[bool]] = None) -> List[torch.Tensor]:
        """Calculate mean through path integration."""
        no_direc = [False for _ in actions] if no_direc is None else no_direc
        a_step = [a if a is not None else 0 for a in actions]
        a_do_step = [a is not None for a in actions]

        # Actions to one-hot
        if self.arch.has_static_action:
            a_onehot = torch.zeros((len(a_step), self.arch.n_actions)).scatter_(
                1, torch.clamp(torch.tensor(a_step).unsqueeze(1) - 1, min=0), 1.0 * (torch.tensor(a_step).unsqueeze(1) > 0)
            )
        else:
            a_onehot = torch.zeros((len(a_step), self.arch.n_actions)).scatter_(1, torch.tensor(a_step).unsqueeze(1), 1.0)

        # Get transition matrices
        D_a = self.MLP_D_a([a_onehot for _ in range(self.arch.n_f)])

        # Replace with non-directional transitions where needed
        for f in range(self.arch.n_f):
            D_a[f][no_direc, :] = self.D_no_a[f]

        # Reshape and apply transitions
        D_a = [
            torch.reshape(D_a[f_to], (-1, sum([self.arch.n_g[f_from] for f_from in range(self.arch.n_f) if self.matrices.g_connections[f_to][f_from]]), self.arch.n_g[f_to]))
            for f_to in range(self.arch.n_f)
        ]

        g_in = [
            torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.arch.n_f) if self.matrices.g_connections[f_to][f_from]], dim=1), 1) for f_to in range(self.arch.n_f)
        ]

        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        g_step = [torch.clamp(g_f, min=-1, max=1) for g_f in g_step]

        return [torch.stack([g_step[f][i, :] if do_step else self.g_init[f] for i, do_step in enumerate(a_do_step)]) for f in range(self.arch.n_f)]

    def _path_integration_sigma(self, g_prev: List[torch.Tensor], actions: List[int]) -> List[torch.Tensor]:
        """Calculate uncertainty through path integration."""
        a_do_step = [a is not None for a in actions]
        from_g = self.MLP_sigma_g_path(g_prev)
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        return [torch.stack([from_g[f][i, :] if do_step else from_prior[f] for i, do_step in enumerate(a_do_step)]) for f in range(self.arch.n_f)]

    def _infer_from_memory(self, p: List[torch.Tensor], x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Infer abstract location from grounded location memory."""
        # Downsample p to get g
        g_downsampled = [torch.matmul(p[f], torch.t(self.matrices.W_repeat[f])) for f in range(self.arch.n_f)]
        mu_g_mem = self.MLP_mu_g_mem(g_downsampled)

        # Compute memory quality measures
        with torch.no_grad():
            from torch_tem.modules import ObservationGenerator  # Avoid circular import

            # Simplified error calculation
            err = torch.zeros(x.shape[0])

        sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]

        mu_g_mem = [torch.clamp(g_f, min=-1, max=1) for g_f in mu_g_mem]
        sigma_g_mem = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma_g_mem[f] + self.model.p2g_scale_offset * self.model.p2g_sig_val for f in range(self.arch.n_f)]

        return mu_g_mem, sigma_g_mem

    def _incorporate_shiny(
        self, mu_g: List[torch.Tensor], sigma_g: List[torch.Tensor], locations: List[dict], shiny_envs: List[bool]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Incorporate shiny object information into abstract location."""
        shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(loc["shiny"], dtype=torch.float) for loc in locations if loc.get("shiny") is not None]), dim=-1)

        mu_g_shiny = self.MLP_mu_g_shiny([shiny_locations for _ in range(self.arch.n_f_g if self.arch.separate_ovc else self.arch.n_f)])
        mu_g_shiny = [torch.abs(mu) for mu in mu_g_shiny]
        mu_g_shiny = [utils.leaky_relu(torch.clamp(mu, min=-1, max=1)) for mu in mu_g_shiny]

        sigma_g_shiny = self.MLP_sigma_g_shiny([shiny_locations for _ in range(self.arch.n_f_g if self.arch.separate_ovc else self.arch.n_f)])

        # Update only OVC modules
        module_start = self.arch.n_f_g if self.arch.separate_ovc else 0
        for f in range(module_start, self.arch.n_f):
            mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs, :], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs, :], sigma_g_shiny[f - module_start]])
            mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
            mask[shiny_envs, :] = True
            mu_g[f] = mu_g[f].masked_scatter(mask, mu)
            sigma_g[f] = sigma_g[f].masked_scatter(mask, sigma)

        return mu_g, sigma_g


class GroundedLocationModule(nn.Module):
    """Manages grounded location (place cell) inference and generation."""

    def __init__(self, arch_config: ArchitectureConfig):
        super().__init__()
        self.config = arch_config
        self.MLP_sigma_p = MLP(arch_config.n_p, arch_config.n_p, activation=[torch.tanh, torch.exp])

    def infer(self, sensory: List[torch.Tensor], abstract: List[torch.Tensor]) -> GroundedLocationState:
        """Infer grounded location from sensory and abstract inputs."""
        p = []
        for f in range(self.config.n_f):
            mu_p = abstract[f] * sensory[f]  # Element-wise multiplication
            mu_p = utils.leaky_relu(torch.clamp(mu_p, min=-1, max=1))
            p.append(mu_p)

        sigma_p = self.MLP_sigma_p(p)
        return GroundedLocationState(p=p, sigma=sigma_p)
