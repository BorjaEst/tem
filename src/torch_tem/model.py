#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored TEM Model - Modular Architecture
Following OOP principles with clear separation of concerns.
"""
import copy
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from scipy.stats import truncnorm

from torch_tem import utils
from torch_tem.config import (
    ArchitectureConfig,
    MemoryConfig,
    ModelConfig,
    StaticMatrices,
)
from torch_tem.core.mlp import MLP
from torch_tem.core.states import (
    AbstractLocationState,
    EpisodeState,
    GroundedLocationState,
    IterationResult,
    IterationStates,
    LossComponents,
    ObservationDistribution,
)
from torch_tem.modules.location import AbstractLocationModule, GroundedLocationModule
from torch_tem.modules.memory import MemorySystem
from torch_tem.modules.sensory import SensoryProcessor


class TEMModel(LightningModule):
    """
    Tolman-Eichenbaum Machine - Refactored Orchestrator.

    Coordinates specialized modules with minimal direct computation.
    """

    def __init__(self, arch_config: ArchitectureConfig, mem_config: MemoryConfig, model_config: ModelConfig):
        super().__init__()

        # Save configurations
        self.arch = arch_config
        self.mem_config = mem_config
        self.model_config = model_config

        # Create static matrices
        self.static_matrices = StaticMatrices.create(arch_config, mem_config)

        # Initialize specialized modules
        self.sensory = SensoryProcessor(arch_config, self.static_matrices)
        self.abstract_loc = AbstractLocationModule(arch_config, model_config, self.static_matrices)
        self.grounded_loc = GroundedLocationModule(arch_config)
        self.memory = MemorySystem(arch_config, mem_config, self.static_matrices)
        self.obs_gen = ObservationGenerator(arch_config, self.static_matrices)
        self.state_mgr = StateManager(arch_config)
        self.loss_calc = LossCalculator()

    def forward(self, walk: List[Tuple], prev_state: Optional[EpisodeState] = None):
        """Process a walk through an environment."""
        # Initialize or continue state
        if prev_state is None:
            batch_size = walk[0][1].shape[0]
            state = self.state_mgr.initialize_episode(batch_size, self.abstract_loc.g_init, not self.mem_config.common_memory)
        else:
            state = prev_state

        results = []

        for location, observation, action in walk:
            result = self._single_iteration(observation, location, action, state)
            results.append(result)
            state = result.next_state

        return results

    def _single_iteration(self, observation: torch.Tensor, location: dict, action: List[int], state: EpisodeState) -> IterationResult:
        """Single TEM iteration - coordinates all modules."""

        # 1. Process sensory input
        sensory_state = self.sensory(observation, state.x_prev)

        # 2. Retrieve grounded location from sensory memory
        p_from_sensory = self.memory.retrieve(sensory_state.memory_ready, state.memory_inf or state.memory_gen, "inference") if self.model_config.use_p_inf else None

        # 3. Generate abstract location (path integration)
        abstract_gen = self.abstract_loc.generate_from_transition(state.g_prev, action, [location])

        # 4. Infer abstract location (memory + path integration + shiny)
        abstract_inf = self.abstract_loc.infer(p_from_sensory, abstract_gen, observation, [location])

        # 5. Prepare abstract location for memory
        abstract_memory = self.abstract_loc.prepare_for_memory(abstract_inf.mu)

        # 6. Infer grounded location
        grounded_inf = self.grounded_loc.infer(sensory_state.memory_ready, abstract_memory)

        # 7. Generate from different pathways
        grounded_from_g_inf = self.memory.retrieve(self.abstract_loc.prepare_for_memory(abstract_inf.mu), state.memory_gen, "generative")
        grounded_from_g_gen = self.memory.retrieve(self.abstract_loc.prepare_for_memory(abstract_gen.mu), state.memory_gen, "generative")

        # 8. Generate observations
        obs_from_p = self.obs_gen(grounded_inf.p[0])
        obs_from_g_inf = self.obs_gen(grounded_from_g_inf[0])
        obs_from_g_gen = self.obs_gen(grounded_from_g_gen[0])

        # 9. Update memory
        memory_gen_new = self.memory.update(state.memory_gen, grounded_inf.p, grounded_from_g_inf, hierarchical=True)

        if self.mem_config.common_memory:
            memory_inf_new = memory_gen_new
        else:
            memory_inf_new = self.memory.update(state.memory_inf, grounded_inf.p, p_from_sensory if p_from_sensory else grounded_inf.p, hierarchical=False)

        # 10. Calculate losses
        losses = self.loss_calc.compute(abstract_inf, abstract_gen, grounded_inf, grounded_from_g_inf, p_from_sensory, observation, [obs_from_p, obs_from_g_inf, obs_from_g_gen])

        # 11. Create next state
        next_state = EpisodeState(memory_gen=memory_gen_new, memory_inf=memory_inf_new, g_prev=abstract_inf.mu, x_prev=sensory_state.filtered, step=state.step + 1)

        return IterationResult(
            losses=losses,
            states=IterationStates(sensory=sensory_state, abstract_inf=abstract_inf, abstract_gen=abstract_gen, grounded_inf=grounded_inf, grounded_gen=grounded_from_g_inf),
            observations=[obs_from_p, obs_from_g_inf, obs_from_g_gen],
            next_state=next_state,
        )

    def training_step(self, batch, batch_idx):
        """PyTorch Lightning training step."""
        results = self.forward(batch)
        total_loss = sum(r.losses.total for r in results)
        self.log("train_loss", total_loss)
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ObservationGenerator(nn.Module):
    """Generates sensory observations from grounded locations."""

    def __init__(self, arch_config: ArchitectureConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.config = arch_config
        self.matrices = static_matrices

        self.w_x = nn.Parameter(torch.tensor(1.0))
        self.b_x = nn.Parameter(torch.zeros(arch_config.n_x_c))
        self.MLP_c_star = MLP(arch_config.n_x_f[0], arch_config.n_x, hidden_dim=20 * arch_config.n_x_c)

    def forward(self, p: torch.Tensor) -> ObservationDistribution:
        """Generate observation distribution from place cells."""
        # Sum over entorhinal preferences
        x_compressed = self.w_x * torch.matmul(p, torch.t(self.matrices.W_tile[0])) + self.b_x

        # Decompress to full observation space
        logits = self.MLP_c_star(x_compressed)
        probabilities = utils.softmax(logits)

        return ObservationDistribution(logits=logits, probabilities=probabilities)


class StateManager:
    """Manages model state across iterations and episodes."""

    def __init__(self, arch_config: ArchitectureConfig):
        self.config = arch_config

    def initialize_episode(self, batch_size: int, g_init: List[torch.Tensor], use_separate_memory: bool) -> EpisodeState:
        """Initialize state for new episode."""
        n_p_total = sum(self.config.n_p)
        M_gen = torch.zeros((batch_size, n_p_total, n_p_total))
        M_inf = None if not use_separate_memory else torch.zeros((batch_size, n_p_total, n_p_total))

        g_prev = [torch.stack([g_init[f] for _ in range(batch_size)]) for f in range(self.config.n_f)]
        x_prev = [torch.zeros((batch_size, self.config.n_x_f[f])) for f in range(self.config.n_f)]

        return EpisodeState(memory_gen=M_gen, memory_inf=M_inf, g_prev=g_prev, x_prev=x_prev, step=0)

    def reset_walks(self, state: EpisodeState, new_walk_mask: List[bool], g_init: List[torch.Tensor]) -> EpisodeState:
        """Reset state for walks starting new environments."""
        for idx, is_new in enumerate(new_walk_mask):
            if is_new:
                state.memory_gen[idx] = 0
                if state.memory_inf is not None:
                    state.memory_inf[idx] = 0
                for f in range(self.config.n_f):
                    state.g_prev[f][idx] = g_init[f]
                    state.x_prev[f][idx] = 0
        return state


class LossCalculator:
    """Calculates all loss components."""

    @staticmethod
    def compute(
        abstract_inf: AbstractLocationState,
        abstract_gen: AbstractLocationState,
        grounded_inf: GroundedLocationState,
        grounded_gen: List[torch.Tensor],
        p_inf_x: Optional[List[torch.Tensor]],
        obs_target: torch.Tensor,
        obs_predictions: List[ObservationDistribution],
    ) -> LossComponents:
        """Compute all loss components."""
        # Grounded location losses
        L_p_g = torch.sum(torch.stack(utils.squared_error(grounded_inf.p, grounded_gen), dim=0), dim=0)
        L_p_x = torch.sum(torch.stack(utils.squared_error(grounded_inf.p, p_inf_x), dim=0), dim=0) if p_inf_x is not None else torch.zeros_like(L_p_g)

        # Abstract location loss
        L_g = torch.sum(torch.stack(utils.squared_error(abstract_inf.mu, abstract_gen.mu), dim=0), dim=0)

        # Observation prediction losses
        labels = torch.argmax(obs_target, 1)
        L_x_p = utils.cross_entropy(obs_predictions[0].logits, labels)
        L_x_g = utils.cross_entropy(obs_predictions[1].logits, labels)
        L_x_gen = utils.cross_entropy(obs_predictions[2].logits, labels)

        # Regularization losses
        L_reg_g = torch.sum(torch.stack([torch.sum(g**2, dim=1) for g in abstract_inf.mu], dim=0), dim=0)
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in grounded_inf.p], dim=0), dim=0)

        return LossComponents(L_p_g=L_p_g, L_p_x=L_p_x, L_x_gen=L_x_gen, L_x_g=L_x_g, L_x_p=L_x_p, L_g=L_g, L_reg_g=L_reg_g, L_reg_p=L_reg_p)
