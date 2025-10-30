#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refactored TEM Model - Modular Architecture
Following OOP principles with clear separation of concerns.
"""
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor

from torch_tem import utils
from torch_tem.config import (
    ArchitectureConfig,
    MemoryConfig,
    ModelConfig,
    StaticMatrices,
)
from torch_tem.core.mlp import MLP
from torch_tem.modules.location import (
    AbstractLocationModule,
    AbstractLocationState,
    GroundedLocationModule,
    GroundedLocationState,
)
from torch_tem.modules.memory import MemorySystem
from torch_tem.modules.sensory import SensoryProcessor, SensoryState

# ==============================================================================
# Orchestrator-Level State Models
# ==============================================================================
# These states coordinate information flow between modules and are owned
# by the TEMModel orchestrator rather than individual modules.


class ObservationDistribution(BaseModel):
    """Container for generated observation distribution.

    Represents a probabilistic prediction of sensory observations,
    typically decoded from place cell activations.

    Attributes:
        logits: Raw network outputs before softmax [batch, n_x]
        probabilities: Normalized probability distribution P(x) [batch, n_x]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    logits: Tensor
    probabilities: Tensor

    def sample(self) -> Tensor:
        """Sample from categorical distribution.

        Returns:
            Sampled observation indices [batch, 1]
        """
        return torch.multinomial(self.probabilities, 1)

    def mode(self) -> Tensor:
        """Return most likely observation.

        Returns:
            Most probable observation index [batch]
        """
        return torch.argmax(self.probabilities, dim=-1)


class EpisodeState(BaseModel):
    """Complete state for an episode.

    Maintains all persistent state across multiple iterations within
    an episode, including memory matrices and previous activations.

    Attributes:
        memory_gen: Generative memory matrix [batch, n_p_total, n_p_total]
        memory_inf: Inference memory matrix if separate [batch, n_p_total, n_p_total]
        g_prev: Previous grid cell activations [n_freq x [batch, n_g[f]]]
        x_prev: Previous sensory observations [n_freq x [batch, n_x_c]]
        step: Current step number within episode
        metadata: Optional additional episode-level information
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    memory_gen: Tensor
    memory_inf: Optional[Tensor]
    g_prev: List[Tensor]
    x_prev: List[Tensor]
    step: int
    metadata: Dict = Field(default_factory=dict)


class IterationStates(BaseModel):
    """All internal states from one iteration.

    Collects the internal representations from all modules during
    a single forward pass, useful for analysis and debugging.

    Attributes:
        sensory: Sensory processing outputs
        abstract_inf: Inferred abstract location (with memory correction)
        abstract_gen: Generated abstract location (path integration only)
        grounded_inf: Inferred grounded location (grid x sensory)
        grounded_gen: Generated grounded location from memory retrieval
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sensory: SensoryState
    abstract_inf: AbstractLocationState
    abstract_gen: AbstractLocationState
    grounded_inf: GroundedLocationState
    grounded_gen: List[Tensor]


class LossComponents(BaseModel):
    """Individual loss components for training.

    Breaks down the total training objective into interpretable components,
    each encouraging a specific aspect of correct behavior.

    Loss Definitions:
    - L_p_g: Consistency between inferred and generated place cells
    - L_p_x: Consistency between inferred place cells and sensory retrieval
    - L_x_gen: Observation reconstruction from generated grid cells
    - L_x_g: Observation reconstruction from inferred grid cells
    - L_x_p: Observation reconstruction from inferred place cells
    - L_g: Consistency between inferred and generated grid cells
    - L_reg_g: L2 regularization on grid cell activations
    - L_reg_p: L1 regularization on place cell activations

    Attributes:
        L_p_g: Grounded location consistency loss [batch]
        L_p_x: Sensory-grounded consistency loss [batch]
        L_x_gen: Generative observation loss [batch]
        L_x_g: Grid-based observation loss [batch]
        L_x_p: Place-based observation loss [batch]
        L_g: Abstract location consistency loss [batch]
        L_reg_g: Grid cell regularization loss [batch]
        L_reg_p: Place cell regularization loss [batch]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    L_p_g: Tensor
    L_p_x: Tensor
    L_x_gen: Tensor
    L_x_g: Tensor
    L_x_p: Tensor
    L_g: Tensor
    L_reg_g: Tensor
    L_reg_p: Tensor

    def as_list(self) -> List[Tensor]:
        """Return losses as list in standard order.

        Returns:
            List of loss tensors in canonical order
        """
        return [
            self.L_p_g,
            self.L_p_x,
            self.L_x_gen,
            self.L_x_g,
            self.L_x_p,
            self.L_g,
            self.L_reg_g,
            self.L_reg_p,
        ]

    @property
    def total(self) -> Tensor:
        """Sum of all loss components.

        Returns:
            Total loss [batch]
        """
        return sum(self.as_list())


class IterationResult(BaseModel):
    """Complete result from one iteration.

    Packages all outputs from a single TEM iteration, including
    losses for training, internal states for analysis, observation
    predictions, and updated state for the next iteration.

    Attributes:
        losses: All loss components for training
        states: Internal module states for analysis
        observations: Observation distributions from different pathways
        next_state: Updated episode state for next iteration
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    losses: LossComponents
    states: IterationStates
    observations: List[ObservationDistribution]
    next_state: EpisodeState


class TEMModel(LightningModule):
    """Tolman-Eichenbaum Machine - Refactored Orchestrator.

    The TEM is a biologically-inspired model of spatial cognition and episodic memory.
    It combines path integration (grid cells), sensory grounding (place cells), and
    Hebbian associative memory to enable flexible navigation and memory-based inference.

    Architecture Overview:
    ======================
    This orchestrator coordinates five specialized modules:

    1. **SensoryProcessor**: Prepares observations for memory operations
       - Temporal filtering, normalization, compression

    2. **AbstractLocationModule**: Grid cell system with path integration
       - Tracks position via self-motion integration
       - Multi-scale spatial representation

    3. **GroundedLocationModule**: Place cell conjunction
       - Combines grid cells with sensory input
       - Creates location-specific activations

    4. **MemorySystem**: Hebbian associative memory
       - Links place cells with observations
       - Supports attractor-based retrieval

    5. **ObservationGenerator**: Decodes observations from place cells
       - Probabilistic observation prediction

    Information Flow (Single Iteration):
    ====================================
    1. Process sensory input → sensory_state
    2. Retrieve place cells from memory using sensory_state → p_from_sensory
    3. Generate grid cells via path integration → abstract_gen
    4. Infer grid cells (combine path + memory + landmarks) → abstract_inf
    5. Infer place cells (grid x sensory) → grounded_inf
    6. Generate observations from place cells → obs_predictions
    7. Update associative memory with new (place, observation) pairs
    8. Calculate losses for training

    Module Independence:
    ====================
    This refactored version uses dependency injection: modules receive only
    primitive parameters, not parent config objects. This enables:
    - Independent testing of each module
    - Clear interfaces between components
    - Easier debugging and maintenance

    Args:
        arch_config: Architecture configuration with all network dimensions,
            frequencies, connectivity patterns
        mem_config: Memory system configuration (learning rates, decay, iterations)
        model_config: Behavioral configuration (sampling, inference modes)

    Attributes:
        sensory: SensoryProcessor instance for observation processing
        abstract_loc: AbstractLocationModule for grid cell operations
        grounded_loc: GroundedLocationModule for place cell inference
        memory: MemorySystem for Hebbian associative memory
        obs_gen: ObservationGenerator for decoding observations
        state_mgr: StateManager for episode state management
        loss_calc: LossCalculator for computing training objectives
    """

    def __init__(self, arch_config: ArchitectureConfig, mem_config: MemoryConfig, model_config: ModelConfig):
        super().__init__()

        # Save configurations
        self.arch = arch_config
        self.mem_config = mem_config
        self.model_config = model_config

        # Create static matrices
        self.static_matrices = StaticMatrices.create(arch_config, mem_config)

        # Initialize specialized modules with minimal dependencies
        self.sensory = SensoryProcessor(
            n_frequencies=arch_config.n_f,
            initial_frequencies=arch_config.f_initial_extended,
            two_hot_table=self.static_matrices.two_hot_table,
            tile_matrices=self.static_matrices.W_tile,
        )

        self.abstract_loc = AbstractLocationModule(arch_config, model_config, self.static_matrices)

        self.grounded_loc = GroundedLocationModule(n_p_dims=arch_config.n_p)

        self.memory = MemorySystem(
            n_p_dims=arch_config.n_p,
            lambda_forget=mem_config.lambda_,
            eta_remember=mem_config.eta,
            kappa_decay=mem_config.kappa,
            n_attractor_iters=mem_config.i_attractor,
            p_update_mask=self.static_matrices.p_update_mask,
            p_retrieve_mask_inf=self.static_matrices.p_retrieve_mask_inf,
            p_retrieve_mask_gen=self.static_matrices.p_retrieve_mask_gen,
        )

        self.obs_gen = ObservationGenerator(arch_config, self.static_matrices)
        self.state_mgr = StateManager(arch_config)
        self.loss_calc = LossCalculator()

    def forward(self, walk: List[Tuple], prev_state: Optional[EpisodeState] = None):
        """Process a walk through an environment.

        A "walk" is a sequence of (location, observation, action) tuples representing
        an agent's trajectory through an environment. This method processes each step
        sequentially, maintaining and updating internal state (memory, grid cells).

        Args:
            walk: List of (location, observation, action) tuples where:
                - location: Dict with position metadata (may include 'shiny' key)
                - observation: Tensor [batch, n_x_dims] of sensory input
                - action: List of action indices (movement direction)
            prev_state: Optional EpisodeState from previous walk to continue episode.
                If None, initializes fresh state.

        Returns:
            List of IterationResult objects, one per walk step, containing:
            - losses: All loss components for training
            - states: Internal states (sensory, abstract, grounded)
            - observations: Generated observation distributions
            - next_state: Updated EpisodeState for next iteration
        """
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
        """Single TEM iteration - coordinates all modules.

        This is the core computational step of the TEM model, implementing the
        full sensory-memory-location processing pipeline:

        1. **Sensory Processing**: Compress and normalize observation
        2. **Memory Retrieval**: Recall place cells from sensory input
        3. **Path Integration**: Update grid cells based on movement
        4. **Location Inference**: Combine path integration with memory and landmarks
        5. **Place Cell Inference**: Ground abstract location with sensory input
        6. **Generative Decoding**: Predict observations from place cells
        7. **Memory Update**: Store new associations (place ↔ observation)
        8. **Loss Calculation**: Compute training objectives

        Args:
            observation: Current sensory observation [batch, n_x_dims]
            location: Location metadata dict (may contain 'shiny' key for landmarks)
            action: List of action indices (movement directions)
            state: Current EpisodeState with memory matrices and previous activations

        Returns:
            IterationResult containing losses, internal states, observation predictions,
            and updated state for next iteration
        """

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
    """Generates sensory observations from grounded locations.

    This module implements the decoding pathway from place cell activations (p)
    back to sensory observations (x). It enables the model to make predictions
    about what observations should be encountered at a given location.

    Decoding Process:
    =================
    1. **Compression**: Sum place cell activations over entorhinal preferences
       x_compressed = w_x · p · W_tile^T + b_x

    2. **Decompression**: Expand compressed representation to full observation space
       logits = MLP(x_compressed)

    3. **Probabilistic Output**: Apply softmax to get observation distribution
       P(x | p) = softmax(logits)

    This allows the model to:
    - Predict expected observations at inferred locations
    - Generate "mental imagery" from memory
    - Calculate prediction errors for learning

    Args:
        arch_config: Architecture configuration containing:
            - n_x: Full observation space dimensionality
            - n_x_c: Compressed observation dimensionality
            - n_x_f: Observation dimensions per frequency
        static_matrices: Pre-computed matrices:
            - W_tile: Tiling matrices for compression

    Attributes:
        w_x: Learnable weight for compression scaling
        b_x: Learnable bias for compressed representation
        MLP_c_star: Decompression network (compressed → full observation space)
    """

    def __init__(self, arch_config: ArchitectureConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.config = arch_config
        self.matrices = static_matrices

        self.w_x = nn.Parameter(torch.tensor(1.0))
        self.b_x = nn.Parameter(torch.zeros(arch_config.n_x_c))
        self.MLP_c_star = MLP(arch_config.n_x_f[0], arch_config.n_x, hidden_dim=20 * arch_config.n_x_c)

    def forward(self, p: torch.Tensor) -> ObservationDistribution:
        """Generate observation distribution from place cells.

        Decodes place cell activations into a probability distribution over
        possible observations. This implements the generative model P(x | p)
        that predicts what the agent should observe at the location encoded by p.

        Args:
            p: Place cell activations [batch, n_p_dims] representing current location

        Returns:
            ObservationDistribution containing:
            - logits: Raw network outputs [batch, n_x] before softmax
            - probabilities: Normalized probability distribution P(x | p)
        """
        # Sum over entorhinal preferences
        x_compressed = self.w_x * torch.matmul(p, torch.t(self.matrices.W_tile[0])) + self.b_x

        # Decompress to full observation space
        logits = self.MLP_c_star(x_compressed)
        probabilities = utils.softmax(logits)

        return ObservationDistribution(logits=logits, probabilities=probabilities)


class StateManager:
    """Manages model state across iterations and episodes.

    The TEM model is stateful: it maintains memory matrices, previous activations,
    and step counters that persist across multiple forward passes within an episode.
    This class encapsulates all state management logic.

    State Components:
    =================
    - **memory_gen**: Generative memory matrix M [batch, n_p, n_p] linking place cells
      to observations for prediction
    - **memory_inf**: Inference memory matrix (if separate) for sensory→place retrieval
    - **g_prev**: Previous grid cell activations for path integration
    - **x_prev**: Previous observations for temporal filtering
    - **step**: Current step counter within episode

    Two Operating Modes:
    ====================
    1. **Common Memory**: Single memory matrix serves both inference and generation
       - Simpler, fewer parameters
       - Used when inference and generation share same associations

    2. **Separate Memory**: Distinct matrices for inference vs. generation
       - More flexible, can specialize each pathway
       - Used when asymmetric memory access is desired

    Args:
        arch_config: Architecture configuration with dimensions and frequencies
    """

    def __init__(self, arch_config: ArchitectureConfig):
        self.config = arch_config

    def initialize_episode(self, batch_size: int, g_init: List[torch.Tensor], use_separate_memory: bool) -> EpisodeState:
        """Initialize state for new episode.

        Creates fresh state for starting a new episode or batch of episodes.
        All memory matrices are zeroed, grid cells are set to learned priors,
        and previous observations are cleared.

        Args:
            batch_size: Number of parallel episodes to initialize
            g_init: Learned prior grid cell activations [n_g[f]] per frequency,
                will be replicated across batch
            use_separate_memory: Whether to create distinct inference and generation
                memory matrices (True) or share a single matrix (False)

        Returns:
            EpisodeState with zeroed memory, prior grid cells, zero previous
            observations, and step counter at 0
        """
        n_p_total = sum(self.config.n_p)
        M_gen = torch.zeros((batch_size, n_p_total, n_p_total))
        M_inf = None if not use_separate_memory else torch.zeros((batch_size, n_p_total, n_p_total))

        g_prev = [torch.stack([g_init[f] for _ in range(batch_size)]) for f in range(self.config.n_f)]
        x_prev = [torch.zeros((batch_size, self.config.n_x_f[f])) for f in range(self.config.n_f)]

        return EpisodeState(memory_gen=M_gen, memory_inf=M_inf, g_prev=g_prev, x_prev=x_prev, step=0)

    def reset_walks(self, state: EpisodeState, new_walk_mask: List[bool], g_init: List[torch.Tensor]) -> EpisodeState:
        """Reset state for walks starting new environments.

        In batch processing, some walks may complete while others continue.
        This method selectively resets state for specific batch elements that
        are starting fresh environments, without affecting ongoing walks.

        Args:
            state: Current EpisodeState to be partially reset
            new_walk_mask: Boolean list indicating which batch elements start
                new environments (True = reset, False = continue)
            g_init: Learned prior grid cell activations to reset to

        Returns:
            Modified EpisodeState with selected elements reset (in-place modification)
        """
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
    """Calculates all loss components.

    The TEM model is trained using multiple loss terms that encourage different
    aspects of correct behavior:

    Loss Components:
    ================
    1. **L_p_g**: Consistency between place cells inferred from sensory input vs.
       generated from grid cells via memory
       - Ensures sensory grounding matches abstract representation

    2. **L_p_x**: Consistency between place cells from sensory input vs. retrieved
       from memory using that sensory input
       - Verifies memory correctly associates sensory observations with locations

    3. **L_g**: Consistency between inferred grid cells (with memory correction) vs.
       generated grid cells (path integration only)
       - Encourages memory to provide useful corrections to path integration

    4. **L_x_p**: Observation reconstruction from inferred place cells
       - Tests if place cells encode sufficient information to predict observations

    5. **L_x_g**: Observation reconstruction from inferred grid cells (via memory)
       - Tests full inference pathway: grid → memory → place → observation

    6. **L_x_gen**: Observation reconstruction from generated grid cells (via memory)
       - Tests generation pathway: path integration → memory → place → observation

    7. **L_reg_g**: L2 regularization on grid cell activations
       - Prevents unbounded growth, encourages sparse representations

    8. **L_reg_p**: L1 regularization on place cell activations
       - Encourages sparsity in place cell code (few active cells per location)

    All losses are computed per batch element and can be weighted/combined
    during training according to learning curriculum.
    """

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
        """Compute all loss components.

        Evaluates the model's performance across multiple objectives simultaneously.
        Each loss term measures a different aspect of spatial cognition and memory.

        Args:
            abstract_inf: Inferred grid cells (path + memory + landmarks)
            abstract_gen: Generated grid cells (path integration only)
            grounded_inf: Inferred place cells (grid x sensory)
            grounded_gen: Generated place cells from inferred grid via memory
            p_inf_x: Place cells retrieved from memory using sensory input
                (None if memory-based inference disabled)
            obs_target: Ground truth observation [batch, n_x] (one-hot or soft labels)
            obs_predictions: List of 3 ObservationDistributions:
                [0] from inferred place cells (p_inf)
                [1] from inferred grid via memory (g_inf → M → p → x)
                [2] from generated grid via memory (g_gen → M → p → x)

        Returns:
            LossComponents containing all individual loss terms:
            - L_p_g: Place consistency (inference vs. generation)
            - L_p_x: Place-memory consistency
            - L_g: Grid consistency (inference vs. generation)
            - L_x_p, L_x_g, L_x_gen: Observation reconstruction losses
            - L_reg_g, L_reg_p: Regularization terms
        """
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
