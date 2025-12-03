"""PyTorch implementation of the Tolman–Eichenbaum Machine (TEM).

This module defines the high-level `TEMModel` for a modern, modular
TEM implementation. It mirrors the functionality of the original
`tem.model.Model` class, but is structured around typed configuration objects
and sub-modules for sensory encoding, inference, memory, and projection.

Implementation follows the reference model.py while using modular torch_tem
components for maintainability and testability.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.config import EnvironmentConfig, ModelConfig
from torch_tem.core.projection import ProjectionHead
from torch_tem.generation import GenerativeModel, GenerativeState
from torch_tem.inference import InferenceModel, InferenceState
from torch_tem.losses import Losses
from torch_tem.memory import MemoryState
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage
from torch_tem.types import AbstractLocation, GroundedLocation, LocationInference, MultiScaleCode, SensoryObservation, SensoryPrediction


@dataclass
class TEMState:
    """Output data from a single TEM iteration.

    Attributes:
        inference_state: Auxiliary outputs from inference pathway (required)
        generative_state: Auxiliary outputs from generative pathway (required)

    Properties (computed from nested states):
        memory: Combined memory state [M_gen, M_inf] for compatibility
        belief: Current belief state (abstract location g_inf)
        prediction: Predicted sensory observation (x_gen values)
        latent_prediction: Inferred latent locations (g_inf, p_inf)
        sensory_prediction: Generated sensory prediction (x_gen)
        g_inf: Inferred abstract location
        p_inf: Inferred grounded location
        g_gen: Generated abstract location
        x_prev: Previous filtered observation

    Theory:
        The model produces beliefs about current location, predictions
        about sensory input, and updated memory associations. Both
        generative and inference pathways contribute to the final state.

        Data is stored once in nested states (inference_state, generative_state)
        and accessed via properties to avoid duplication. This follows the
        "deep class" principle where complexity is hidden but API remains simple.

        Memory is split between pathways: M_gen in generative_state,
        M_inf in inference_state. The memory property reconstructs the
        combined MemoryState for backward compatibility.

        Note: action is not part of state - it's an input argument to iteration.
    """

    # Pathway states (contain all information including their respective memories)
    inference_state: InferenceState
    generative_state: GenerativeState  # Memory property (reconstructed from pathway states)

    @property
    def memory(self) -> MemoryState:
        """Combined memory state [M_gen, M_inf]."""
        return [self.generative_state.memory_gen, self.inference_state.memory_inf]

    # Convenience properties (no duplication)
    @property
    def belief(self) -> AbstractLocation:
        """Current belief (g_inf from inference)."""
        return self.inference_state.latent_prediction.abstract

    @property
    def prediction(self) -> MultiScaleCode:
        """Sensory prediction (x_gen values from generation)."""
        return self.generative_state.sensory_prediction.values

    @property
    def latent_prediction(self) -> LocationInference:
        """Inferred locations (g_inf, p_inf)."""
        return self.inference_state.latent_prediction

    @property
    def sensory_prediction(self) -> SensoryPrediction:
        """Generated sensory prediction."""
        return self.generative_state.sensory_prediction

    @property
    def g_inf(self) -> AbstractLocation:
        """Inferred abstract location."""
        return self.latent_prediction.abstract

    @property
    def p_inf(self) -> GroundedLocation:
        """Inferred grounded location."""
        return self.latent_prediction.grounded

    @property
    def g_gen(self) -> AbstractLocation:
        """Generated abstract location."""
        return self.generative_state.g_gen

    @property
    def x_prev(self) -> MultiScaleCode:
        """Previous filtered observation."""
        return self.inference_state.filtered_observation


class TEMModel(InferenceModel, GenerativeModel, nn.Module):
    """Top-level Tolman–Eichenbaum Machine model.

    This class is the main entry point for running TEM in PyTorch. It combines
    sensory encoding, abstract and grounded location inference, generative
    decoding, and Hebbian memory dynamics. Conceptually, it corresponds to the
    original ``Model`` class in ``model.py``, but is designed to work with
    structured configuration objects and reusable sub-modules under
    ``torch_tem.*``.

    The methods defined here are intentionally left unimplemented; they
    document the public API and expected behavior so that the implementation
    can be ported from the reference code in a controlled, testable way.
    """

    def __init__(self, params: ModelConfig):
        """Initialise a new TEM model from configuration parameters.

        Parameters
        ----------
        params:
            High-level model configuration specifying paramsitectural choices
            (numbers of cells, frequency modules, connectivity patterns),
            environment and inference options, and training hyper-parameters.

        Notes
        -----
        The original implementation accepts a plain parameter dictionary
        (see ``model.Model.__init__``). This method uses the typed
        ``ModelConfig`` dataclass while constructing and registering all
        trainable sub-modules (encoders, projections, memory, etc.).
        """
        # Initialize nn.Module first to avoid diamond inheritance issues
        nn.Module.__init__(self)

        # Store configuration for access throughout the model
        self.config = params

        # Compute configuration-derived matrices
        g_downsample = utils.create_g_downsample(params.n_g, params.n_g_subsampled_combined)
        p_update_mask = utils.create_p_update_mask(params.n_p, params.n_f, params.n_f_g, params.n_f_ovc, params.f_extended)
        mask_inf = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_inf)
        mask_gen = utils.create_p_retrieve_mask(params.n_p, params.i_attractor, params.max_freq_gen)

        # Store configuration and generate sub-module parameters
        self.projection = ProjectionHead(params, g_downsample)
        self.attractor = AttractorDynamics(params, mask_inf, mask_gen)
        self.storage = MemoryStorage(params, p_update_mask)
        GenerativeModel.__init__(self, params, self.projection, self.attractor)
        InferenceModel.__init__(self, params, self.projection, self.attractor)

    def init_state(self, x: SensoryObservation, memory: MemoryState) -> TEMState:
        """Initialize TEM state for a new sequence.

        This method sets up the initial TEMState for a new walk sequence,
        initializing belief, memory, and auxiliary outputs to default values.

        Parameters
        ----------
        x:
            Initial sensory observations (one-hot or encoded) for the batch.
        memory:
            Previous memory state to continue from a prior simulation.
        Returns
        -------
        TEMState
            Complete initial TEM state including belief, memory, and
            auxiliary outputs from both inference and generative pathways.
        Theory:
            The initial state sets belief to zero (unknown location),
            memory to the provided state, and all auxiliary outputs to
            default values (zeros or None as appropriate).
        """
        # Initialize inference and generative states
        inference_state = InferenceModel.init_state(self, x.shape[0], x.device)
        generative_state = GenerativeModel.init_state(self, x.shape[0], x.device)

        # Set initial memory states
        inference_state.memory_inf = memory[1]
        generative_state.memory_gen = memory[0]

        return TEMState(inference_state=inference_state, generative_state=generative_state)

    def iteration(self, x: SensoryObservation, locations: List[Dict], action: int, state: TEMState) -> Tuple[Losses, TEMState]:
        """Perform a single TEM iteration for one time step.

        This method combines transition dynamics, inference, generative
        prediction, Hebbian memory update, and loss computation for a single
        step in a batched random walk.

        Parameters
        ----------
        x:
            Current sensory observations (typically one-hot or compressed
            encodings) for each walk in the batch.
        locations:
            Per-walk environment descriptors (e.g. metadata including shiny
            objects) used by the transition and inference models.
        action:
            Action taken at the previous timestep (used for transition dynamics).
        state:
            Previous TEM state containing belief, memory, and auxiliary
            outputs from the previous timestep.

        Returns
        -------
        Losses:
            Structured loss object containing all loss components for this step.

        TEMState:
            Complete output for this iteration including belief, prediction,
            losses, memory, and auxiliary generated/inferred states.

        Theory:
            Each iteration implements the full TEM cycle:
            1. Transition: Path integration (g_prev, a → g_gen)
            2. Inference: Bottom-up sensory processing (x → g_inf, p_inf)
            3. Generation: Top-down prediction (g → p → x)
            4. Memory: Hebbian update (p_inf ⊗ p_gen → M)
            5. Loss: Consistency and reconstruction objectives
        """

        # 1. Transition dynamics: g_prev + a_prev → g_gen
        g_gen, g_gen_params = self.gen_g(action, state.belief, locations)

        # 2. Inference path: x → (g_inf, p_inf)
        # Pass state directly - inference extracts needed data
        inf_state = self.inference(x, locations, state.inference_state, g_gen=g_gen_params)

        # 3. Generative path: (g_inf, p_inf, g_gen) → x_gen
        # Pass latent predictions, g_gen, and previous generative state
        gen_state = self.generative(inf_state.latent_prediction, g_gen, state.generative_state)

        # 4. Memory update: Hebbian plasticity
        p_inf_flat = utils.concatenate_frequencies(inf_state.latent_prediction.grounded)
        p_gen_flat = utils.concatenate_frequencies(gen_state.p_gen)

        # Update generative memory using Hebbian plasticity
        # eta: remembering rate, kappa: forgetting rate (1 - lambda)
        lamb = 1.0 - self.config.kappa  # Convert kappa (forgetting) to lambda (retention)
        self.storage.update(p_inf_flat, p_gen_flat, self.config.eta, lamb)

        # Get updated memories from storage
        inf_state.memory_inf = self.storage.M_gen
        gen_state.memory_gen = self.storage.M_inf if self.storage.use_dual_memory else self.storage.M_gen

        # 5. Loss computation
        state = TEMState(inference_state=inf_state, generative_state=gen_state)
        losses = self.loss(x, g_gen, state)

        return losses, state

    def loss(self, x: SensoryObservation, g_gen: AbstractLocation, state: TEMState) -> Losses:
        """Compute all TEM loss components for a single time step.

        The loss combines consistency terms between inferred and generated
        grounded locations, path-integrated versus inferred abstract
        locations, reconstruction losses over observations, and regularisation
        penalties on abstract and grounded codes.

        Parameters
        ----------
        x:
            Ground-truth sensory observations (one-hot or encoded).
        g_gen:
            Generated abstract location from transition model.
        state:
            Complete TEM state containing both inference and generative
            pathway outputs with all predictions and intermediate representations.

        Returns
        -------
        Losses
            Structured loss object containing all loss components:
            ``L_p_g``, ``L_p_x``, ``L_x_gen``, ``L_x_g``, ``L_x_p``,
            ``L_g``, ``L_reg_g``, ``L_reg_p``.

        Theory:
            Loss balances multiple objectives:
            - Consistency: ||p_inf - p_gen||², ||g_inf - g_gen||²
            - Reconstruction: CE(x, x_gen), CE(x, x_g), CE(x, x_p)
            - Regularization: ||g||² (L2), ||p||₁ (L1 sparsity)
        """

        # Unpack structured types from state
        g_inf = state.inference_state.latent_prediction.abstract
        p_inf = state.inference_state.latent_prediction.grounded
        p_gen = state.generative_state.p_gen
        p_x = state.inference_state.retrieved_grounded

        # Prepare observation labels for cross-entropy losses
        labels = torch.argmax(x, 1)

        # Get batch shape for creating zero tensors
        use_p_inf = self.config.use_p_inf and p_x is not None
        batch_shape = p_inf[0].shape[0] if isinstance(p_inf, list) else p_inf.shape[0]

        return Losses(
            # L_p_g: ||p_inf - p_gen||²
            L_p_g=torch.sum(torch.stack(utils.squared_error_freq(p_inf, p_gen), dim=0), dim=0),
            # L_p_x: ||p_inf - p_x||² (if using inference memory)
            L_p_x=torch.sum(torch.stack(utils.squared_error_freq(p_inf, p_x), dim=0), dim=0) if use_p_inf else torch.zeros(batch_shape),
            # L_g: ||g_inf - g_gen||²
            L_g=torch.sum(torch.stack(utils.squared_error_freq(g_inf, g_gen), dim=0), dim=0),
            # Cross-entropy losses for observation reconstruction
            L_x_gen=torch.nn.functional.cross_entropy(state.generative_state.x_gen.logits, labels, reduction="none"),
            L_x_g=torch.nn.functional.cross_entropy(state.generative_state.x_g.logits, labels, reduction="none"),
            L_x_p=torch.nn.functional.cross_entropy(state.generative_state.x_p.logits, labels, reduction="none"),
            # Regularization: L2 on abstract location, L1 on grounded location
            L_reg_g=torch.sum(torch.stack([torch.sum(g**2, dim=1) for g in g_inf], dim=0), dim=0),
            L_reg_p=torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0),
        )


class Simulation(Iterator[TEMState]):
    """Iterator for running TEM simulation over a walk sequence.

    This class wraps the TEMModel to provide a stateful iteration over a walk,
    maintaining the TEM state between steps and yielding TEMState objects.

    The class properly implements the Iterator protocol by inheriting from
    collections.abc.Iterator, making it a proper iterator that can be used
    in for loops and with next().

    Note:
        As per Iterator protocol, this class is single-use. Once exhausted,
        create a new Simulation instance to iterate over the walk again.

    Example:
        >>> simulation = Simulation(model, walk)
        >>> for state in simulation:
        ...     # Process each TEM state
        ...     losses = compute_losses(state)
        ...     losses.backward()
    """

    def __init__(self, model: TEMModel, walk, memory: MemoryState):
        """Initialize simulation with model and walk sequence.

        Parameters
        ----------
        model : TEMModel
            The TEM model to use for inference and generation.
        walk : Iterable
            Walk sequence of (locations, observations, actions) tuples.
            Must contain at least one step.
        memory : MemoryState
            Previous memory state to continue from a previous
            simulation. Must match model configuration.

        Raises
        ------
        ValueError
            If walk sequence is empty or memory configuration mismatch.
        """
        self.__model = model
        self.__walk_iter = iter(walk)

        # Validate memory configuration
        if self.__model.config.common_memory and memory[0] is not memory[1]:
            raise ValueError("Model configured with common_memory=True but memory has distinct memory tensors")

        # Get first step to initialize state
        try:
            first_locations, first_x, first_a = walk[0]
        except IndexError:
            raise ValueError("Walk sequence must contain at least one step")

        # Initialize TEM state with first step data
        self.__state = self.__model.init_state(first_x, memory)

    def __next__(self) -> TEMState:
        """Execute one TEM iteration and return resulting state.

        This method implements the Iterator protocol's __next__ method,
        enabling the Simulation to be used in for loops and with next().

        Returns
        -------
        TEMState
            Complete TEM state after processing current step, containing:
            - belief: Current abstract location (g_inf)
            - prediction: Sensory prediction (x_gen)
            - memory: Updated Hebbian memory matrices
            - All auxiliary outputs from inference and generative pathways

        Raises
        ------
        StopIteration
            When walk sequence is exhausted, as per Iterator protocol.
        """
        # Get next step from walk
        locations, x, a = next(self.__walk_iter)

        # Run one TEM iteration
        _, self.__state = self.__model.iteration(x, locations, a, self.__state)

        return self.__state
