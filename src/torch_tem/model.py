"""PyTorch implementation of the Tolman–Eichenbaum Machine (TEM).

This module defines the high-level `TEMModel` for a modern, modular
TEM implementation. It mirrors the functionality of the original
`tem.model.Model` class, but is structured around typed configuration objects
and sub-modules for sensory encoding, inference, memory, and projection.

Implementation follows the reference model.py while using modular torch_tem
components for maintainability and testability.
"""

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
from torch_tem.types import AbstractLocation, GroundedLocation
from torch_tem.types import LatentPrediction
from torch_tem.types import LatentPrediction as Location
from torch_tem.types import MultiScaleCode
from torch_tem.types import SensoryPrediction
from torch_tem.types import SensoryPrediction as Observation


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
    def latent_prediction(self) -> LatentPrediction:
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

    def forward(self, walk, prev_M=None):
        """Process a full walk sequence through the TEM model.

        Executes the full TEM pipeline for each timestep in the walk,
        performing transition, inference, generation, memory update, and
        loss computation.

        Parameters
        ----------
        walk : Iterable
            Iterable of (locations, observations, actions) for each step.
            Each entry should be a 3-tuple:

            - ``locations``: Batch of location data structures.
            - ``observations``: Observations of shape ``(batch, ...)``, to be
              encoded to ``x``.
            - ``actions``: Batch of actions taken at this step.

        prev_M : List[Tensor], optional
            Optional previous memory state from a previous call, used to
            continue inference across multiple walks.

        Returns
        -------
        List[TEMState]
            A sequence of per-step TEMState objects containing inferred and
            generated belief states (abstract and grounded), observations,
            memory states, and losses.
        """
        steps = None

        # Process each timestep in walk
        for locations, x, a in walk:
            # Initialize if first step
            if steps is None:
                steps = [self.init_state(locations, x, [None] * len(a), prev_M)]

            # Perform single TEM iteration - pass action separately
            step_output = self.iteration(
                x=x,
                locations=locations,
                action=a,
                state=steps[-1],
            )

            # Store iteration results
            steps.append(step_output)

        # Remove initialization step
        return steps[1:]

    def iteration(self, x: Tensor, locations: List[Dict], action: int, state: TEMState) -> Tuple[Losses, TEMState]:
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

    def loss(self, x: Tensor, g_gen: AbstractLocation, state: TEMState) -> Losses:
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

    def init_trainable(self):
        """Create and register all trainable parameters and sub-modules.

        This method should allocate PyTorch parameters and small networks used
        across the model, such as frequency-specific scale factors, MLPs for
        transition dynamics and uncertainties, and sensory compression /
        decompression layers. It plays the same role as ``init_trainable`` in
        the original implementation.
        """
        # All trainable components are already initialized in __init__
        # This method is kept for compatibility with the original API
        pass

    def init_walks(self, prev_iter):
        """Reset per-walk state when new walks start in a batch.

        Parameters
        ----------
        prev_iter:
            Optional previous iteration object or list of iterations. Entries
            corresponding to newly started walks (e.g. where the previous
            action is ``None``) should have their memory and state reset.

        Returns
        -------
        Any
            Updated previous iteration structure with per-walk state
            re-initialised where required, ready to be used by
            :meth:`iteration`.
        """
        if prev_iter is not None:
            state = prev_iter[0]
            # Handle both single action and batched actions
            actions = [state.action] if not isinstance(state.action, list) else state.action

            for a_i, a in enumerate(actions):
                if a is None:
                    # Reset memory for this walk
                    for M in state.memory:
                        M[a_i, :, :] = 0

                    # Reset abstract location to prior (from config or learned g_init)
                    for f, g_inf in enumerate(state.belief):
                        # Use learned g_init from abstract inference module if available
                        if hasattr(self.abstract, "g_init"):
                            g_inf[a_i, :] = self.abstract.g_init[f]
                        else:
                            g_inf[a_i, :] = torch.zeros(self.config.paramsitecture.n_g[f])

                    # Reset filtered sensory to zeros
                    for f, x_inf in enumerate(state.x_prev):
                        x_inf[a_i, :] = torch.zeros(self.config.paramsitecture.n_x_f[f])

        return prev_iter

    def init_state(
        self,
        locations: List[Dict],
        x: MultiScaleCode,
        a_prev: List[Optional[int]],
        M_prev: Optional[MemoryState],
    ) -> TEMState:
        """Initialize the first state of a walk sequence.

        Parameters
        ----------
        locations:
            Environment descriptors for each walk in the batch.
        x:
            Initial sensory observations.
        a_prev:
            Previous actions (typically all None for first iteration).
        M_prev:
            Previous memory state.

        Returns
        -------
        TEMState
            Initial state with priors and zero states.
        """
        batch_size = len(locations)
        n_freqs = len(self.config.paramsitecture.n_g)

        # Initialize abstract locations with priors or zeros
        g_inf = []
        for f in range(n_freqs):
            if hasattr(self.abstract, "g_init"):
                g_init_f = self.abstract.g_init[f].unsqueeze(0).expand(batch_size, -1)
            else:
                g_init_f = torch.zeros(batch_size, self.config.paramsitecture.n_g[f])
            g_inf.append(g_init_f)

        # Initialize filtered sensory as zeros
        x_inf = [torch.zeros(batch_size, n_x_f) for n_x_f in self.config.paramsitecture.n_x_f]

        # Initialize memory if not provided
        # Memory matrices are global (not per-batch) and stored in storage module
        if M_prev is None:
            M_gen = self.storage.M_gen
            M_inf = self.storage.M_inf if self.storage.use_dual_memory else M_gen
        else:
            M_gen = M_prev[0]
            M_inf = M_prev[1] if len(M_prev) > 1 else M_gen

        # Create initial losses (all zeros)
        initial_losses = Losses(
            L_p_g=torch.zeros(batch_size),
            L_p_x=torch.zeros(batch_size),
            L_x_gen=torch.zeros(batch_size),
            L_x_g=torch.zeros(batch_size),
            L_x_p=torch.zeros(batch_size),
            L_g=torch.zeros(batch_size),
            L_reg_g=torch.zeros(batch_size),
            L_reg_p=torch.zeros(batch_size),
        )

        # Create initial inference state
        initial_latent_pred = LatentPrediction(
            abstract=g_inf,
            grounded=[torch.zeros(batch_size, n_p) for n_p in self.config.paramsitecture.n_p],
        )

        initial_inf_state = InferenceState(
            memory_inf=M_inf,
            latent_prediction=initial_latent_pred,
            filtered_observation=x_inf,
            retrieved_grounded=None,
        )

        # Create initial generative state (placeholder values)
        initial_sensory_pred = SensoryPrediction(
            values=x_inf,
            logits=x_inf,
        )

        initial_gen_state = GenerativeState(
            memory_gen=M_gen,
            g_gen=g_inf,  # Use initial g_inf as placeholder
            sensory_prediction=initial_sensory_pred,
            x_p=initial_sensory_pred,
            x_g=initial_sensory_pred,
            x_gen=initial_sensory_pred,
            p_g=initial_latent_pred.grounded,
            p_gen=initial_latent_pred.grounded,
        )

        # Create state with initial values - data stored once in nested states
        state_0 = TEMState(
            losses=initial_losses,
            inference_state=initial_inf_state,
            generative_state=initial_gen_state,
        )

        return state_0
