"""PyTorch implementation of the Tolman-Eichenbaum Machine (TEM).

This module defines the high-level `TEMModel` for a modern, modular
TEM implementation. It mirrors the functionality of the original
`tem.model.Model` class, but is structured around typed configuration objects
and sub-modules for sensory encoding, inference, memory, and projection.

Implementation follows the reference model.py while using modular torch_tem
components for maintainability and testability.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Dict, List, Optional

from torch import nn

from . import config, core, hpc, lec, losses, mec
from .types import AbstractLocation, GroundedLocation, LocationInference, MultiScaleCode, Observation, SensoryPrediction


class TEMParams(config.ModelConfig):
    """Protocol for TEM model initialization parameters.

    Extends ModelConfig with TEM-specific training parameters.

    Attributes:
        batch_size: Batch size for parallel environments.
        eta: Learning rate for Hebbian memory update.
    """

    batch_size: int  # Batch size for parallel environments
    eta: float  # Learning rate for Hebbian memory update

    # TODO: Placeholder for TEM-specific parameters if needed


@dataclass
class TEMState:
    """State container for the TEM model.

    Maintains the complete state across all TEM pathways (LEC, MEC, HPC).

    Attributes:
        grounded_location: Inferred hippocampal place cells (inference mode).
        prediction: Predicted sensory observation (generative mode).
        lec: Lateral entorhinal cortex state.
        mec: Medial entorhinal cortex state.
    """

    grounded_location: Optional[GroundedLocation]
    prediction: Optional[SensoryPrediction]
    lec: lec.LECState
    mec: mec.MECState

    @property
    def compressed_observation(self) -> MultiScaleCode:
        """Return the compressed sensory observation from LEC state."""
        return self.lec.compressed_observation

    @property
    def filtered_observation(self) -> MultiScaleCode:
        """Return the filtered sensory observation from LEC state."""
        return self.lec.filtered_observation

    @property
    def transition_stats(self) -> AbstractLocation:
        """Return the transition statistics from MEC state."""
        return self.mec.transition_stats

    @property
    def abstract_location(self) -> AbstractLocation:
        """Return the abstract location from MEC state."""
        return self.mec.abstract_location


class TEMModel(nn.Module):
    """Tolman-Eichenbaum Machine (TEM) for spatial navigation and memory.

    TEM combines two pathways:
    - LEC pathway: Sensory processing (x → x_f → x_ → p_x)
    - MEC pathway: Abstract location processing (g → g_ → p_g)

    These converge in the hippocampus to form conjunctive place cells (p = g ⊗ x).
    Memory updates via Hebbian learning strengthen associations between co-active patterns.
    """

    def __init__(self, params: TEMParams):
        """Initialize TEM model.

        Args:
            params: Configuration with all component parameters and training settings.
        """
        super().__init__()
        self.batch_size = params.batch_size
        self.eta = params.eta

        # Initialize components
        self.grounded = core.GroundedLocInference(params)  # Hippocampal inference module
        self.memory = hpc.Memory(params)  # Unified memory system (masks computed internally)
        self.lec = lec.LECModel(params)  # LEC pathway module
        self.mec = mec.MECModel(params)  # MEC pathway module

    def forward(self, x: Optional[Observation], locations: List[Dict], a: Optional[int], state: TEMState) -> TEMState:
        """Forward pass through TEM model.

        Processes sensory input and actions to update abstract and grounded locations,
        generate predictions, and update memory via Hebbian learning.

        Args:
            x: Sensory observation (None for generative mode).
            locations: Environment descriptors for landmark cues.
            a: Action taken (None for initial state).
            state: Previous TEM state.

        Returns:
            Updated TEM state with new locations and predictions.
        """

        # LEC Pathway steps; We process sensory input to prepare for memory retrieval
        state_lec = self.lec(x, state.lec) if x is not None else state.lec  # Process observation: x → x_f
        x_ = state_lec.projection  # Project to hippocampal input: x_f → x_
        p_x = self.memory.retrieve(x_, for_inference=True) if x is not None else None

        # MEC Pathway steps; We infer abstract location from action and previous location
        state_mec = self.mec(p_x, locations, a, state.mec)  # Update abstract location: g → g
        g_ = state_mec.projection  # Project to hippocampal input: g → g_
        p_g = self.memory.retrieve(g_, for_inference=False)  # Retrieve memory from grid cells

        # Infer predictions from LEC and MEC pathways
        x_hat = self.lec.decode(p_g)  # Decode observation: p → x (generate sensory prediction)
        p = self.grounded(g_, x_) if x is not None else None  # Infer hippocampus (grounded location)

        if p is not None:  # Only when we have direct inference from both pathways
            self.memory.update(p, p_g, self.eta)

        return TEMState(grounded_location=p, prediction=x_hat, lec=state_lec, mec=state_mec)

    def inference(self, x: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> TEMState:
        """Infer current location from sensory observation.

        Args:
            x: Sensory observation.
            locations: Environment descriptors.
            a: Action taken.
            state: Previous TEM state.

        Returns:
            Location inference with abstract and grounded locations.
        """
        state = self.forward(x, locations, a, state)  # What we expect to see after action
        return LocationInference(abstract=state.abstract_location, grounded=state.grounded_location)

    def generative(self, locations: List[Dict], a: int, state: TEMState) -> SensoryPrediction:
        """Generate sensory prediction from action.

        Args:
            locations: Environment descriptors.
            a: Action to take.
            state: Current TEM state.

        Returns:
            Predicted sensory observation.
        """
        state = self.forward(None, locations, a, state)  # Where do I expect to be
        return state.prediction

    def init_state(self, x: Observation) -> TEMState:
        """Initialize TEM state from first observation.

        Args:
            x: Initial sensory observation (for device placement).

        Returns:
            Initial TEM state with zero-initialized locations.
        """
        lec_state = self.lec.init_state(x[0].device)  # Initialize LEC state
        mec_state = self.mec.init_state(x[0].device)  # Initialize MEC state
        return TEMState(grounded_location=None, prediction=None, lec=lec_state, mec=mec_state)

    def loss(self, x: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> losses.LossOutput:
        """Compute ELBO loss for TEM training.

        Implements the evidence lower bound (ELBO) following Gemici et al. (2017),
        training both the generative model q(g,p,x|a) and inference network f(g,p|x,a)
        jointly with pathway consistency constraints (teacher forcing).

        The ELBO decomposes into:
        - Reconstruction losses: p(x|p), p(x|g), p(x|g_prev,a)
        - Consistency losses: q(p|g_inf) ≈ f(p|x), q(g|a,g_prev) ≈ f(g|x)
        - Regularization: Priors on g (L2) and p (L1)

        Args:
            x: Sensory observation (ground truth).
            locations: Environment descriptors for landmark cues.
            a: Action taken.
            state: Previous TEM state.

        Returns:
            LossOutput with total ELBO loss and individual components.

        Example:
            >>> output = model.loss(x, locations, a, state)
            >>> output.total.backward()
        """
        # Process through both pathways to get all outputs
        state_updated = self.forward(x, locations, a, state)

        # Extract pathway outputs for ELBO computation
        # Generative pathway: q(g,p,x|a,g_prev) via transition → memory → decoder
        gen_outputs = _GenerativeOutputs(
            g=state_updated.mec.transition_stats.mean,  # q(g|a,g_prev): Generated abstract location
            p=state_updated.mec.projection,  # q(p|g): Retrieved grounded location (multi-scale)
            x=state_updated.prediction,  # q(x|p): Generated sensory prediction
        )

        # Inference pathway: f(g,p|x,a) via encoder → memory → grounded inference
        inf_outputs = _InferenceOutputs(
            g=state_updated.mec.abstract_location,  # f(g|x,p_x): Inferred abstract location
            p=state_updated.lec.projection,  # f(p_x|x): Sensory-retrieved grounded location (multi-scale)
            x=state_updated.prediction,  # Reconstructed observation (shared decoder)
            p_x=state_updated.grounded_location,  # f(p|g,x): Final grounded location inference
        )

        # Compute ELBO using TEMLoss (teacher forcing between pathways)
        loss_fn = losses.TEMLoss()
        return loss_fn(gen_outputs, inf_outputs, x)


# Internal helper classes for protocol compliance
@dataclass
class _GenerativeOutputs:
    """Internal implementation of GenerativeOutputs protocol."""

    g: List
    p: List
    x: SensoryPrediction


@dataclass
class _InferenceOutputs:
    """Internal implementation of InferenceOutputs protocol."""

    g: List
    p: Optional[List]
    x: SensoryPrediction
    p_x: Optional[List] = None


class Simulation(Iterator[TEMState]):
    """Iterator for running TEM model through a walk trajectory.

    Automatically processes walk data timestep by timestep, maintaining state
    and performing memory updates internally.

    Args:
        model: TEMModel instance (memory managed internally).
        walk: Walk object with observations, actions, and locations.

    Example:
        >>> from torch_tem.data import WalkGenerator
        >>> walk = walk_gen.generate_walks(n_walks=1, walk_length=100)[0]
        >>> sim = Simulation(model, walk)
        >>> for state in sim:
        ...     # Process state.grounded_location, state.prediction, etc.
        ...     pass
    """

    def __init__(self, model: TEMModel, walk):
        self.__model = model
        self.__walk = walk
        self.__walk_length = len(walk)
        self.__current_step = 0

        # Initialize state with first observation
        first_x = walk.observations[0].unsqueeze(0)  # [n_x] -> [1, n_x]
        self.__state = self.__model.init_state(first_x)

    def __iter__(self):
        """Return iterator interface."""
        return self

    def __next__(self) -> TEMState:
        """Process next timestep and return updated state.

        Returns:
            TEMState with updated locations, predictions, and memory.

        Raises:
            StopIteration: When walk is complete.
        """
        if self.__current_step >= self.__walk_length:
            raise StopIteration

        # Extract current timestep data
        x = self.__walk.observations[self.__current_step].unsqueeze(0)  # [1, n_x]
        # First timestep uses action index 0 (no previous action to learn from)
        a = self.__walk.actions[self.__current_step].unsqueeze(0)  # [1]

        # Format locations as list of dicts (expected by TEMModel.forward)
        # Note: Shiny markers not currently used in examples
        locations = [{"shiny": None}]

        # Process through TEM model (updates internal memory automatically)
        self.__state = self.__model.forward(x, locations, a, self.__state)

        self.__current_step += 1
        return self.__state
