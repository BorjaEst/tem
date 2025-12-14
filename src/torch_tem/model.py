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
from typing import Dict, List, Optional, Tuple

from torch import nn

from . import config, hpc, lec, losses, mec

from .types import AbstractLocation, GroundedLocation, LocationInference  # isort: skip
from .types import Observation, SensoryPrediction, MultiScaleCode  # isort: skip


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
        grounded: Tuple of (p_x, p_g, p) grounded locations.
            - p_x: Retrieved from sensory input (inference memory).
            - p_g: Retrieved from abstract location (generative memory).
            - p: Inferred from conjunction of sensory and abstract.
        lec: Lateral entorhinal cortex state.
        mec: Medial entorhinal cortex state.
    """

    grounded: Optional[Tuple[Optional[GroundedLocation], GroundedLocation, Optional[GroundedLocation]]]
    lec: lec.LECState  # LEC pathway state with sensory codes
    mec: mec.MECState  # MEC pathway state with abstract locations

    @property
    def grounded_sensory(self) -> Optional[GroundedLocation]:
        """Return the retrieved sensory grounded location from TEM state."""
        return self.grounded[0] if self.grounded is not None else None

    @property
    def grounded_abstract(self) -> Optional[GroundedLocation]:
        """Return the retrieved abstract grounded location from TEM state."""
        return self.grounded[1] if self.grounded is not None else None

    @property
    def grounded_location(self) -> Optional[GroundedLocation]:
        """Return the inferred grounded location from TEM state."""
        return self.grounded[2] if self.grounded is not None else None

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
        self.grounded = hpc.GroundedLocInference(params)  # Hippocampal inference module
        self.memory = hpc.Memory(params)  # Unified memory system (masks computed internally)
        self.lec = lec.LECModel(params)  # LEC pathway module
        self.mec = mec.MECModel(params)  # MEC pathway module

        # Initialize loss components
        self.loss_x_fn = losses.SensoryReconstructionLoss()
        self.loss_p_fn = losses.GroundedLocationLoss()
        self.loss_g_fn = losses.AbstractLocationLoss()
        self.loss_reg_fn = losses.RegularizationLoss()
        self.loss_total_fn = losses.TEMLoss()

    def init_state(self, x: Observation) -> TEMState:
        """Initialize TEM state from first observation.

        Args:
            x: Initial sensory observation (for device placement).

        Returns:
            Initial TEM state with zero-initialized locations.
        """
        lec_state = self.lec.init_state(x[0].device)  # Initialize LEC state
        mec_state = self.mec.init_state(x[0].device)  # Initialize MEC state
        return TEMState(grounded=None, lec=lec_state, mec=mec_state)

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

        # Infer grounded location and generate sensory prediction
        p = self.grounded(g_, x_) if x is not None else None  # Infer hippocampus (grounded location)
        return TEMState(grounded=(p_x, p_g, p), lec=state_lec, mec=state_mec)

    def inference(self, x: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> LocationInference:
        """Infer current location from sensory observation.

        Performs inference by combining sensory input with path integration to
        determine the current abstract and grounded locations.

        Args:
            x: Sensory observation.
            locations: Environment descriptors for landmark cues.
            a: Action taken (None for initial state).
            state: Previous TEM state.

        Returns:
            LocationInference with abstract location (g) and grounded location (p).
        """
        state = self.forward(x, locations, a, state)
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
        return self.lec.decode(state.grounded_abstract)  # What do I expect to see

    def update_memory(self, state: TEMState) -> None:
        """Update Hebbian memory using current TEM state.

        Implements the Hebbian update rule to strengthen associations between
        co-active patterns in the inference and generative pathways.

        Args:
            state: Current TEM state containing inferred and generated locations.
        """
        p_inferred = state.grounded_location  # p: Inferred from conjunction (g ⊗ x)
        p_generated = state.grounded_abstract  # p_g: Retrieved from abstract location (g)
        self.memory.update(p_inferred, p_generated, self.eta)

    def loss(self, x: Observation, state: TEMState) -> losses.LossOutput:
        """Compute Evidence Lower Bound (ELBO) loss for TEM.

        The total loss comprises three components following the TEM paper:
        1. L_x: Sensory reconstruction loss (from three pathways)
        2. L_p: Grounded location consistency loss
        3. L_g: Abstract location KL divergence loss

        Args:
            x: Ground truth sensory observation.
            state: Current TEM state with all pathway outputs.

        Returns:
            LossOutput containing total loss and individual components.
        """
        # Extract grounded locations from TEM state for loss computation
        p_x, p_g, p = state.grounded
        g_gen = self.mec.projection(state.mec.transition_stats.mean)  # Project predicted abstract location
        p_gen = self.memory.retrieve(g_gen, for_inference=False)  # Retrieve from generative memory

        # L_x: Sensory reconstruction from three pathways (teacher forcing)
        Lx = [
            # self.loss_x_fn(prediction=self.lec.decode(p_x), target=x),  # From sensory retrieval
            self.loss_x_fn(prediction=self.lec.decode(p_g), target=x),  # From abstract retrieval
            self.loss_x_fn(prediction=self.lec.decode(p), target=x),  # From inference
            self.loss_x_fn(prediction=self.lec.decode(p_gen), target=x),  # From generative prediction
        ]
        # L_p: Grounded location consistency (inference matches memory retrieval)
        Lp = self.loss_p_fn(p=p, p_g=p_g)
        # L_g: Abstract location KL divergence (posterior vs prior)
        Lg = self.loss_g_fn(g=state.abstract_location, g_gen=state.transition_stats)

        # Regularization losses
        L_reg_g, L_reg_p = self.loss_reg_fn(g=state.abstract_location, p=p)

        # Compute total ELBO
        return self.loss_total_fn(sum(Lx), Lp, Lg, L_reg_g, L_reg_p)


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
