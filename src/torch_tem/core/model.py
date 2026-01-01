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
from typing import Dict, List, Optional, Protocol, Tuple

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from torch import nn

from torch_tem.core.hpc import HPCConfig, HPCModel, HPCState
from torch_tem.core.lec import LECConfig, LECModel, LECState
from torch_tem.core.mec import MECConfig, MECModel, MECState
from torch_tem.data.environment import Environment
from torch_tem.modules.projection import LECPConfig, MECPConfig, Projection

from torch_tem.types import AbstractLocation, GroundedLocation, LocationInference  # isort: skip
from torch_tem.types import Observation, SensoryPrediction, MultiScaleCode  # isort: skip
from torch_tem.types import Matrix, BatchedMemory  # isort: skip


__all__ = ["TEMConfig", "TEMContext", "StandardTEMContext", "TEMState", "TEMModel", "Simulation"]


class TEMConfig(BaseModel):
    """TEM model configuration parameters.

    Combines configurations for all TEM components:
    - LEC pathway (sensory processing)
    - MEC pathway (abstract location processing)
    - HPC pathway (memory and grounded inference)
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    lec: LECConfig = Field(default_factory=LECConfig, description="LEC pathway configuration")
    mec: MECConfig = Field(default_factory=MECConfig, description="MEC pathway configuration")
    hpc: HPCConfig = Field(default_factory=HPCConfig, description="HPC pathway configuration")

    projection_lec: Projection = Field(default_factory=LECPConfig, description="LEC to HPC projection configuration")
    projection_mec: Projection = Field(default_factory=MECPConfig, description="LEC to HPC projection configuration")


class TEMContext(Protocol):
    """Protocol for TEM model initialization parameters.

    Attributes:
        environment: Environment instance (provides n_observations, n_actions, n_locations)
        f_initial: Grid frequency values for hierarchical connections [n_f_grid]
        W_tile: Tiling matrices for LEC projection [n_f_total]
        W_down: Downsampling matrices for MEC projection [n_f_total]
        W_repeat: Expansion matrices for MEC projection [n_f_total]
    """

    environment: Environment
    f_initial: List[float]
    W_tile: List[Matrix]
    W_down: List[Matrix]
    W_repeat: List[Matrix]

    @property
    def n_o(self) -> int: ...

    @property
    def n_a(self) -> int: ...

    @property
    def n_f(self) -> int: ...


@dataclass
class StandardTEMContext:
    """Standard implementation of TEMContext protocol.

    Provides default property implementations deriving values from core fields.
    Structurally conforms to TEMContext protocol without explicit inheritance.
    """

    environment: Environment
    f_initial: List[float]
    W_tile: List[Matrix]
    W_down: List[Matrix]
    W_repeat: List[Matrix]
    mask_inference: List[Matrix]
    mask_generative: List[Matrix]
    update_mask: Matrix

    @property
    def n_o(self) -> int:
        """Number of sensory observation neurons."""
        return self.environment.n_observations

    @property
    def n_a(self) -> int:
        """Number of possible actions."""
        return self.environment.n_actions

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return len(self.W_down)


class TEMState:
    """State container for the TEM model.

    Maintains the complete state across all TEM pathways (LEC, MEC, HPC).

    Attributes:
        hpc: Hippocampus state with grounded locations and memory.
        lec: Lateral entorhinal cortex state.
        mec: Medial entorhinal cortex state.
    """

    hpc: HPCState  # Hippocampal state with memory codes
    lec: LECState  # LEC pathway state with sensory codes
    mec: MECState  # MEC pathway state with abstract locations
    pathways: Optional[Tuple[GroundedLocation, GroundedLocation]] = None

    def detach(self) -> "TEMState":
        """Detach all tensors in the TEM state from the computation graph."""
        if self.pathways:
            pathways_detached = (
                [p.detach() for p in self.pathways[0]],
                [p.detach() for p in self.pathways[1]],
            )
        else:
            pathways_detached = None
        return TEMState(
            hpc=self.hpc.detach(),
            lec=self.lec.detach(),
            mec=self.mec.detach(),
            pathways=pathways_detached,
        )

    @property
    def grounded_location(self) -> Optional[GroundedLocation]:
        """Return the inferred grounded location from TEM state."""
        return self.hpc.grounded_location

    @property
    def grounded(self) -> Tuple[Optional[GroundedLocation], Optional[GroundedLocation], GroundedLocation]:
        """Return tuple of (p_x, p_g, p)."""
        return self.grounded_sensory, self.grounded_abstract, self.grounded_location

    @property
    def grounded_sensory(self) -> Optional[GroundedLocation]:
        """Return retrieved grounded location from sensory pathway (p_x)."""
        return self.pathways[0] if self.pathways else None

    @property
    def grounded_abstract(self) -> Optional[GroundedLocation]:
        """Return retrieved grounded location from abstract pathway (p_g)."""
        return self.pathways[1] if self.pathways else None

    @property
    def memory(self) -> List[BatchedMemory]:
        """Return the hippocampal memory matrices from HPC state."""
        return self.hpc.memory

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
    - LEC pathway: Sensory processing (x → x → x_ → p_x)
    - MEC pathway: Abstract location processing (g → g_ → p_g)

    These converge in the hippocampus to form conjunctive place cells (p = g ⊗ x).
    Memory updates via Hebbian learning strengthen associations between co-active patterns.
    """

    def __init__(self, context: TEMContext, config: TEMConfig):
        """Initialize TEM model.

        Args:
            context: Model initialization parameters (architecture)
            config: Configuration with all component parameters and training settings
        """
        super().__init__()
        self._config = config  # Store model configuration

        # Initialize components
        self.hpc = HPCModel(context, config.hpc)  # Hippocampus with memory and grounded inference
        self.lec = LECModel(context, config.lec)  # LEC pathway module
        self.mec = MECModel(context, config.mec)  # MEC pathway module

        # Initialize projections
        self.projection_lec = Projection(n_in=self.lec.n_x, n_out=self.hpc.n_p, config=config.projection_lec)
        self.projection_mec = Projection(n_in=self.mec.n_g, n_out=self.hpc.n_p, config=config.projection_mec)

    def init_state(self, o: Observation) -> TEMState:
        """Initialize TEM state from first observation.

        Args:
            o: Initial sensory observation [B, n_o] for device placement.

        Returns:
            Initial TEM state with zero-initialized locations.
        """
        hpc_state: HPCState = self.hpc.init_state(o.device)  # Initialize HPC state
        lec_state: LECState = self.lec.init_state(o.device)  # Initialize LEC state
        mec_state: MECState = self.mec.init_state(o.device)  # Initialize MEC state
        return TEMState(hpc=hpc_state, lec=lec_state, mec=mec_state)

    def forward(self, o: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> TEMState:
        """Forward pass through TEM model.

        Processes sensory input and actions to update abstract and grounded locations,
        generate predictions, and update memory via Hebbian learning.

        Args:
            o: Sensory observation.
            locations: Environment descriptors for landmark cues.
            a: Action taken (None for initial state).
            state: Previous TEM state.

        Returns:
            Updated TEM state with new locations and predictions.
        """

        # LEC Pathway: Process sensory input to prepare for memory retrieval
        state_lec: LECState = self.lec(o, state.lec)
        x = state_lec.filtered_observation  # Filtered sensory code
        x_ = self.projection_lec(x)  # Projected sensory code for HPC retrieval
        p_x = self.hpc.retrieve(x_, for_inference=True, state=state.hpc)
        x = self.projection_lec.inverse(p_x)  # Reconstructed sensory code

        # MEC Pathway: Infer abstract location from action and previous location
        state_mec: MECState = self.mec(x, locations, a, state.mec)
        g = state_mec.abstract_location  # Abstract location
        g_ = self.projection_mec(g)  # Projected abstract location for HPC retrieval
        p_g = self.hpc.retrieve(g_, for_inference=False, state=state.hpc)

        # HPC Pathway: Infer grounded location and update memory
        state_hpc: HPCState = self.hpc(g_, x_, p_g, state.hpc)

        # Store pathways for teacher forcing in loss computation
        return TEMState(hpc=state_hpc, lec=state_lec, mec=state_mec, pathways=(p_x, p_g))

    def inference(self, o: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> LocationInference:
        """Infer current location from sensory observation.

        Performs inference by combining sensory input with path integration to
        determine the current abstract and grounded locations.

        Args:
            o: Sensory observation.
            locations: Environment descriptors for landmark cues.
            a: Action taken (None for initial state).
            state: Previous TEM state.

        Returns:
            LocationInference with abstract location (g) and grounded location (p).
        """
        # LEC Pathway: Process sensory input to prepare for memory retrieval
        state_lec: LECState = self.lec(o, state.lec)  # Process sensory input through LEC
        x_ = state_lec.projection  # Projected sensory code for HPC retrieval
        p_x = self.hpc.retrieve(x_, for_inference=True, state=state.hpc)

        # MEC Pathway: Infer abstract location from action and previous location
        state_mec: MECState = self.mec(p_x, locations, a, state.mec)  # Infer abstract location via MEC
        g_ = state_mec.projection

        # HPC Pathway: Infer grounded location from abstract location and sensory input
        p = self.hpc.grounded(g_, x_)
        return LocationInference(abstract=state_mec.abstract_location, grounded=p)

    def generative(self, locations: List[Dict], a: Optional[int], state: TEMState) -> SensoryPrediction:
        """Generate sensory prediction from action.

        Args:
            locations: Environment descriptors.
            a: Action to take.
            state: Current TEM state.

        Returns:
            Predicted sensory observation.
        """
        # Update MEC state via path integration
        state_mec = self.mec(None, locations, a, state.mec)
        g_ = state_mec.projection

        # Retrieve grounded location from generative memory
        p_g = self.hpc.retrieve(g_, for_inference=False, state=state.hpc)

        # Decode to sensory prediction
        return self.lec.decode(p_g)


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
        first_x = walk.observations[0].unsqueeze(0)  # [n_o] -> [1, n_o]
        self.__state = self.__model.init_state(first_x)

    def __iter__(self) -> Iterator[TEMState]:
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
        x = self.__walk.observations[self.__current_step].unsqueeze(0)  # [1, n_o]
        # First timestep uses action index 0 (no previous action to learn from)
        a = self.__walk.actions[self.__current_step].unsqueeze(0)  # [1]

        # Format locations as list of dicts (expected by TEMModel.forward)
        # Note: Shiny markers not currently used in examples
        locations = [{"shiny": None}]

        # Process through TEM model (updates internal memory automatically)
        self.__state = self.__model.forward(x, locations, a, self.__state)

        self.__current_step += 1
        return self.__state
