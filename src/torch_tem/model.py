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
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from . import config, core
from . import hpc as hpc_
from . import lec as lec_
from . import losses
from . import mec as mec_
from . import utils
from .types import AbstractLocation, GroundedLocation, LocationInference, Matrix, MemoryState, MultiScaleCode, Observation, SensoryPrediction


class TEMParams(config.ModelConfig):
    """ """

    batch_size: int  # Batch size for parallel environments
    eta: float  # Learning rate for Hebbian memory update

    # TODO: Placeholder for TEM-specific parameters if needed


@dataclass
class TEMState:
    grounded_location: Optional[GroundedLocation]
    prediction: Optional[SensoryPrediction]
    lec: lec_.LECState
    mec: mec_.MECState

    @property
    def compressed_observation(self) -> MultiScaleCode:
        return self.lec.compressed_observation

    @property
    def filtered_observation(self) -> MultiScaleCode:
        return self.lec.filtered_observation

    @property
    def transition_stats(self) -> AbstractLocation:
        return self.mec.transition_stats

    @property
    def abstract_location(self) -> AbstractLocation:
        return self.mec.abstract_location


class TEMModel(nn.Module):
    def __init__(self, params: TEMParams):
        """ """
        super().__init__()
        self.batch_size = params.batch_size
        self.eta = params.eta

        # Initialize components
        self.grounded = core.GroundedLocInference(params)  # Hippocampal inference module
        self.memory = hpc_.Memory(params)  # Unified memory system (masks computed internally)
        self.lec = lec_.LECModel(params)  # LEC pathway module
        self.mec = mec_.MECModel(params)  # MEC pathway module

    def forward(self, x: Optional[Observation], locations: List[Dict], a: Optional[int], state: TEMState) -> TEMState:
        """ """

        # LEC Pathway steps; We process sensory input to prepare for memory retrieval
        state_lec = state.lec(x, state.lec) if x else state.lec  # Process observation: x → x_f
        x_ = state_lec.projection  # Project to hippocampal input: x_f → x_
        p_x = self.memory.retrieve(x_, for_inference=True) if x else None

        # MEC Pathway steps; We infer abstract location from action and previous location
        state_mec = self.mec(p_x, locations, a, state.mec)  # Update abstract location: g → g
        g_ = state_mec.projection  # Project to hippocampal input: g → g_
        p_g = self.memory.retrieve(g_, for_inference=False)  # Retrieve memory from grid cells

        # Infer predictions from LEC and MEC pathways
        x_hat = self.lec.decode(p_g)  # Decode observation: p → x (generate sensory prediction)
        p = self.grounded(g_, x_) if x is not None else None  # Infer hippocampus (grounded location)

        # Update memory with Hebbian plasticity and return new state
        if p_x is not None and p_g is not None:
            p_x_flat = torch.cat(p_x, dim=1)  # [B, sum(n_p)]
            p_g_flat = torch.cat(p_g, dim=1)  # [B, sum(n_p)]
            self.memory.update(p_x_flat, p_g_flat, self.eta)

        return TEMState(grounded_location=p, prediction=x_hat, lec=state_lec, mec=state_mec)

    def inference(self, x: Observation, locations: List[Dict], a: Optional[int], state: TEMState) -> TEMState:
        """ """
        state = self.forward(x, locations, a, state)  # What we expect to see after action
        return LocationInference(abstract=state.abstract_location, grounded=state.grounded_location)

    def generative(self, locations: List[Dict], a: int, state: TEMState) -> SensoryPrediction:
        """ """
        state = self.forward(None, locations, a, state)  # Where do I expect to be
        return state.prediction

    def init_state(self, x: Observation) -> TEMState:
        """ """
        lec_state = self.lec.init_state(x[0].device)  # Initialize LEC state
        mec_state = self.mec.init_state(x[0].device)  # Initialize MEC state
        return TEMState(grounded_location=None, prediction=None, lec=lec_state, mec=mec_state)

    def loss(self, x: Observation, g_gen: AbstractLocation, state: TEMState) -> losses.Losses:
        raise NotImplementedError("TEMModel.loss is not yet implemented.")


class Simulation(Iterator[TEMState]):
    def __init__(self, model: TEMModel, walk, memory: MemoryState):
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
        locations, x, a = next(self.__walk_iter)
        _, self.__state = self.__model.iteration(x, locations, a, self.__state)
        return self.__state
