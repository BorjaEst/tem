from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import tee
from typing import List, Literal, Optional, Sequence, Tuple, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules.autoencoder import AutoencoderModule
from torch_tem.modules.hpc import HPCModel, HPCState
from torch_tem.modules.lec import LECModel, LECState
from torch_tem.modules.mec import MECModel, MECState
from torch_tem.modules.projection import ProjectionModule
from torch_tem.settings import AutoencoderSettings, HPCSettings, LECProjectionSettings, LECSettings, MECProjectionSettings, MECSettings, SpaceContractSettings
from torch_tem.types import AbstractLocation, GroundedLocation, LocationLabel, MultiScaleCode, Observation, Reduction, Scalar, Walk


class TEMConfig(BaseModel):
    """Complete settings tree for TEM model configuration."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    space_contract: SpaceContractSettings = Field(
        default_factory=SpaceContractSettings,
        description="Settings for the space contract used by the model.",
    )
    autoencoder: AutoencoderSettings = Field(
        default_factory=AutoencoderSettings,
        description="Autoencoder module settings.",
    )
    f_initial: List[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01],
        frozen=True,
        description="Initial spatial frequencies for multi-scale modules.",
    )
    n_features: int = Field(
        default=10,
        frozen=True,
        description="Number of LEC context features.",
    )
    lec_settings: LECSettings = Field(
        default_factory=LECSettings,
        description="LEC module settings.",
    )
    lec_projection: LECProjectionSettings = Field(
        default_factory=LECProjectionSettings,
        description="LEC projection module settings.",
    )
    n_grids: List[int] = Field(
        default_factory=lambda: [30, 30, 24, 18, 18],
        frozen=True,
        description="Number of MEC neurons per frequency module.",
    )
    n_ovc: Union[Literal["off", "merged"], List[int]] = Field(
        default="merged",
        frozen=True,
        description="Number of OVC neurons per frequency module. 'merged' to merge with n_grids.",
    )
    mec_settings: MECSettings = Field(
        default_factory=MECSettings,
        description="MEC module settings.",
    )
    mec_projection: MECProjectionSettings = Field(
        default_factory=MECProjectionSettings,
        description="MEC projection module settings.",
    )
    n_hippocampal: List[int] = Field(
        default_factory=lambda: [100, 100, 80, 60, 60],
        frozen=True,
        description="Number of HPC neurons per frequency module.",
    )
    hpc_settings: HPCSettings = Field(
        default_factory=HPCSettings,
        description="HPC module settings.",
    )
    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether to use inferred ground location while inferring new abstract location",
    )


@dataclass
class TEMLabel:
    observation: Observation  # True sensory observation (for loss computation)
    locations: List[LocationLabel]  # True locations (for loss computation)


@dataclass
class TEMState:
    lec: LECState
    mec: MECState
    hpc: HPCState

    def detach(self) -> "TEMState":
        states = [x.detach() for x in (self.lec, self.mec, self.hpc)]
        return TEMState(*states)


@dataclass
class TEMInference:
    g_inf: AbstractLocation
    p_inf: GroundedLocation
    p_xi: GroundedLocation


@dataclass
class TEMGenerative:
    g_gen: AbstractLocation
    p_gen_gg: GroundedLocation
    p_gen_gi: GroundedLocation


@dataclass
class TEMReconstruction:
    o_hat: Sequence[Observation]
    o_logits: Sequence[Tensor]


@dataclass
class TEMOutput:
    inference: TEMInference
    generative: TEMGenerative
    reconstruction: TEMReconstruction


class Model(nn.Module):
    def __init__(self, config: Optional[TEMConfig] = None):
        super().__init__()
        self._config = config or TEMConfig()
        n_observations = config.space_contract.n_observations  # Number of observation dimensions
        n_actions = config.space_contract.n_actions_move  # Number of possible discrete actions
        n_features = config.n_features  # Number of LEC features (compressed observation)
        n_grids = config.n_grids  # Number of MEC grid cells per frequency
        n_ovc = config.n_ovc if config.n_ovc != "off" else []  # Number of MEC OVC cells per frequency
        n_ovc = config.n_ovc if config.n_ovc != "merged" else None  # Merge OVC with grid cells
        n_hippocampal = config.n_hippocampal  # Number of HPC place cells per frequency
        f_initial = config.f_initial  # Initial firing rate for all cells

        # Autoencoder module for observation compression/decoding
        self.autoencoder = AutoencoderModule(n_observations, n_features, settings=config.autoencoder)

        # Entorhinal Hippocampal Circuit components
        self.lec = LECModel(n_features, f_initial, settings=config.lec_settings)
        self.mec = MECModel(n_actions, n_hippocampal, n_grids, n_ovc, f_initial, settings=config.mec_settings)
        self.hpc = HPCModel(len(n_grids), n_hippocampal, f_initial, settings=config.hpc_settings)

        # Projection modules
        self.lec_projection = ProjectionModule(self.lec, self.hpc, settings=config.lec_projection)
        self.mec_projection = ProjectionModule(self.mec, self.hpc, settings=config.mec_projection)

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> TEMState:
        lec_state = self.lec.init_state(batch_size, device)
        state_mec = self.mec.init_state(batch_size, device)
        hpc_state = self.hpc.init_state(batch_size, device)
        return TEMState(lec_state, state_mec, hpc_state)

    def set_runtime(self, eta: float, hebbian_decay: float, p2g_uncertainty_offset: float) -> None:
        """Set runtime hyperparameters (called by training loop each step).

        Args:
            eta: Hebbian learning rate (rate of remembering)
            hebbian_decay: Hebbian decay factor (rate of forgetting)
            p2g_uncertainty_offset: Additive uncertainty offset for p->g inference
        """
        self.mec.set_runtime(p2g_uncertainty_offset=p2g_uncertainty_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    @property
    def config(self) -> TEMConfig:
        """Return TEM settings object constructed from model parameters."""
        return self._config

    @property
    def n_observations(self) -> int:
        """Return the number of observation dimensions."""
        return self.autoencoder.n_observations

    @property
    def n_features(self) -> int:
        """Return the number of LEC features (compressed observation)."""
        return self.autoencoder.n_features

    @property
    def n_actions(self) -> int:
        """Return the number of possible discrete actions."""
        return self.mec.n_actions

    @property
    def n_grids(self) -> List[int]:
        """Return the number of MEC grid cells per frequency."""
        return self.mec.n_grids

    @property
    def n_ovc(self) -> Optional[List[int]]:
        """Return the number of MEC OVC cells per frequency (or None)."""
        return self.mec.n_ovc

    @property
    def n_hippocampal(self) -> List[int]:
        """Return the number of HPC place cells per frequency."""
        return self.hpc.shape

    def forward(self, observation: Observation, locations: List[LocationLabel], a_prev: List[Optional[int]], state: TEMState) -> tuple[TEMOutput, TEMState]:
        state = self.setup_state(state, a_prev, observation.device)
        device = observation.device  # Get device from observation tensor
        actions = utils.one_hot_with_zero(a_prev, self.n_actions, device=device)

        features = self.autoencoder.encode(observation)  # Encode observation to compressed format
        inference, generative, state = self.step(features, locations, actions, state)
        output = self.compute_output(inference, generative, observation)

        # Build full output, state and return
        return output, state

    def setup_state(self, state: TEMState, a_prev: List[Optional[int]], device: torch.device) -> TEMState:
        # Handle reset boundaries: where a_prev is None, reset state to priors before transition
        # Handle reset boundaries: where a_prev is None, reset state to priors before transition
        # TODO: This responsability to reset state should go somewhere else, e.g., in the RolloutStream class
        # TODO: Then we do not need None actions, and use a Tensor[int] for a_prev
        state_lec, state_mec, state_hpc = state.lec, state.mec, state.hpc
        reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool, device=device)
        if torch.any(reset_mask):
            # Reset g to priors for envs with no previous action
            g_reset = [torch.where(reset_mask.unsqueeze(-1), self.mec.cells_init[f].unsqueeze(0), state.mec.cells[f]) for f in range(self.mec.n_freq)]
            state_mec = state.mec.new(g_reset, uncertainty=None)
        return TEMState(state_lec, state_mec, state_hpc)

    def step(self, features: MultiScaleCode, locations: List[LocationLabel], actions: Tensor, state: TEMState) -> tuple[TEMInference, TEMGenerative, TEMState]:
        # Observe / infer: LEC filtering + HPC retrieval + MEC correction
        x_inf, state.lec = self.lec.inference(features, state.lec)
        x_ = self.lec_projection(x_inf)  # Project to memory format
        p_xi = self.hpc.recall(x_, state.hpc, mode="full") if self.config.use_x_cued_recall else None

        # LocationBelief: MEC path integration (action-driven)
        g_gen, state.mec = self.mec.generative(actions, locations, state.mec)  # Updates mec state with g_path
        g_ = self.mec_projection(g_gen)
        p_gg = self.hpc.recall(g_, state.hpc, mode="hierarchical")

        # Infer abstract location by using state and sensory experience
        g_inf, state.mec = self.mec.inference(p_xi, locations=locations, state=state.mec)
        g_ = self.mec_projection(g_inf)
        p_gi = self.hpc.recall(g_, state.hpc, mode="hierarchical")

        # Generate grounded location from inferred abstract location
        p_gen_gi, state.hpc = self.hpc.generative(p_gi, state.hpc)
        p_gen_gg, state.hpc = self.hpc.generative(p_gg, state.hpc)

        # Infer grounded location from abstract location and sensory experience
        p_inf, state.hpc = self.hpc.inference(x_, g_, state.hpc)

        # Build tem generative and inference interfaces
        generative = TEMGenerative(g_gen=g_gen, p_gen_gg=p_gen_gg, p_gen_gi=p_gen_gi)
        inference = TEMInference(g_inf=g_inf, p_inf=p_inf, p_xi=p_xi)

        # Update memory and return new state
        state.hpc = self.hpc.update(p_inf, p_gen_gi, p_xi, state.hpc)
        return inference, generative, TEMState(state.lec, state.mec, state.hpc)

    def compute_output(self, inference: TEMInference, generative: TEMGenerative, observation: Observation) -> TEMOutput:
        # Generate observation prediction from inferred grounded location
        x = self.lec_projection.inverse(inference.p_inf)
        c_p_inf = self.lec.generative(x)
        o_p_inf_logits = self.autoencoder.decode(c_p_inf)
        o_p_inf = utils.softmax(o_p_inf_logits)

        # Generate observation from inferred grounded location
        x = self.lec_projection.inverse(generative.p_gen_gi)
        c_p_gen_gi = self.lec.generative(x)
        o_gen_gi_logits = self.autoencoder.decode(c_p_gen_gi)
        o_gen_gi = utils.softmax(o_gen_gi_logits)

        # Generate observation from generated grounded location
        x = self.lec_projection.inverse(generative.p_gen_gg)
        c_p_gen_gg = self.lec.generative(x)
        x_gen_gg_logits = self.autoencoder.decode(c_p_gen_gg)
        o_gen_gg = utils.softmax(x_gen_gg_logits)

        # Return all generated observations and their corresponding logits
        o_hat, o_logits = (o_p_inf, o_gen_gi, o_gen_gg), (o_p_inf_logits, o_gen_gi_logits, x_gen_gg_logits)
        reconstructions = TEMReconstruction(o_hat=o_hat, o_logits=o_logits)
        return TEMOutput(inference=inference, generative=generative, reconstruction=reconstructions)


class RolloutStream(Iterator[Tuple[TEMOutput, TEMLabel, TEMState]]):
    """ """

    # TODO: Add docstring

    def __init__(self, model: Model, walk: Walk, initial: Optional[TEMState] = None):
        """ """
        # TODO: Add docstring
        self.model = model  # TEM model to rollout
        self.walk = iter(walk)  # Walk to rollout over

        # Peek at first observation to get batch size and device
        _, first_observation, _ = next(tee(self.walk)[0])  # Peek at first observation
        batch_size, device = first_observation.shape[0], first_observation.device

        # Initialize state and previous actions
        self._state = initial or model.init_state(batch_size, device)
        self._a_prev = [None for _ in range(first_observation.shape[0])]

    def __iter__(self) -> "RolloutStream":
        """Return self as iterator."""
        return self

    @property
    def state(self) -> TEMState:
        """Return current TEM state."""
        return self._state

    @property
    def previous_action(self) -> List[Optional[int]]:
        """Return previous actions."""
        return self._a_prev

    def __next__(self) -> TEMState:
        """Process next timestep and return state.

        Returns:
            TEMState for current timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        locations, observation, action = next(self.walk)
        output, self._state = self.model(observation, locations, self.previous_action, self._state)
        labels = TEMLabel(observation, locations)
        self._a_prev = action  # Update action for next iteration
        return output, labels, self.state
