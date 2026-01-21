from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.core.hpc import HPCModel, HPCState
from torch_tem.core.lec import LECModel, LECState
from torch_tem.core.mec import MECModel, MECState
from torch_tem.modules.autoencoder import AutoencoderModule
from torch_tem.modules.projection import ProjectionModule
from torch_tem.settings import TEMSettings
from torch_tem.types import Transition, Walk


@dataclass
class TEMLabel:
    o: Tensor  # True sensory observation (for loss computation)
    locations: List[Tensor]  # True locations (for loss computation)


@dataclass
class TEMState:
    lec_state: LECState
    mec_state: MECState
    hpc_state: HPCState

    def detach(self) -> "TEMState":
        return TEMState(
            lec_state=self.lec_state.detach(),
            mec_state=self.mec_state.detach(),
            hpc_state=self.hpc_state.detach(),
        )


@dataclass
class TEMInference:
    g_inf: List[Tensor]
    p_inf: List[Tensor]
    p_xi: List[Tensor]


@dataclass
class TEMGenerative:
    g_gen: List[Tensor]
    p_gen_gg: List[Tensor]
    p_gen_gi: List[Tensor]


@dataclass
class TEMReconstruction:
    o_hat: Sequence[Tensor]
    o_logits: Sequence[Tensor]


@dataclass
class TEMOutput:
    inference: TEMInference
    generative: TEMGenerative
    reconstruction: TEMReconstruction


class TEMModel(nn.Module):
    def __init__(self, n_observations: int, n_actions: int, settings: Optional[TEMSettings] = None):
        super().__init__()
        self._settings = settings or TEMSettings()
        n_features = settings.n_features  # Number of LEC features (compressed observation)
        n_grids = settings.n_grids  # Number of MEC grid cells per frequency
        n_ovc = settings.n_ovc  # Number of OVC cells per frequency (or None)
        n_hippocampal = settings.n_hippocampal  # Number of HPC place cells per frequency
        f_initial = settings.f_initial  # Initial firing rate for all cells

        # Autoencoder module for observation compression/decoding
        self.autoencoder = AutoencoderModule(n_observations, n_features, settings=settings.autoencoder)

        # Entorhinal Hippocampal Circuit components
        self.lec = lec = LECModel(n_features, f_initial, settings=settings.lec_settings)
        self.mec = mec = MECModel(n_actions, n_hippocampal, n_grids, n_ovc, f_initial, settings=settings.mec_settings)
        self.hpc = hpc = HPCModel(len(n_grids), n_hippocampal, f_initial, settings=settings.hpc_settings)

        # Projection modules
        self.lec_projection = ProjectionModule(lec, hpc, settings=settings.lec_projection)
        self.mec_projection = ProjectionModule(mec, hpc, settings=settings.mec_projection)

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
    def settings(self) -> TEMSettings:
        """Return TEM settings object constructed from model parameters."""
        return self._settings

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

    def forward(self, o, locations, a_prev, state: TEMState) -> tuple[TEMOutput, TEMState]:
        mec_state, lec_state, hpc_state = state.mec_state, state.lec_state, state.hpc_state
        c = self.autoencoder.encode(o)
        device = o.device

        # Handle reset boundaries: where a_prev is None, reset state to priors before transition
        reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool, device=device)
        if torch.any(reset_mask):
            # Reset g to priors for envs with no previous action
            g_reset = [torch.where(reset_mask.unsqueeze(-1), self.mec.cells_init[f].unsqueeze(0), mec_state.cells[f]) for f in range(self.mec.n_freq)]
            mec_state.transition = Transition(g_reset, uncertainty=None)

        # Convert actions to one-hot format expected by MEC (use 0 for None, will be reset above)
        a = utils.one_hot_with_zero(a_prev, self.n_actions, device=device)

        # Observe / infer: LEC filtering + HPC retrieval + MEC correction
        x_inf, lec_state = self.lec.inference(c, lec_state)
        x_ = self.lec_projection(x_inf)  # Project to memory format
        p_xi = self.hpc.recall(x_, hpc_state, mode="full") if self.settings.use_x_cued_recall else None

        # Transition: MEC path integration (action-driven)
        g_gen, mec_state = self.mec.generative(a, locations, mec_state)  # Updates mec state with g_path
        g_ = self.mec_projection(g_gen)
        p_gg = self.hpc.recall(g_, hpc_state, mode="hierarchical")

        # Infer abstract location by using state and sensory experience
        g_inf, mec_state = self.mec.inference(p_xi, locations=locations, state=mec_state)
        g_ = self.mec_projection(g_inf)
        p_gi = self.hpc.recall(g_, hpc_state, mode="hierarchical")

        # Generate grounded location from inferred abstract location
        p_gen_gi, hpc_state = self.hpc.generative(p_gi, hpc_state)
        p_gen_gg, hpc_state = self.hpc.generative(p_gg, hpc_state)

        # Infer grounded location from abstract location and sensory experience
        p_inf, hpc_state = self.hpc.inference(x_, g_, hpc_state)

        # Update memory (Hebbian write)
        hpc_state = self.hpc.update(p_inf, p_gen_gi, p_xi, hpc_state)

        # Build tem state
        state = TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)

        # Build tem generative and inference interfaces
        generative = TEMGenerative(g_gen=g_gen, p_gen_gg=p_gen_gg, p_gen_gi=p_gen_gi)
        inference = TEMInference(g_inf=g_inf, p_inf=p_inf, p_xi=p_xi)

        # Generate observation prediction from inferred grounded location
        x = self.lec_projection.inverse(p_inf)
        c_p_inf = self.lec.generative(x)
        o_p_inf_logits = self.autoencoder.decode(c_p_inf)
        o_p_inf = utils.softmax(o_p_inf_logits)

        # Generate observation from inferred grounded location
        x = self.lec_projection.inverse(p_gen_gi)
        c_p_gen_gi = self.lec.generative(x)
        o_gen_gi_logits = self.autoencoder.decode(c_p_gen_gi)
        o_gen_gi = utils.softmax(o_gen_gi_logits)

        # Generate observation from generated grounded location
        x = self.lec_projection.inverse(p_gen_gg)
        c_p_gen_gg = self.lec.generative(x)
        x_gen_gg_logits = self.autoencoder.decode(c_p_gen_gg)
        o_gen_gg = utils.softmax(x_gen_gg_logits)

        # Return all generated observations and their corresponding logits
        o_hat, o_logits = (o_p_inf, o_gen_gi, o_gen_gg), (o_p_inf_logits, o_gen_gi_logits, x_gen_gg_logits)
        reconstructions = TEMReconstruction(o_hat=o_hat, o_logits=o_logits)

        # Build full output, state and return
        output = TEMOutput(inference=inference, generative=generative, reconstruction=reconstructions)
        state = TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)
        return output, state

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> TEMState:
        # Create initial LEC state (x starts as x_filtered since no scaling/normalization yet)
        lec_state = self.lec.init_state(batch_size, device)
        # Initialise previous abstract location by stacking abstract location prior
        mec_state = self.mec.init_state(batch_size, device)
        # Create initial HPC state with initialized memory
        hpc_state = self.hpc.init_state(batch_size, device)
        # And construct new iteration for that g, o, a, and M
        return TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)


class Rollout(Iterator[TEMState]):
    def __init__(self, model: TEMModel, walk: Walk, initial: Optional[TEMState] = None):
        self.model = model
        self.walk = list(walk)  # Materialize for predictable indexing

        if len(self.walk) == 0:
            raise ValueError("Rollout requires at least 1 timestep in walk")

        # Extract first step to determine batch size and initialize state
        locations_0, o_0, _ = self.walk[0]

        # Determine initial state
        state = initial or model.init_state(batch_size=o_0.shape[0], device=o_0.device)

        # Initialize prev-values for first forward pass
        self._a_prev = [None for _ in range(o_0.shape[0])]
        self._state = state

        # Track current position in walk
        self._idx = 0

    def __iter__(self) -> "Rollout":
        """Return self as iterator."""
        return self

    def __next__(self) -> TEMState:
        """Process next timestep and return state.

        Returns:
            TEMState for current timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._idx >= len(self.walk):
            raise StopIteration

        # Get current timestep
        locations, o, a = self.walk[self._idx]
        self._idx += 1

        # Run model forward
        output, state = self.model(o, locations, self._a_prev, self._state)

        # Build state with updated components
        state.g = locations
        state.a_prev = a

        # Update prev-values for next iteration
        self._a_prev = a
        self._state = state

        # Build labels for current timestep
        labels = TEMLabel(o=o, locations=locations)

        return output, labels, state
