from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from .. import utils
from ..core.projection import ProjectionHead
from ..generation import ObservationDecoder
from ..memory.attractor import AttractorDynamics
from ..types import GroundedLocation, LocationInference, Matrix, MultiScaleCode, SensoryObservation, SensoryPrediction, TransitionParams
from . import abstract, grounded, sensory
from .abstract import AbstractLocParams
from .grounded import GroundedLocParams
from .sensory import EncoderParams, ProcessorParams, ProjectionParams


class Parameters(EncoderParams, ProcessorParams, ProjectionParams, GroundedLocParams, AbstractLocParams):
    """Parameters needed by InferenceModel."""

    n_x: int
    n_x_c: int
    n_g_subsampled_combined: int
    n_x_f: int


@dataclass
class InferenceState:
    """Auxiliary outputs and intermediate states from the inference pathway.

    Attributes:
        memory_inf: Inference memory (M_inf) for sensory-to-place retrieval
        latent_prediction: Inferred abstract and grounded locations (g_inf, p_inf)
        filtered_observation: Temporally filtered sensory representation (x_f)
        retrieved_grounded: Optional grounded location retrieved directly from
                           memory using sensory input (p_x), if dual memory

    Theory:
        The inference pathway produces intermediate representations that are
        used for temporal integration and memory-based inference. The filtered
        observation maintains temporal context, while retrieved grounded
        locations enable direct sensory-to-place mapping.

        The latent_prediction contains both abstract (g_inf) and grounded (p_inf)
        locations from the inference process, which are used for memory updates
        and consistency losses.

        Memory is split: InferenceState stores M_inf (x→p pathway),
        GenerativeState stores M_gen (g→p pathway). This maintains proper
        isolation between inference and generative pathways.
    """

    memory_inf: Optional[Matrix]  # To be updated by Hebbian plasticity after inference step
    latent_prediction: LocationInference
    filtered_observation: MultiScaleCode
    retrieved_grounded: Optional[GroundedLocation]


class InferenceModel(nn.Module):
    """Inference TEM model"""

    def __init__(self, params: Parameters, projection: ProjectionHead, attractor: AttractorDynamics, decoder: ObservationDecoder):
        super().__init__()
        self.config = params  # Store configuration

        # Compute configuration-derived matrices
        two_hot_table = utils.create_two_hot_table(params.n_x, params.n_x_c)
        W_tile = utils.create_W_tile(params.n_g_subsampled_combined, params.n_x_f)

        # Initialize sub-modules (encoder, processor, tiling, projection, attractor, grounded)
        self.encoder = sensory.SensoryEncoder(params, two_hot_table)  # Sensory encoder module
        self.processor = sensory.SensoryProcessor(params)  # Temporal processor module
        self.tiling = sensory.SensoryProjection(params, W_tile)  # Sensory tilling module
        self.attractor = attractor  # Attractor dynamics for memory retrieval
        self.abstract = abstract.AbstractLocInference(params, projection, decoder)  # Abstract location inference module
        self.projection = projection  # Projection module for g to p (downsample + expand)
        self.grounded = grounded.GroundedLocInference(params)  # Grounded location inference (element-wise product)

    def forward(self, x: SensoryObservation, locations: List[Dict[str, Any]], state: InferenceState, g_gen: TransitionParams) -> InferenceState:
        """Run the inference path to obtain abstract and grounded locations.

        The inference path compresses and temporally filters sensory input,
        retrieves grounded locations from memory, infers abstract locations,
        and prepares representations for memory interaction.

        Parameters
        ----------
        x:
            Current sensory observations.
        locations:
            Environment descriptors required for handling special cases (e.g.
            shiny objects, environment-specific masks).
        state:
            Previous inference state, including filtered observations and
        g_gen:
            Abstract location statistics obtained from the transition model
            (e.g. mean and standard deviation from path integration).

        Returns
        -------
        InferenceState:
            New inference state with updated latent predictions, filtered
            observation, and optionally retrieved grounded locations

        Theory:
            The inference pathway combines bottom-up sensory processing with
            top-down predictions to infer both abstract (grid-like) and
            grounded (place-like) representations. Precision weighting balances
            path integration and sensory evidence.
        """
        if state.memory_inf is None:
            raise ValueError("InferenceModel.forward requires state.memory_inf to be set.")

        # Inference pathway steps (see publication.Inference architecture for details)
        x_c = self.encoder(x)  # 1. Compress sensory observation: x → x_c (one-hot to two-hot)
        x_f = self.processor(x_c, state.filtered_observation)  # 2. Temporally filter sensorium: x_c → x_f
        x_ = self.tiling(x_f)  # 3. Sensory input to hippocampus: x_f → x_ (prepare for memory indexing)
        p_x = self.attractor(x_, state.memory_inf, for_inference=True) if self.config.use_p_inf else None  # 4. Retrieve memory
        g = self.abstract(p_x, g_gen, x, locations)  # 5. Infer entorhinal (abstract location)
        g_ = self.projection(g)  # 6. Entorhinal input to hippocampus: g → g_ (project to grounded space)
        p = self.grounded(g_, x_)  # 7. Infer hippocampus (grounded location)

        # Return updated inference state with new latent predictions and filtered observation
        prediction = LocationInference(abstract=g, grounded=p)
        return InferenceState(latent_prediction=prediction, filtered_observation=x_f, retrieved_grounded=p_x)

    def init_state(self, batch_size: int, device: torch.device) -> InferenceState:
        """Initialize inference state with zeros and default values.

        Parameters
        ----------
        batch_size:
            Number of samples in the batch.
        device:
            Device to place the tensors on.

        Returns
        -------
        InferenceState
            Initialized inference state with zeroed latent predictions
            and filtered observations.

        Theory:
            The inference state is initialized to provide a starting point
            for temporal filtering and memory retrieval. Latent predictions
            are set to zero, indicating no prior knowledge, while filtered
            observations are also zeroed to avoid biasing the initial state.
        """

        # Initialize filtered observation x_f with zeros
        x_f = [torch.zeros((batch_size, self.config.n_x_f[f]), dtype=torch.float, device=device) for f in range(self.config.n_f)]
        # Initialize abstract location g_inf with g_init (learned prior)
        g = [self.abstract.g_init[f].unsqueeze(0).expand(batch_size, -1).to(device) for f in range(self.config.n_f)]
        # Initialize grounded location p_inf with zeros
        p = [torch.zeros((batch_size, self.config.n_p[f]), dtype=torch.float, device=device) for f in range(self.config.n_f)]

        # Return initialized inference state with zeroed latent predictions and filtered observation
        prediction = LocationInference(abstract=g, grounded=p)
        return InferenceState(latent_prediction=prediction, filtered_observation=x_f)
