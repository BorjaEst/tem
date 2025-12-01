from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn

from .. import utils
from ..core.projection import ProjectionHead
from ..memory.attractor import AttractorDynamics
from ..types import (
    AbstractLocation,
    GroundedLocation,
    LatentPrediction,
    Matrix,
    MultiScaleCode,
    TEMState,
    TransitionParams,
)
from . import abstract, grounded, precission, sensory
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
    latent_prediction: LatentPrediction
    filtered_observation: MultiScaleCode
    retrieved_grounded: Optional[GroundedLocation]


class InferenceModel(nn.Module):
    """Inference TEM model"""

    def __init__(self, params: Parameters, projection: ProjectionHead, attractor: AttractorDynamics):
        super().__init__()
        # Compute configuration-derived matrices
        two_hot_table = utils.create_two_hot_table(params.n_x, params.n_x_c)
        W_repeat = utils.create_W_repeat(params.n_g_subsampled_combined, params.n_x_f)
        W_tile = utils.create_W_tile(params.n_g_subsampled_combined, params.n_x_f)

        # Initialize sub-modules (encoder, processor, tiling, grounded, abstract, projection, attractor)
        self.encoder = sensory.SensoryEncoder(params, two_hot_table)  # Sensory encoder module
        self.processor = sensory.SensoryProcessor(params)  # Temporal processor module
        self.tiling = sensory.SensoryProjection(params, W_tile)  # Sensory tilling module
        self.grounded = grounded.GroundedLocInference(params, W_repeat, W_tile)  # Grounded location inference module
        self.abstract = abstract.AbstractLocInference(params)  # Abstract location inference module
        self.projection = projection  # Projection module for g to p
        self.attractor = attractor  # Attractor dynamics for memory retrieval

    def inference(self, x: Tensor, locations: List[Dict[str, Any]], state: InferenceState, g_gen: TransitionParams) -> InferenceState:
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

        # 1. Encode: x → x_c (one-hot to two-hot)
        x_c = self.encoder(x)

        # 2. Filter: x_c → x_f (temporal filtering)
        x_f = self.processor(x_c, state.filtered_observation)

        # 3. Tile: x_f → x_ (prepare for memory indexing)
        x_ = self.tiling(x_f)

        # 4. Retrieve from memory (if using inference memory)
        p_x = None
        p_x_downsampled = None
        if self.config.inference.use_p_inf:
            x_flat = utils.concatenate_frequencies(x_)
            M_inf = self.storage.get_memory(for_inference=True)
            p_x_flat = self.attractor.retrieve(x_flat, M_inf, for_inference=True)
            p_x = utils.split_to_frequencies(p_x_flat, self.config.architecture.n_p)

            # Downsample p_x to n_g_subsampled_combined by summing over sensory preferences
            # This projects from (n_g_subsampled_combined * n_x_f) to n_g_subsampled_combined
            p_x_downsampled = [torch.matmul(p_x[f], torch.t(self.W_repeat[f])) for f in range(self.config.architecture.n_f)]

        # 5. Infer abstract location (precision-weighted fusion)
        g_gen_mu, sigma_gen = g_gen

        # Handle shiny signals if present
        shiny_signals = None
        # TODO: Implement shiny object processing when needed
        # shiny_envs = [loc.get('shiny') is not None for loc in locations]
        # if any(shiny_envs): ...

        g = self.abstract(
            g_gen_mu,
            sigma_gen,
            p_x_downsampled,
            shiny_signals=shiny_signals,
            p2g_scale_offset=self.config.inference.p2g_offset if hasattr(self.config.inference, "p2g_offset") else 0.0,
        )

        # 6. Downsample and normalize g for inference
        g_ = self.projection.downsample(self.projection.normalize_g(g))

        # 7. Infer grounded location: g ⊗ x (outer product)
        p = self.grounded(g_, x_f)

        return InferenceState(
            latent_prediction=LatentPrediction(abstract=g, grounded=p),
            filtered_observation=x_f,
            retrieved_grounded=p_x,
        )

    def inf_g(
        self,
        p_x: Optional[GroundedLocation],
        g_gen: TransitionParams,
        x: Tensor,
        locations: List[Dict[str, Any]],
    ) -> AbstractLocation:
        """Infer abstract locations from memory retrieval and path integration.

        Parameters
        ----------
        p_x:
            Grounded locations retrieved from memory using sensory input.
        g_gen:
            Abstract location statistics from the transition model
            (path-integration prior).
        x:
            Current sensory observations, used e.g. for estimating memory
            quality.
        locations:
            Environment descriptors, including shiny object metadata.

        Returns
        -------
        AbstractLocation
            Inferred abstract locations (per frequency module), optionally
            including object-vector contributions for shiny environments.
        """
        # Delegate to AbstractLocInference
        g_gen_mu, sigma_gen = g_gen

        # Handle shiny signals if present
        shiny_signals = None
        # TODO: Implement shiny object processing when needed

        g = self.abstract(g_gen_mu, sigma_gen, p_x, shiny_signals, p2g_scale_offset=self.config.inference.p2g_offset if hasattr(self.config.inference, "p2g_offset") else 0.0)

        return g

    def inf_p(self, x_: MultiScaleCode, g_: MultiScaleCode) -> GroundedLocation:
        """Infer grounded locations from filtered sensory input and abstract codes.

        Parameters
        ----------
        x_:
            Sensory features prepared for memory interaction (e.g. weighted
            and tiled representations).
        g_:
            Downsampled and repeated abstract location codes aligned with the
            sensory tiling.

        Returns
        -------
        GroundedLocation
            Inferred grounded locations per frequency module, typically after
            applying a sparsity-inducing nonlinearity.
        """
        # Delegate to GroundedLocInference (outer product g ⊗ x)
        p = self.grounded(g_, x_)
        return p

    def x_prev2x(self, x_prev: MultiScaleCode, x_c: Tensor) -> MultiScaleCode:
        """Temporally filter sensory observations across time steps.

        Parameters
        ----------
        x_prev:
            Previous temporally filtered sensory representations.
        x_c:
            Current compressed sensory observation.

        Returns
        -------
        MultiScaleCode
            Updated filtered sensory representations for each frequency
            module, using a learned exponential smoothing factor.
        """
        return self.processor.filter_temporal(x_c, x_prev)

    def x2x_(self, x: MultiScaleCode) -> MultiScaleCode:
        """Prepare sensory input for Hebbian memory interaction.

        This includes normalisation and re-weighting for each frequency
        module, followed by tiling into the shape required for outer-product
        interactions with abstract codes.

        Parameters
        ----------
        x:
            Temporally filtered sensory representations.

        Returns
        -------
        MultiScaleCode
            Memory-ready sensory representations ``x_`` per frequency module.
        """
        return self.processor.normalize(x)

    def g2g_(self, g: AbstractLocation) -> MultiScaleCode:
        """Prepare abstract locations for Hebbian memory interaction.

        Parameters
        ----------
        g:
            Abstract location codes per frequency module.

        Returns
        -------
        MultiScaleCode
            Downsampled and repeated abstract codes ``g_`` compatible with the
            sensory tiling used in the Hebbian memory.
        """
        g_normalized = self.projection.normalize_g(g)
        return self.projection.downsample(g_normalized)
