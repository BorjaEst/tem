from typing import Any, Dict, List, Optional, Protocol

from torch import Tensor, nn

from .. import utils
from ..core.projection import ProjectionHead
from ..memory.attractor import AttractorDynamics
from . import abstract, grounded, precission, sensory, tiling
from .abstract import AbstractLocParams
from .grounded import GroundedLocParams
from .sensory import EncoderParams, ProcessorParams, ProjectionParams


class Parameters(EncoderParams, ProcessorParams, ProjectionParams, GroundedLocParams, AbstractLocParams):
    """Parameters needed by InferenceModel."""

    n_f_g: int
    n_f_ovc: int


class InferenceModel(nn.Module):
    """Inference TEM model"""

    def __init__(self, params: Parameters, projection: ProjectionHead, attractor: AttractorDynamics):
        super().__init__()
        # Compute configuration-derived matrices
        two_hot_table = utils.create_two_hot_table(params.n_x, params.n_x_c)
        W_repeat = utils.create_W_repeat(params.n_g_subsampled_combined, params.n_x_f)
        W_tile = utils.create_W_tile(params.n_g_subsampled_combined, params.n_x_f)
        g_downsample = utils.create_g_downsample(params.n_g, params.n_g_subsampled_combined)
        g_connections = utils.create_g_connections(params.n_f, params.n_f_g, params.n_f_ovc, params.f_initial_extended)
        p_update_mask = utils.create_p_update_mask(params.n_p, params.n_f, params.n_f_g, params.n_f_ovc, params.f_initial_extended)

        # Initialize sub-modules (encoder, processor, tiling, grounded, abstract, projection, attractor)
        self.encoder = sensory.SensoryEncoder(params, two_hot_table)  # Sensory encoder module
        self.processor = sensory.SensoryProcessor(params)  # Temporal processor module
        self.tiling = sensory.SensoryProjection(params, W_tile)  # Sensory tilling module
        self.grounded = grounded.GroundedLocInference(params, W_repeat, W_tile)  # Grounded location inference module
        self.abstract = abstract.AbstractLocInference(params)  # Abstract location inference module
        self.projection = projection  # Projection module for g to p
        self.attractor = attractor  # Attractor dynamics for memory retrieval

    def inference(self, x, locations, M_prev, x_prev, g_gen):
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
            shiny objects, environment‑specific masks).
        M_prev:
            Previous memory state for the inference path.
        x_prev:
            Previous temporally filtered sensory representation.
        g_gen:
            Abstract location statistics obtained from the transition model
            (e.g. mean and standard deviation from path integration).

        Returns
        -------
        tuple
            ``(x_f, g, p_x, p)`` where ``x_f`` is filtered sensory input,
            ``g`` is the inferred abstract location, ``p_x`` is the grounded
            location retrieved from memory using sensory input, and ``p`` is
            the grounded location inferred from ``x_f`` and ``g``.
        """
        # 1. Encode: x → x_c (one-hot to two-hot)
        x_c = self.encoder(x)

        # 2. Filter: x_c → x_f (temporal filtering)
        x_f = self.processor(x_c, x_prev)

        # 3. Tile: x_f → x_ (prepare for memory indexing)
        x_ = self.tiling(x_f)

        # 4. Retrieve from memory (if using inference memory)
        p_x = None
        p_x_downsampled = None
        if self.config.inference.use_p_inf:
            x_flat = concatenate_frequencies(x_)
            M_inf = self.storage.get_memory(for_inference=True)
            p_x_flat = self.attractor.retrieve(x_flat, M_inf, for_inference=True)
            p_x = split_to_frequencies(p_x_flat, self.config.architecture.n_p)

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

        return x_f, g, p_x, p

    def inf_g(self, p_x, g_gen, x, locations):
        """Infer abstract locations from memory retrieval and path integration.

        Parameters
        ----------
        p_x:
            Grounded locations retrieved from memory using sensory input.
        g_gen:
            Abstract location statistics from the transition model
            (path‑integration prior).
        x:
            Current sensory observations, used e.g. for estimating memory
            quality.
        locations:
            Environment descriptors, including shiny object metadata.

        Returns
        -------
        Any
            Inferred abstract locations (per frequency module), optionally
            including object‑vector contributions for shiny environments.
        """
        # Delegate to AbstractLocInference
        g_gen_mu, sigma_gen = g_gen

        # Handle shiny signals if present
        shiny_signals = None
        # TODO: Implement shiny object processing when needed

        g = self.abstract(g_gen_mu, sigma_gen, p_x, shiny_signals, p2g_scale_offset=self.config.inference.p2g_offset if hasattr(self.config.inference, "p2g_offset") else 0.0)

        return g

    def inf_p(self, x_, g_):
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
        Any
            Inferred grounded locations per frequency module, typically after
            applying a sparsity‑inducing nonlinearity.
        """
        # Delegate to GroundedLocInference (outer product g ⊗ x)
        p = self.grounded(g_, x_)
        return p
