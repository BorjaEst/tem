from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import torch
from torch import Tensor, nn

from .. import utils
from ..core import ProjectionHead
from ..memory import AttractorDynamics
from ..types import AbstractLocation, GroundedLocation, LocationInference, Matrix, SensoryPrediction, TransitionOutput
from . import location, observation, transition
from .location import LocationGeneratorParams
from .observation import DecoderParams
from .transition import TransitionParams


class Parameters(TransitionParams, DecoderParams, LocationGeneratorParams):
    """Model parameters needed by Model."""

    n_f: int
    n_f_g: int
    n_f_ovc: int

    @property
    def f_extended(self) -> List[bool]:
        """List indicating which frequency modules are extended."""
        ...

    @property
    def n_g_subsampled_combined(self) -> List[int]:
        """Grid + OVC subsampled cell counts per module."""
        ...


@dataclass
class GenerativeState:
    """Auxiliary outputs and intermediate states from the generative pathway.

    Attributes:
        memory_gen: Generative memory (M_gen) for abstract-to-place mapping
        g_gen: Generated abstract location from transition (path integration)
        x_p: Sensory prediction from p_inf (Route 1: p_inf → x_p)
        x_g: Sensory prediction from g_inf via memory (Route 2: g_inf → p → x_g)
        x_gen: Primary sensory prediction from g_gen via memory (Route 3: g_gen → p → x_gen)
        p_g: Intermediate grounded location from g_inf via memory
        p_gen: Intermediate grounded location from g_gen via memory

    Theory:
        The generative model produces multiple sensory predictions through
        different pathways. These are used to compute consistency losses
        between inference and generation, ensuring coherent representations.
        Field names match the mathematical notation in TEM theory for clarity.

        The g_gen represents the predicted abstract location from path integration,
        which is compared against g_inf for consistency. It is stored for
        observability and loss computation, but the next iteration uses g_inf
        (belief) as the recurrent state.

        Memory is split: GenerativeState stores M_gen (g→p pathway),
        InferenceState stores M_inf (x→p pathway). This maintains proper
        isolation between inference and generative pathways.
    """

    memory_gen: Optional[Matrix]  # Updated by Hebbian plasticity after generative step
    g_gen: AbstractLocation
    x_p: SensoryPrediction
    x_g: SensoryPrediction
    x_gen: SensoryPrediction
    p_g: GroundedLocation
    p_gen: GroundedLocation


class GenerativeModel(nn.Module):
    """Generative TEM model"""

    def __init__(self, params: Parameters, projection: ProjectionHead, attractor: AttractorDynamics):
        super().__init__()
        self.config = params  # Store configuration

        # Compute configuration-derived matrices
        g_connections = utils.create_g_connections(params.n_f, params.n_f_g, params.n_f_ovc, params.f_extended)
        W_repeat = utils.matrices.create_W_repeat(params.n_g_subsampled_combined, [params.n_x_c] * params.n_f)

        # Initialize sub-modules (transition, projection, decoder, attractor)
        self.transition = transition.TransitionModel(params, g_connections)  # Transition dynamics module
        self.decoder = observation.ObservationDecoder(params)  # Observation decoder module
        self.location = location.LocationGenerator(params, attractor, W_repeat)  # Location generator module (stateless)
        self.projection = projection  # Projection module for g to p
        self.attractor = attractor  # Attractor dynamics for memory retrieval

    def init_state(self, batch_size: int, device: torch.device) -> GenerativeState:
        """Initialize generative state with zeros and default values.

        Parameters
        ----------
        batch_size:
            Number of samples in the batch.
        device:
            Device to place the tensors on.

        Returns
        -------
        GenerativeState
            Initialized generative state with zeroed tensors.
        """

        return GenerativeState(
            memory_gen=None,  # Initialize memory_gen to None (will be set externally if used)
            # Initialize g_gen to zeros
            g_gen=[torch.zeros((batch_size, self.config.n_g[f]), dtype=torch.float, device=device) for f in range(self.config.n_f)],
            # Initialize sensory predictions to zeros
            x_p=SensoryPrediction(
                values=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
                logits=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
            ),
            # Initialize sensory predictions to zeros
            x_g=SensoryPrediction(
                values=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
                logits=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
            ),
            # Initialize sensory predictions to zeros
            x_gen=SensoryPrediction(
                values=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
                logits=torch.zeros((batch_size, self.config.n_x_c), dtype=torch.float, device=device),
            ),
            # Initialize grounded locations to zeros
            p_g=[torch.zeros((batch_size, self.config.n_p[f]), dtype=torch.float, device=device) for f in range(self.config.n_f)],
            p_gen=[torch.zeros((batch_size, self.config.n_p[f]), dtype=torch.float, device=device) for f in range(self.config.n_f)],
        )

    def generative(self, latent: LocationInference, g_gen: AbstractLocation, state: GenerativeState) -> GenerativeState:
        """Run the generative path to reconstruct observations and locations.

        Using the inferred abstract and grounded locations, and the previous
        memory state, this method predicts grounded locations and sensory
        observations via the generative model.

        Parameters
        ----------
        latent:
            Inferred latent locations containing abstract (g_inf) and
            grounded (p_inf) representations from the inference pathway.
        g_gen:
            Abstract location from the transition model used for
            generative prediction.
        state:
            Previous generative state containing memory_gen and other
            auxiliary outputs from the previous timestep.

        Returns
        -------
        GenerativeState:
            New generative state with updated sensory predictions and
            intermediate grounded locations. Note: memory_gen is copied
            from state and will be updated later by Hebbian plasticity.

        Theory:
            The generative model produces multiple predictions:
            - Route 1: p_inf → x_p (direct from inferred grounded location)
            - Route 2: g_inf → p → x_g (from inferred abstract location)
            - Route 3: g_gen → p → x_gt (from generated abstract location)
            These enable consistency losses between inference and generation.
        """

        # Route 1: Direct from p_inf → x_p
        x_p, x_p_logits = self.decoder(latent.grounded)

        # Route 2: g_inf → memory → p → x_g
        p_g_inf = self.gen_p(latent.abstract, state.memory_gen)
        x_g, x_g_logits = self.gen_x(p_g_inf)

        # Route 3: g_gen → memory → p → x_gt (primary generative pathway)
        p_g_gen = self.gen_p(g_gen, state.memory_gen)
        x_gt, x_gt_logits = self.gen_x(p_g_gen)

        # Package auxiliary state with theory-aligned names
        return GenerativeState(
            memory_gen=state.memory_gen,  # Copy from previous state, will be updated by Hebbian plasticity
            g_gen=g_gen,
            x_p=SensoryPrediction(values=x_p, logits=x_p_logits),
            x_g=SensoryPrediction(values=x_g, logits=x_g_logits),
            x_gen=SensoryPrediction(values=x_gt, logits=x_gt_logits),
            p_g=p_g_inf,
            p_gen=p_g_gen,
        )

    def gen_g(self, a_prev: List[Optional[int]], g_prev: AbstractLocation, locations: List[Dict[str, Any]]) -> TransitionOutput:
        """Generate abstract location codes via transition dynamics.

        Parameters
        ----------
        a_prev:
            Previous actions for each walk in the batch (list or tensor).
        g_prev:
            Previous abstract locations for all frequency modules.
        locations:
            Environment descriptors, used to handle special transition rules
            (such as ignoring action direction in shiny environments).

        Returns
        -------
        TransitionOutput
            Generated abstract locations and associated uncertainty
            statistics, suitable for use by :meth:`inference` and
            :meth:`generative`.
        """
        # Convert actions to tensor if needed
        if isinstance(a_prev, list):
            # Filter out None values and convert to tensor
            a_valid = [a if a is not None else 0 for a in a_prev]
            a_tensor = torch.tensor(a_valid, dtype=torch.long, device=g_prev[0].device)
        else:
            a_tensor = a_prev

        # Check for shiny environments (no directional transitions)
        shiny_envs = [loc.get("shiny") is not None if isinstance(loc, dict) else False for loc in locations]

        # Compute transition with action
        g, sigma_g = self.transition(g_prev, a_tensor, use_action=True)

        # For shiny environments, recompute without action direction
        if any(shiny_envs):
            g_gen, _ = self.transition(g_prev, a_tensor, use_action=False)
        else:
            g_gen = g

        return g_gen, (g, sigma_g)

    def gen_p(self, g: AbstractLocation, M_prev: Matrix) -> GroundedLocation:
        """Retrieve grounded locations from memory using abstract codes.

        Parameters
        ----------
        g:
            Abstract location codes for each frequency module.
        M_prev:
            Hebbian memory connectivity used for attractor-based retrieval.

        Returns
        -------
        GroundedLocation
            Grounded locations ``p`` obtained by pattern completion in the
            attractor network.
        """
        # Normalize and downsample g for memory indexing
        g_ = self.projection.downsample(self.projection.normalize_g(g))

        # Delegate to LocationGenerator with explicit memory state (required)
        p = self.location.generate(g_, M_prev, for_inference=False)

        return p

    def gen_x(self, p: GroundedLocation) -> SensoryPrediction:
        """Generate observations from grounded locations.

        Parameters
        ----------
        p:
            Grounded location codes (possibly across multiple frequency
            modules) from which observations should be decoded.

        Returns
        -------
        tuple
            ``(x, logits)`` where ``x`` is a categorical distribution or
            one-hot approximation over observations, and ``logits`` are the
            corresponding pre-softmax scores.
        """
        # Decode observation from grounded location (decoder uses highest frequency)
        x_probs, x_logits = self.decoder(p)

        # Return probabilities and logits
        return x_probs, x_logits
