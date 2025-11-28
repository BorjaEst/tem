from typing import Any, Dict, List, Optional, Protocol

import torch
from torch import Tensor, nn

from .. import utils
from ..core import ProjectionHead
from ..memory import AttractorDynamics
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


class GenerativeModel(nn.Module):
    """Generative TEM model"""

    def __init__(self, params: Parameters, projection: ProjectionHead, attractor: AttractorDynamics):
        super().__init__()
        # Compute configuration-derived matrices
        g_connections = utils.create_g_connections(params.n_f, params.n_f_g, params.n_f_ovc, params.f_extended)
        W_repeat = utils.matrices.create_W_repeat(params.n_g_subsampled_combined, [params.n_x_c] * params.n_f)

        # Initialize sub-modules (transition, projection, decoder, attractor)
        self.transition = transition.TransitionModel(params, g_connections)  # Transition dynamics module
        self.decoder = observation.ObservationDecoder(params)  # Observation decoder module
        self.location = location.LocationGenerator(params, attractor, W_repeat)  # Location generator module (stateless)
        self.projection = projection  # Projection module for g to p
        self.attractor = attractor  # Attractor dynamics for memory retrieval

    def generative(self, M_prev, p_inf, g_inf, g_gen):
        """Run the generative path to reconstruct observations and locations.

        Using the inferred abstract and grounded locations, and the previous
        memory state, this method predicts grounded locations and sensory
        observations via the generative model.

        Parameters
        ----------
        M_prev:
            Previous Hebbian memory state for the generative path.
        p_inf:
            Inferred grounded locations (hippocampal code) per frequency
            module.
        g_inf:
            Inferred abstract locations (entorhinal‑like code) per module.
        g_gen:
            Abstract location statistics from the transition model used for
            generative prediction.

        Returns
        -------
        tuple
            ``(x_gen, x_logits, p_gen)`` where ``x_gen`` are generated
            observations from different generative routes, ``x_logits`` are
            their pre‑softmax logits, and ``p_gen`` are grounded locations
            retrieved from memory.
        """
        # Route 1: Direct from p_inf → x_p
        x_p, x_p_logits = self.decoder(p_inf)

        # Route 2: g_inf → memory → p → x_g
        p_g_inf = self.gen_p(g_inf, M_prev[0])
        x_g, x_g_logits = self.gen_x(p_g_inf)

        # Route 3: g_gen → memory → p → x_gt
        p_g_gen = self.gen_p(g_gen, M_prev[0])
        x_gt, x_gt_logits = self.gen_x(p_g_gen)

        # Package outputs
        x_gen = (x_p, x_g, x_gt)
        x_logits = (x_p_logits, x_g_logits, x_gt_logits)

        return x_gen, x_logits, p_g_inf

    def gen_g(self, a_prev, g_prev, locations):
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
        Any
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

    def gen_p(self, g, M_prev):
        """Retrieve grounded locations from memory using abstract codes.

        Parameters
        ----------
        g:
            Abstract location codes for each frequency module.
        M_prev:
            Hebbian memory connectivity used for attractor‑based retrieval.

        Returns
        -------
        Any
            Grounded locations ``p`` obtained by pattern completion in the
            attractor network.
        """
        # Normalize and downsample g for memory indexing
        g_ = self.projection.downsample(self.projection.normalize_g(g))

        # Delegate to LocationGenerator with explicit memory state (required)
        p = self.location.generate(g_, M_prev, for_inference=False)

        return p

    def gen_x(self, p):
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
            one‑hot approximation over observations, and ``logits`` are the
            corresponding pre‑softmax scores.
        """
        # Decode observation from grounded location (decoder uses highest frequency)
        x_probs, x_logits = self.decoder(p)

        # Return probabilities and logits
        return x_probs, x_logits
