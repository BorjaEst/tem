"""PyTorch implementation of the Tolman–Eichenbaum Machine (TEM).

This module defines the high‑level `TEMModel` for a modern, modular
TEM implementation. It mirrors the functionality of the original
`tem.model.Model` class, but is structured around typed configuration objects
and sub‑modules for sensory encoding, inference, memory, and projection.

Implementation follows the reference model.py while using modular torch_tem
components for maintainability and testability.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem import data, figures, utils
from torch_tem.config import EnvironmentConfig, InferenceConfig, ModelConfig
from torch_tem.core.decoder import ObservationDecoder
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.core.projection import ProjectionHead
from torch_tem.core.tiling import SensoryProjection
from torch_tem.core.transition import TransitionModel
from torch_tem.inference.abstract import AbstractLocationInference
from torch_tem.inference.grounded import GroundedLocationInference
from torch_tem.inference.sensory import SensoryProcessor
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage
from torch_tem.utils.masks import (
    create_g_connections,
    create_p_retrieve_masks,
    create_p_update_mask,
)
from torch_tem.utils.matrices import (
    concatenate_frequencies,
    create_g_downsample,
    create_two_hot_table,
    create_W_repeat,
    create_W_tile,
    split_to_frequencies,
)


class TEMModel(nn.Module):
    """Top‑level Tolman–Eichenbaum Machine model.

    This class is the main entry point for running TEM in PyTorch. It combines
    sensory encoding, abstract and grounded location inference, generative
    decoding, and Hebbian memory dynamics. Conceptually, it corresponds to the
    original ``Model`` class in ``model.py``, but is designed to work with
    structured configuration objects and reusable sub‑modules under
    ``torch_tem.*``.

    The methods defined here are intentionally left unimplemented; they
    document the public API and expected behavior so that the implementation
    can be ported from the reference code in a controlled, testable way.
    """

    def __init__(self, params: ModelConfig):
        """Initialise a new TEM model from configuration parameters.

        Parameters
        ----------
        params:
            High‑level model configuration specifying architectural choices
            (numbers of cells, frequency modules, connectivity patterns),
            environment and inference options, and training hyper‑parameters.

        Notes
        -----
        The original implementation accepts a plain parameter dictionary
        (see ``model.Model.__init__``). This method uses the typed
        ``ModelConfig`` dataclass while constructing and registering all
        trainable sub‑modules (encoders, projections, memory, etc.).
        """
        super().__init__()
        self.config = params

        # Extract common config references
        arch = params.architecture
        inf = params.inference

        # Compute configuration-derived matrices
        self.two_hot_table = create_two_hot_table(arch.n_x, arch.n_x_c)
        self.W_repeat = create_W_repeat(arch.n_g_subsampled_combined, arch.n_x_f)
        self.W_tile = create_W_tile(arch.n_g_subsampled_combined, arch.n_x_f)
        self.g_downsample = create_g_downsample(arch.n_g, arch.n_g_subsampled_combined)
        self.g_connections = create_g_connections(arch.n_f, arch.n_f_g, arch.n_f_ovc, arch.f_initial_extended)
        self.p_update_mask = create_p_update_mask(arch.n_p, arch.n_f, arch.n_f_g, arch.n_f_ovc, arch.f_initial_extended)

        # Compute retrieval masks for attractor dynamics
        # max_freq_inf and max_freq_gen control hierarchical early-stopping
        max_freq_inf = [min(f + 1, arch.i_attractor) for f in range(arch.n_f)]
        max_freq_gen = [min(f + 1, arch.i_attractor) for f in range(arch.n_f)]
        self.p_retrieve_mask_inf, self.p_retrieve_mask_gen = create_p_retrieve_masks(arch.n_p, arch.i_attractor, max_freq_inf, max_freq_gen)

        # Instantiate all components
        self.encoder = SensoryEncoder(arch, self.two_hot_table)
        self.processor = SensoryProcessor(arch)
        self.transition = TransitionModel(arch, self.g_connections)
        self.tiling = SensoryProjection(arch, self.W_tile)
        self.grounded = GroundedLocationInference(arch, self.W_repeat, self.W_tile)
        self.abstract = AbstractLocationInference(arch, inf)
        self.storage = MemoryStorage(arch, inf, self.p_update_mask)
        self.attractor = AttractorDynamics(arch, inf, self.p_retrieve_mask_inf, self.p_retrieve_mask_gen)
        self.projection = ProjectionHead(arch, self.g_downsample)
        self.decoder = ObservationDecoder(arch)

    def forward(self, walk, prev_iter=None, prev_M=None):
        """Run the TEM model on a sequence of observations and actions.

        Parameters
        ----------
        walk:
            Iterable of triples ``(locations, x, a)`` describing a batched
            random walk through the environment: current abstract/ground
            locations, sensory observations, and discrete actions.
        prev_iter:
            Optional iteration object from a previous call, used to continue
            partially processed walks (e.g. across episode boundaries).
        prev_M:
            Optional previous Hebbian memory state, allowing memory to be
            carried across calls.

        Returns
        -------
        steps:
            A sequence of per‑step iteration objects containing inferred and
            generated variables (abstract and grounded locations, observations,
            memory states, and losses), analogous to ``Iteration`` instances
            in the reference implementation.
        """
        # Initialize walks (reset new episodes)
        steps = self.init_walks(prev_iter)

        # Process each timestep in walk
        for locations, x, a in walk:
            # Initialize if first step
            if steps is None:
                steps = [self.init_iteration(locations, x, [None] * len(a), prev_M)]

            # Perform single TEM iteration
            L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = self.iteration(x, locations, steps[-1].a, steps[-1].M, steps[-1].x_inf, steps[-1].g_inf)

            # Store iteration results
            steps.append(Iteration(locations, x, a, L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf))

        # Remove initialization step
        return steps[1:]

    def iteration(self, x, locations, a_prev, M_prev, x_prev, g_prev):
        """Perform a single TEM iteration for one time step.

        This method combines transition dynamics, inference, generative
        prediction, Hebbian memory update, and loss computation for a single
        step in a batched random walk.

        Parameters
        ----------
        x:
            Current sensory observations (typically one‑hot or compressed
            encodings) for each walk in the batch.
        locations:
            Per‑walk environment descriptors (e.g. metadata including shiny
            objects) used by the transition and inference models.
        a_prev:
            List or tensor of previous discrete actions for each walk.
        M_prev:
            Previous memory state(s) for the Hebbian attractor network.
        x_prev:
            Temporally filtered sensory observations from the previous step.
        g_prev:
            Previous abstract location codes for each frequency module.

        Returns
        -------
        tuple
            A tuple containing per‑step losses, updated memory, generated and
            inferred variables (matching the structure of the original
            ``Model.iteration`` return value).
        """
        # 1. Transition dynamics
        g_gen, (g_gen_mu, sigma_gen) = self.gen_g(a_prev, g_prev, locations)

        # 2. Inference path
        x_inf, g_inf, p_inf_x, p_inf = self.inference(x, locations, M_prev, x_prev, (g_gen_mu, sigma_gen))

        # 3. Generative path
        x_gen, x_logits, p_gen = self.generative(M_prev, p_inf, g_inf, g_gen)

        # 4. Memory update
        p_inf_flat = concatenate_frequencies(p_inf)
        p_gen_flat = concatenate_frequencies(p_gen)

        # Update generative memory
        self.storage.update(p_inf_flat, p_gen_flat, self.config.inference.eta_g, self.config.inference.eta_p)
        M = [self.storage.M_gen]

        # Update inference memory if using it
        if self.config.inference.use_p_inf:
            if self.storage.use_dual_memory:
                # Separate inference memory
                p_x_flat = concatenate_frequencies(p_inf_x) if p_inf_x is not None else p_inf_flat
                # For inference memory, update with p_inf ⊗ p_x (not p_gen)
                # We need a separate update that doesn't use hierarchical connections
                # For now, reuse the generative memory if common_memory is True
                M.append(self.storage.M_inf if self.storage.M_inf is not None else self.storage.M_gen)
            else:
                # Common memory
                M.append(M[0])

        # 5. Loss computation
        L = self.loss(g_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev)

        return L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf

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
        if self.config.inference.use_p_inf:
            x_flat = concatenate_frequencies(x_)
            M_inf = self.storage.get_memory(for_inference=True)
            p_x_flat = self.attractor.retrieve(x_flat, M_inf, for_inference=True)
            p_x = split_to_frequencies(p_x_flat, self.config.architecture.n_p)

        # 5. Infer abstract location (precision-weighted fusion)
        g_gen_mu, sigma_gen = g_gen

        # Handle shiny signals if present
        shiny_signals = None
        # TODO: Implement shiny object processing when needed
        # shiny_envs = [loc.get('shiny') is not None for loc in locations]
        # if any(shiny_envs): ...

        g = self.abstract(
            g_gen_mu, sigma_gen, p_x, shiny_signals=shiny_signals, p2g_scale_offset=self.config.inference.p2g_offset if hasattr(self.config.inference, "p2g_offset") else 0.0
        )

        # 6. Downsample and normalize g for inference
        g_ = self.projection.downsample(self.projection.normalize_g(g))

        # 7. Infer grounded location: g ⊗ x (outer product)
        p = self.grounded(g_, x_f)

        return x_f, g, p_x, p

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
        x_p, x_p_logits = self.gen_x(p_inf)

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

    def loss(self, g_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev):
        """Compute all TEM loss components for a single time step.

        The loss combines consistency terms between inferred and generated
        grounded locations, path‑integrated versus inferred abstract
        locations, reconstruction losses over observations, and regularisation
        penalties on abstract and grounded codes.

        Parameters
        ----------
        g_gen:
            Generated abstract locations from the transition model.
        p_gen:
            Grounded locations retrieved from the generative path.
        x_logits:
            Tuple of pre‑softmax logits for observations from different
            generative routes.
        x:
            Ground‑truth sensory observations.
        g_inf:
            Inferred abstract locations.
        p_inf:
            Inferred grounded locations.
        p_inf_x:
            Grounded locations retrieved from memory using sensory input
            alone.
        M_prev:
            Previous memory state, used for any memory‑related regularisers.

        Returns
        -------
        list
            List of loss tensors corresponding to the individual components
            (e.g. ``L_p_g``, ``L_p_x``, ``L_x_gen``, ``L_x_g``, ``L_x_p``,
            ``L_g``, ``L_reg_g``, ``L_reg_p``).
        """

        # Helper function for squared error across frequencies
        def squared_error_freq(value, target):
            if isinstance(value, list) and isinstance(target, list):
                return [torch.sum((v - t) ** 2, dim=1) * 0.5 for v, t in zip(value, target)]
            return torch.sum((value - target) ** 2, dim=1) * 0.5

        # L_p_g: ||p_inf - p_gen||²
        L_p_g = torch.sum(torch.stack(squared_error_freq(p_inf, p_gen), dim=0), dim=0)

        # L_p_x: ||p_inf - p_x||² (if using inference memory)
        if self.config.inference.use_p_inf and p_inf_x is not None:
            L_p_x = torch.sum(torch.stack(squared_error_freq(p_inf, p_inf_x), dim=0), dim=0)
        else:
            L_p_x = torch.zeros_like(L_p_g)

        # L_g: ||g_inf - g_gen||²
        L_g = torch.sum(torch.stack(squared_error_freq(g_inf, g_gen), dim=0), dim=0)

        # Cross-entropy losses for observation reconstruction
        labels = torch.argmax(x, 1)
        L_x_gen = torch.nn.functional.cross_entropy(x_logits[2], labels, reduction="none")  # x_gt (from g_gen)
        L_x_g = torch.nn.functional.cross_entropy(x_logits[1], labels, reduction="none")  # x_g (from g_inf)
        L_x_p = torch.nn.functional.cross_entropy(x_logits[0], labels, reduction="none")  # x_p (from p_inf)

        # Regularization: L2 on abstract location, L1 on grounded location
        L_reg_g = torch.sum(torch.stack([torch.sum(g**2, dim=1) for g in g_inf], dim=0), dim=0)
        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)

        return [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]

    def init_trainable(self):
        """Create and register all trainable parameters and sub‑modules.

        This method should allocate PyTorch parameters and small networks used
        across the model, such as frequency‑specific scale factors, MLPs for
        transition dynamics and uncertainties, and sensory compression /
        decompression layers. It plays the same role as ``init_trainable`` in
        the original implementation.
        """
        # All trainable components are already initialized in __init__
        # This method is kept for compatibility with the original API
        pass

    def init_walks(self, prev_iter):
        """Reset per‑walk state when new walks start in a batch.

        Parameters
        ----------
        prev_iter:
            Optional previous iteration object or list of iterations. Entries
            corresponding to newly started walks (e.g. where the previous
            action is ``None``) should have their memory and state reset.

        Returns
        -------
        Any
            Updated previous iteration structure with per‑walk state
            re‑initialised where required, ready to be used by
            :meth:`iteration`.
        """
        if prev_iter is not None:
            for a_i, a in enumerate(prev_iter[0].a):
                if a is None:
                    # Reset memory for this walk
                    for M in prev_iter[0].M:
                        M[a_i, :, :] = 0

                    # Reset abstract location to prior (from config or learned g_init)
                    for f, g_inf in enumerate(prev_iter[0].g_inf):
                        # Use learned g_init from abstract inference module if available
                        if hasattr(self.abstract, "g_init"):
                            g_inf[a_i, :] = self.abstract.g_init[f]
                        else:
                            g_inf[a_i, :] = torch.zeros(self.config.architecture.n_g[f])

                    # Reset filtered sensory to zeros
                    for f, x_inf in enumerate(prev_iter[0].x_inf):
                        x_inf[a_i, :] = torch.zeros(self.config.architecture.n_x_f[f])

        return prev_iter

    def init_iteration(self, locations, x, a_prev, M_prev):
        """Initialize the first iteration of a walk sequence.

        Parameters
        ----------
        locations:
            Environment descriptors for each walk in the batch.
        x:
            Initial sensory observations.
        a_prev:
            Previous actions (typically all None for first iteration).
        M_prev:
            Previous memory state.

        Returns
        -------
        List[Iteration]
            Initial iteration objects with priors and zero states.
        """
        batch_size = len(locations)
        n_freqs = len(self.config.architecture.n_g)

        # Initialize abstract locations with priors or zeros
        g_inf = []
        for f in range(n_freqs):
            if hasattr(self.abstract, "g_init"):
                g_init_f = self.abstract.g_init[f].unsqueeze(0).expand(batch_size, -1)
            else:
                g_init_f = torch.zeros(batch_size, self.config.architecture.n_g[f])
            g_inf.append(g_init_f)

        # Initialize filtered sensory as zeros
        x_inf = [torch.zeros(batch_size, n_x_f) for n_x_f in self.config.architecture.n_x_f]

        # Initialize memory if not provided
        if M_prev is None:
            M = self.storage.init_memory(batch_size)
        else:
            M = M_prev

        # Create iteration with initial state
        iter_0 = Iteration(
            g=g_inf,
            x=x_inf,
            a=a_prev,
            L=[torch.zeros(batch_size) for _ in range(8)],  # 8 loss components
            M=M,
            g_gen=None,
            p_gen=None,
            x_gen=None,
            x_logits=None,
            x_inf=x_inf,
            g_inf=g_inf,
            p_inf=None,
        )

        return [iter_0]

    def gen_g(self, a_prev, g_prev, locations):
        """Generate abstract location codes via transition dynamics.

        Parameters
        ----------
        a_prev:
            Previous actions for each walk in the batch.
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
        # Check for shiny environments (no directional transitions)
        shiny_envs = [loc.get("shiny") is not None if isinstance(loc, dict) else False for loc in locations]

        # Compute transition with action
        g, sigma_g = self.transition(g_prev, a_prev, use_action=True)

        # For shiny environments, recompute without action direction
        if any(shiny_envs):
            g_gen, _ = self.transition(g_prev, a_prev, use_action=False)
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

        # Retrieve from memory via attractor dynamics
        g_flat = concatenate_frequencies(g_)
        p_flat = self.attractor.retrieve(g_flat, M_prev, for_inference=False)

        # Convert back to per-frequency format
        p = split_to_frequencies(p_flat, self.config.architecture.n_p)

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
        # Delegate to AbstractLocationInference
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
        # Delegate to GroundedLocationInference (outer product g ⊗ x)
        p = self.grounded(g_, x_)
        return p

    def x_prev2x(self, x_prev, x_c):
        """Temporally filter sensory observations across time steps.

        Parameters
        ----------
        x_prev:
            Previous temporally filtered sensory representations.
        x_c:
            Current compressed sensory observation.

        Returns
        -------
        Any
            Updated filtered sensory representations for each frequency
            module, using a learned exponential smoothing factor.
        """
        return self.processor.filter_temporal(x_c, x_prev)

    def x2x_(self, x):
        """Prepare sensory input for Hebbian memory interaction.

        This includes normalisation and re‑weighting for each frequency
        module, followed by tiling into the shape required for outer‑product
        interactions with abstract codes.

        Parameters
        ----------
        x:
            Temporally filtered sensory representations.

        Returns
        -------
        Any
            Memory‑ready sensory representations ``x_`` per frequency module.
        """
        return self.processor.normalize(x)

    def g2g_(self, g):
        """Prepare abstract locations for Hebbian memory interaction.

        Parameters
        ----------
        g:
            Abstract location codes per frequency module.

        Returns
        -------
        Any
            Downsampled and repeated abstract codes ``g_`` compatible with the
            sensory tiling used in the Hebbian memory.
        """
        g_normalized = self.projection.normalize_g(g)
        return self.projection.downsample(g_normalized)

    def f_mu_g_path(self, a_prev, g_prev, no_direc=None):
        """Compute mean abstract locations after applying transition dynamics.

        Parameters
        ----------
        a_prev:
            Previous actions.
        g_prev:
            Previous abstract locations.
        no_direc:
            Optional boolean mask per walk indicating environments where
            directional information should be ignored (e.g. shiny
            environments in the generative model).

        Returns
        -------
        Any
            Mean abstract locations for each frequency module after applying
            the learned transition operator.
        """
        raise NotImplementedError("f_mu_g_path method is not implemented yet.")

    def f_sigma_g_path(self, a_prev, g_prev):
        """Compute uncertainty of abstract locations after transition.

        Parameters
        ----------
        a_prev:
            Previous actions.
        g_prev:
            Previous abstract locations (including prior‑initialised ones).

        Returns
        -------
        Any
            Standard deviation (or similar scale parameters) for abstract
            locations in each frequency module.
        """
        raise NotImplementedError("f_sigma_g_path method is not implemented yet.")

    def f_mu_g_mem(self, g_downsampled):
        """Compute mean abstract locations inferred purely from memory.

        Parameters
        ----------
        g_downsampled:
            Downsampled abstract locations obtained by summing grounded
            locations over sensory preferences.

        Returns
        -------
        Any
            Mean abstract locations inferred from memory quality for each
            frequency module.
        """
        raise NotImplementedError("f_mu_g_mem method is not implemented yet.")

    def f_sigma_g_mem(self, g_downsampled):
        """Compute uncertainty of abstract locations inferred from memory.

        Parameters
        ----------
        g_downsampled:
            Features describing the reliability of memory retrieval (e.g.
            norms and reconstruction errors).

        Returns
        -------
        Any
            Standard deviations for memory‑based abstract location estimates
            per frequency module.
        """
        raise NotImplementedError("f_sigma_g_mem method is not implemented yet.")

    def f_mu_g_shiny(self, shiny):
        """Compute abstract location means driven by shiny object presence.

        Parameters
        ----------
        shiny:
            Boolean or float indicators of shiny object presence per
            environment and location.

        Returns
        -------
        Any
            Object‑vector‑like abstract codes for the relevant frequency
            modules.
        """
        raise NotImplementedError("f_mu_g_shiny method is not implemented yet.")

    def f_sigma_g_shiny(self, shiny):
        """Compute uncertainty of shiny‑driven abstract location components.

        Parameters
        ----------
        shiny:
            Boolean or float indicators of shiny object presence.

        Returns
        -------
        Any
            Standard deviations associated with the shiny‑driven abstract
            codes.
        """
        raise NotImplementedError("f_sigma_g_shiny method is not implemented yet.")

    def f_sigma_p(self, p):
        """Compute uncertainty for grounded locations retrieved from memory.

        Parameters
        ----------
        p:
            Grounded location activations before sampling.

        Returns
        -------
        Any
            Standard deviations over grounded locations per frequency module.
        """
        raise NotImplementedError("f_sigma_p method is not implemented yet.")

    def f_x(self, p):
        """Decode grounded locations into categorical observation distributions.

        Parameters
        ----------
        p:
            Grounded location codes, typically flattened across abstract and
            sensory dimensions.

        Returns
        -------
        tuple
            ``(probability, logits)`` giving both softmax probabilities and
            raw logits over observations.
        """
        raise NotImplementedError("f_x method is not implemented yet.")

    def f_c_star(self, compressed):
        """Decompress highest‑frequency sensory representation to full space.

        Parameters
        ----------
        compressed:
            Compressed sensory features at the highest frequency module.

        Returns
        -------
        Any
            Decompressed logits over the full observation space.
        """
        raise NotImplementedError("f_c_star method is not implemented yet.")

    def f_c(self, decompressed):
        """Compress raw observations into a compact representation.

        Parameters
        ----------
        decompressed:
            One‑hot or dense observation vectors as provided by the
            environment.

        Returns
        -------
        Any
            Compressed sensory representation (e.g. two‑hot encoding) used by
            the rest of the model.
        """
        raise NotImplementedError("f_c method is not implemented yet.")

    def f_n(self, x):
        """Normalise sensory observations frequency‑wise.

        Parameters
        ----------
        x:
            Sensory observations or features per frequency module.

        Returns
        -------
        Any
            Normalised sensory features suitable for subsequent weighting and
            tiling.
        """
        raise NotImplementedError("f_n method is not implemented yet.")

    def f_g(self, g):
        """Downsample abstract location codes.

        Parameters
        ----------
        g:
            Abstract location codes per frequency module.

        Returns
        -------
        Any
            Downsampled abstract codes for each frequency module.
        """
        raise NotImplementedError("f_g method is not implemented yet.")

    def f_g_clamp(self, g):
        """Clamp abstract location activations to a bounded range.

        Parameters
        ----------
        g:
            Abstract location codes to be clamped.

        Returns
        -------
        Any
            Clamped abstract codes (e.g. in ``[-1, 1]``) per module.
        """
        raise NotImplementedError("f_g_clamp method is not implemented yet.")

    def f_p(self, p):
        """Apply nonlinearity to grounded location activations.

        Parameters
        ----------
        p:
            Grounded location activations (single tensor or list per
            frequency module).

        Returns
        -------
        Any
            Activated grounded location codes, typically after clamping and a
            leaky‑ReLU‑like nonlinearity to encourage sparsity.
        """
        raise NotImplementedError("f_p method is not implemented yet.")

    def attractor(self, p_query, M, retrieve_it_mask=None):
        """Run attractor dynamics to retrieve grounded locations from memory.

        Parameters
        ----------
        p_query:
            Initial query grounded locations (e.g. from abstract codes or
            sensory input) across frequency modules.
        M:
            Hebbian memory connectivity matrix storing grounded locations.
        retrieve_it_mask:
            Optional per‑iteration mask for hierarchical retrieval, allowing
            some frequencies to stop updating earlier than others.

        Returns
        -------
        Any
            Retrieved grounded location codes ``p`` per frequency module
            after running the attractor dynamics.
        """
        raise NotImplementedError("Attractor method is not implemented yet.")

    def hebbian(self, M_prev, p_inferred, p_generated, do_hierarchical_connections=True):
        """Update Hebbian memory with inferred and generated grounded locations.

        Parameters
        ----------
        M_prev:
            Previous Hebbian connectivity matrix for grounded locations.
        p_inferred:
            Grounded locations inferred from sensory input and abstract codes.
        p_generated:
            Grounded locations generated by the model (e.g. from abstract
            codes alone).
        do_hierarchical_connections:
            If ``True``, apply a connectivity mask implementing hierarchical
            frequency‑to‑frequency connections during the update.

        Returns
        -------
        Any
            Updated Hebbian memory matrix ``M``.
        """
        raise NotImplementedError("Hebbian method is not implemented yet.")


class Iteration:
    """Data container for a single iteration's state and outputs.

    Attributes
    ----------
    g : List[Tensor]
        Abstract location codes per frequency module.
    x : List[Tensor]
        Filtered sensory observations per frequency module.
    a : List
        Actions taken at this timestep.
    L : List[Tensor]
        Loss components for this iteration.
    M : List[Tensor]
        Hebbian memory matrices.
    g_gen : List[Tensor], optional
        Generated abstract locations from transition model.
    p_gen : List[Tensor], optional
        Generated grounded locations from memory retrieval.
    x_gen : Tensor, optional
        Generated observations from decoder.
    x_logits : List[Tensor], optional
        Logits for observation predictions.
    x_inf : List[Tensor], optional
        Inferred filtered sensory features.
    g_inf : List[Tensor], optional
        Inferred abstract locations.
    p_inf : List[Tensor], optional
        Inferred grounded locations.
    """

    def __init__(self, g, x, a, L, M, g_gen=None, p_gen=None, x_gen=None, x_logits=None, x_inf=None, g_inf=None, p_inf=None):
        self.g = g
        self.x = x
        self.a = a
        self.L = L
        self.M = M
        self.g_gen = g_gen
        self.p_gen = p_gen
        self.x_gen = x_gen
        self.x_logits = x_logits
        self.x_inf = x_inf
        self.g_inf = g_inf
        self.p_inf = p_inf

    def correct(self, g=None, p=None):
        """Correct inference with ground truth locations.

        Parameters
        ----------
        g : List[Tensor], optional
            Ground truth abstract locations.
        p : List[Tensor], optional
            Ground truth grounded locations.
        """
        if g is not None:
            self.g_inf = g
        if p is not None:
            self.p_inf = p

    def detach(self):
        """Detach all tensors from computation graph."""
        self.g = [g.detach() if g is not None else None for g in self.g] if self.g else None
        self.x = [x.detach() if x is not None else None for x in self.x] if self.x else None
        self.L = [l.detach() if l is not None else None for l in self.L] if self.L else None
        self.M = [m.detach() if m is not None else None for m in self.M] if self.M else None
        if self.g_gen:
            self.g_gen = [g.detach() if g is not None else None for g in self.g_gen]
        if self.p_gen:
            self.p_gen = [p.detach() if p is not None else None for p in self.p_gen]
        if self.x_gen is not None:
            self.x_gen = self.x_gen.detach()
        if self.x_logits:
            self.x_logits = [xl.detach() if xl is not None else None for xl in self.x_logits]
        if self.x_inf:
            self.x_inf = [x.detach() if x is not None else None for x in self.x_inf]
        if self.g_inf:
            self.g_inf = [g.detach() if g is not None else None for g in self.g_inf]
        if self.p_inf:
            self.p_inf = [p.detach() if p is not None else None for p in self.p_inf]
