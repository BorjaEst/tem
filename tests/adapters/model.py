"""Legacy model adapter for backward compatibility with run.py.

This module provides a drop-in replacement for the legacy model.Model class,
wrapping the new torch_tem.TEMModel to work with existing training scripts.
"""

import copy
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

import utils
from torch_tem.config import ModelConfig
from torch_tem.core.hpc import MemoryStorage
from torch_tem.core.model import TEMModel


class Iteration:
    """Legacy Iteration object wrapping TEM state and intermediate values.

    This matches the interface expected by run.py for accessing states,
    losses, and computing accuracy.

    Attributes:
        g: Ground truth abstract locations (list of dicts).
        x: Sensory observations (tensor).
        a: Actions (list of ints or None).
        L: Loss components (list of 8 tensors).
        M: Memory matrices [M_gen, M_inf].
        g_gen: Generated abstract locations (list of tensors per frequency).
        p_gen: Generated grounded locations (list of tensors per frequency).
        x_gen: Generated observations (list of tensors per frequency).
        x_logits: Observation logits (list of tensors).
        x_inf: Filtered sensory observations (list of tensors per frequency).
        g_inf: Inferred abstract locations (list of tensors per frequency).
        p_inf: Inferred grounded locations (list of tensors per frequency).
    """

    def __init__(self, g=None, x=None, a=None, L=None, M=None, g_gen=None, p_gen=None, x_gen=None, x_logits=None, x_inf=None, g_inf=None, p_inf=None):
        """Initialize iteration with all state variables."""
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

    def correct(self):
        """Calculate prediction accuracy for each frequency module.

        Returns:
            List of boolean arrays indicating correct predictions per module.
        """
        observation = self.x.detach().numpy()
        predictions = [tensor.detach().numpy() for tensor in self.x_gen]
        accuracy = [np.argmax(prediction, axis=-1) == np.argmax(observation, axis=-1) for prediction in predictions]
        return accuracy

    def detach(self):
        """Detach all tensors from computation graph.

        Returns:
            Self with all tensors detached.
        """
        if self.L is not None:
            self.L = [tensor.detach() for tensor in self.L]
        if self.M is not None:
            self.M = [tensor.detach() for tensor in self.M]
        if self.g_gen is not None:
            self.g_gen = [tensor.detach() for tensor in self.g_gen]
        if self.p_gen is not None:
            self.p_gen = [tensor.detach() for tensor in self.p_gen]
        if self.x_gen is not None:
            self.x_gen = [tensor.detach() for tensor in self.x_gen]
        if self.x_inf is not None:
            self.x_inf = [tensor.detach() for tensor in self.x_inf]
        if self.g_inf is not None:
            self.g_inf = [tensor.detach() for tensor in self.g_inf]
        if self.p_inf is not None:
            self.p_inf = [tensor.detach() for tensor in self.p_inf]
        return self


class Model(TEMModel):
    """Legacy compatibility wrapper for TEMModel.

    This class provides a drop-in replacement for the original `model.Model`
    class, adapting the new `TEMModel` interface to match the legacy API.

    The adapter handles:
    - Converting dict-based params to typed config
    - Maintaining .hyper dict for runtime parameter updates
    - Walk/chunk format conversion
    - Iteration object wrapping
    - Loss computation

    Note:
        This class is intended for backward compatibility with existing
        codebases using the original `model.Model`. New code should use
        `TEMModel` directly.
    """

    def __init__(self, params):
        """Initialize model from legacy parameter dict.

        Args:
            params: Legacy parameter dictionary (from parameters.parameters()).
        """
        # Import adapter here to avoid circular dependency
        from .legacy_adapter import legacy_to_typed

        # Convert legacy params to typed config
        config = legacy_to_typed(params)

        # Store config wrapper for later access
        self.config_wrapper = config

        # Initialize parent TEMModel with architecture config only
        super().__init__(config.architecture)

        # Store mutable hyperparameters dict for backward compatibility
        # This allows run.py to update eta, lambda, etc. at runtime
        self.hyper = copy.deepcopy(params)

        # Set batch size (don't call set_batch_size here, just set directly)
        # The parent already initialized batch_size from config.architecture

    @property
    def g_init(self):
        """Access g_init from MEC abstract location module."""
        if hasattr(self.mec, "abstract") and hasattr(self.mec.abstract, "g_init"):
            return self.mec.abstract.g_init
        # Fallback: create default g_init if not found
        return nn.ParameterList([nn.Parameter(torch.randn(self.hyper["n_g"][f]) * self.hyper.get("g_init_std", 0.5)) for f in range(self.hyper["n_f"])])

    def set_batch_size(self, batch_size: int):
        """Update batch size for memory storage.

        Args:
            batch_size: New batch size.
        """
        # Update batch size in config wrapper
        self.config_wrapper.architecture.batch_size = batch_size
        self.batch_size = batch_size

        # Re-initialize storage with new batch size
        # Keep the same params and mask
        if hasattr(self, "memory") and hasattr(self.memory, "storage"):
            p_update_mask = self.memory.storage.p_update_mask
            self.memory.storage = MemoryStorage(self.config, p_update_mask, batch_size)

    def forward(self, walk, prev_iter=None, prev_M=None) -> List[Iteration]:
        """Forward pass for compatibility with legacy Model.

        Parameters
        ----------
        walk : List[(locations, observations, actions), ...]
            List of walk steps, where each step is a tuple of:
            - locations: List of location dicts for each batch element
            - observations: Stacked observation tensor [batch, n_o]
            - actions: List of action ints (or None) for each batch element
        prev_iter : List[Iteration], optional
            List of Iteration objects from previous walk segment.
            If provided, the last iteration's memory is used for initialization.
        prev_M : List[Tensor], optional
            Previous memory state [M_gen, M_inf]. Used if prev_iter is None.

        Returns
        -------
        List[Iteration]
            List of Iteration objects for each step in the walk.
        """
        # Initialize walk states from previous iteration
        steps = self.init_walks(prev_iter)

        # Process each step in the walk
        for locations, x, actions in walk:
            # If no previous iteration: create initial iteration
            if steps is None:
                steps = [self.init_iteration(locations, x, [None for _ in actions], prev_M)]

            # Perform TEM iteration
            L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = self.iteration(x, locations, steps[-1].a, steps[-1].M, steps[-1].x_inf, steps[-1].g_inf)

            # Store iteration results
            steps.append(Iteration(g=locations, x=x, a=actions, L=L, M=M, g_gen=g_gen, p_gen=p_gen, x_gen=x_gen, x_logits=x_logits, x_inf=x_inf, g_inf=g_inf, p_inf=p_inf))

        # Remove initialization step (first element)
        steps = steps[1:]

        return steps

    def iteration(self, x, locations, a_prev, M_prev, x_prev, g_prev):
        """Single TEM iteration matching legacy interface.

        Args:
            x: Sensory observations [batch, n_o].
            locations: List of location dicts.
            a_prev: Previous actions (list of ints or None).
            M_prev: Previous memory [M_gen, M_inf].
            x_prev: Previous filtered observations (list per frequency).
            g_prev: Previous abstract locations (list per frequency).

        Returns:
            Tuple of (L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf).
        """
        # Transition step (needed for both inference and generative)
        gt_gen, gt_inf = self.gen_g(a_prev, g_prev, locations)

        # Inference pathway: x → x_inf, g_inf, p_inf
        x_inf, g_inf, p_inf_x, p_inf = self.inference(x, locations, M_prev, x_prev, gt_inf)

        # Generative pathway: g_inf → p_gen → x_gen
        x_gen, x_logits, p_gen = self.generative(M_prev, p_inf, g_inf, gt_gen)

        # Memory update (Hebbian learning)
        M = [self.hebbian(M_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]

        # Add inference memory if used
        if self.hyper["use_p_inf"]:
            if self.hyper["common_memory"]:
                M.append(M[0])
            else:
                M.append(self.hebbian(M_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1), do_hierarchical_connections=False))

        # Compute losses
        L = self.loss(gt_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev)

        return L, M, gt_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf

    def init_iteration(self, g, x, a, M):
        """Initialize first iteration with default values.

        Args:
            g: Ground truth locations.
            x: Initial observations.
            a: Initial actions (typically [None, ...]).
            M: Memory matrices or None.

        Returns:
            Initial Iteration object.
        """
        # Update batch size from data
        batch_size = x.shape[0]
        self.hyper["batch_size"] = batch_size

        # Initialize memory if not provided
        if M is None:
            n_p_total = sum(self.hyper["n_p"])
            M = [torch.zeros((batch_size, n_p_total, n_p_total), dtype=torch.float, device=x.device)]

            if self.hyper["use_p_inf"]:
                if self.hyper["common_memory"]:
                    M.append(M[0])
                else:
                    M.append(torch.zeros((batch_size, n_p_total, n_p_total), dtype=torch.float, device=x.device))

        # Initialize abstract location with learned prior
        g_inf = [torch.stack([self.g_init[f] for _ in range(batch_size)]) for f in range(self.hyper["n_f"])]

        # Initialize filtered observations as zeros
        x_inf = [torch.zeros((batch_size, self.hyper["n_x"][f]), device=x.device) for f in range(self.hyper["n_f"])]

        return Iteration(g=g, x=x, a=a, M=M, x_inf=x_inf, g_inf=g_inf)

    def init_walks(self, prev_iter):
        """Reset parameters for new walks in the batch.

        Args:
            prev_iter: Previous iteration list or None.

        Returns:
            Updated previous iteration or None.
        """
        if prev_iter is not None:
            # Check for new walks (indicated by None actions)
            for a_i, a in enumerate(prev_iter[0].a):
                if a is None:
                    # Reset memory for this walk
                    for M in prev_iter[0].M:
                        M[a_i, :, :] = 0

                    # Reset abstract location to prior
                    for f, g_inf in enumerate(prev_iter[0].g_inf):
                        g_inf[a_i, :] = self.g_init[f]

                    # Reset filtered observations
                    for f, x_inf in enumerate(prev_iter[0].x_inf):
                        x_inf[a_i, :] = 0

        return prev_iter

    def gen_g(self, a_prev, g_prev, locations):
        """Generate abstract location from transition.

        Delegates to MEC model's transition logic.

        Args:
            a_prev: Previous actions.
            g_prev: Previous abstract locations.
            locations: Current location dicts.

        Returns:
            Tuple of (g_gen, g_inf) - generated and inferred abstract locations.
        """
        # Use MEC's transition model
        # For now, return g_prev as placeholder
        # TODO: Implement proper transition logic from MEC
        return g_prev, g_prev

    def inference(self, x, locations, M_prev, x_prev, g_gen):
        """Inference pathway: sensory observation → grounded location.

        Args:
            x: Sensory observations.
            locations: Location dicts.
            M_prev: Previous memory.
            x_prev: Previous filtered observations.
            g_gen: Generated abstract location.

        Returns:
            Tuple of (x, g, p_x, p).
        """
        # Compress and filter sensory observation (LEC pathway)
        x_c = self.f_c(x)
        x = self.x_prev2x(x_prev, x_c)
        x_ = self.x2x_(x)

        # Retrieve from memory using sensory input
        p_x = self.attractor(x_, M_prev[1], for_inference=True) if self.hyper["use_p_inf"] else None

        # Infer abstract location
        g = self.inf_g(p_x, g_gen, x, locations)

        # Project abstract location
        g_ = self.g2g_(g)

        # Infer grounded location from sensory and abstract
        p = self.inf_p(x_, g_)

        return x, g, p_x, p

    def generative(self, M_prev, p_inf, g_inf, g_gen):
        """Generative pathway: abstract location → sensory prediction.

        Args:
            M_prev: Previous memory.
            p_inf: Inferred grounded locations.
            g_inf: Inferred abstract locations.
            g_gen: Generated abstract locations.

        Returns:
            Tuple of (x_gen, x_logits, p_gen).
        """
        # Generate from inferred grounded location (full multi-scale)
        x_p, x_p_logits = self.gen_x(p_inf)

        # Retrieve grounded location from inferred abstract location
        p_g_inf = self.gen_p(g_inf, M_prev[0])
        x_g, x_g_logits = self.gen_x(p_g_inf)

        # Retrieve grounded location from generated abstract location
        p_g_gen = self.gen_p(g_gen, M_prev[0])
        x_gt, x_gt_logits = self.gen_x(p_g_gen)

        return (x_p, x_g, x_gt), (x_p_logits, x_g_logits, x_gt_logits), p_g_inf

    def loss(self, g_gen, p_gen, x_logits, x, g_inf, p_inf, p_inf_x, M_prev):
        """Compute all loss components.

        Args:
            g_gen: Generated abstract locations.
            p_gen: Generated grounded locations.
            x_logits: Observation prediction logits.
            x: True observations.
            g_inf: Inferred abstract locations.
            p_inf: Inferred grounded locations.
            p_inf_x: Sensory-inferred grounded locations.
            M_prev: Previous memory.

        Returns:
            List of 8 loss tensors: [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p,
                                     L_g, L_reg_g, L_reg_p].
        """
        # Grounded location consistency losses
        L_p_g = torch.sum(torch.stack(utils.squared_error(p_inf, p_gen), dim=0), dim=0)

        L_p_x = torch.sum(torch.stack(utils.squared_error(p_inf, p_inf_x), dim=0), dim=0) if self.hyper["use_p_inf"] else torch.zeros_like(L_p_g)

        # Abstract location consistency loss
        L_g = torch.sum(torch.stack(utils.squared_error(g_inf, g_gen), dim=0), dim=0)

        # Observation reconstruction losses
        labels = torch.argmax(x, 1)
        L_x_gen = utils.cross_entropy(x_logits[2], labels)
        L_x_g = utils.cross_entropy(x_logits[1], labels)
        L_x_p = utils.cross_entropy(x_logits[0], labels)

        # Regularization losses
        L_reg_g = torch.sum(torch.stack([torch.sum(g**2, dim=1) for g in g_inf], dim=0), dim=0)

        L_reg_p = torch.sum(torch.stack([torch.sum(torch.abs(p), dim=1) for p in p_inf], dim=0), dim=0)

        return [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]

    # Placeholder methods - delegate to torch_tem components or implement stubs
    def f_c(self, x):
        """Sensory compression (one-hot to two-hot)."""
        # Use LEC encoder
        return self.lec.encoder(x)

    def x_prev2x(self, x_prev, x_c):
        """Temporal filtering of sensory observations."""
        # Use LEC processor - note argument order: (x_c, x_prev)
        return self.lec.processor(x_c, x_prev)

    def x2x_(self, x):
        """Project filtered observation to hippocampal input."""
        # Use LEC projection with tiling matrices
        return self.lec.projection(x, self.lec.get_W_tile())

    def attractor(self, x_, M, for_inference=False):
        """Memory retrieval via attractor dynamics."""
        # Use HPC attractor - note: for_inference flag replaces n_iterations mask
        return self.memory.attractor(x_, M, for_inference=for_inference)

    def inf_g(self, p_x, g_gen, x, locations):
        """Infer abstract location from memory and transition."""
        # Use MEC inference
        # For now, return g_gen as placeholder
        return g_gen

    def g2g_(self, g):
        """Project abstract location to hippocampal input."""
        # Use MEC projection to expand g to hippocampal dimension
        return self.mec.projection(g)

    def inf_p(self, x_, g_):
        """Infer grounded location from sensory and abstract inputs."""
        # Use grounded inference
        return self.grounded(g_, x_)

    def gen_x(self, p):
        """Generate sensory observation from grounded location."""
        # Use LEC decoder with first tiling matrix
        pred = self.lec.decoder(p, self.lec.get_tile_matrix(0))
        # Unwrap the list (SensoryPrediction contains single-element lists)
        return pred.values[0], pred.logits[0]

    def gen_p(self, g, M):
        """Generate grounded location from abstract location."""
        # Project abstract location to hippocampal space first
        g_ = self.g2g_(g)
        # Then retrieve from memory using the projected query
        return self.memory.retrieve(g_, for_inference=False)

    def hebbian(self, M, p_pre, p_post, do_hierarchical_connections=True):
        """Hebbian memory update.

        Args:
            M: Current memory matrix.
            p_pre: Pre-synaptic grounded locations.
            p_post: Post-synaptic grounded locations.
            do_hierarchical_connections: Whether to apply hierarchical mask.

        Returns:
            Updated memory matrix.
        """
        # Use memory storage update
        eta = self.hyper["eta"]
        lamb = 1.0 - self.hyper.get("kappa", 0.0)  # Convert retention to forgetting

        # Simple Hebbian update without going through storage
        # M_new = lamb * M + eta * (p_post @ p_pre.T)
        dM = torch.bmm(p_post.unsqueeze(2), p_pre.unsqueeze(1))
        M_new = lamb * M + eta * dM

        return M_new
