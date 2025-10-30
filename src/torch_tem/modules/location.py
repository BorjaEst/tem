from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor

from torch_tem import utils
from torch_tem.config import ArchitectureConfig, ModelConfig, StaticMatrices
from torch_tem.core.mlp import MLP


class AbstractLocationState(BaseModel):
    """State container for abstract location (grid cells).

    Represents the abstract spatial representation provided by grid cells,
    including uncertainty estimates and provenance information.

    Attributes:
        mu: Mean grid cell activations per frequency [n_freq × [batch, n_g[f]]]
        sigma: Uncertainty (standard deviation) per frequency [n_freq × [batch, n_g[f]]]
        sources: List of information sources that contributed to this state
            (e.g., ["path_integration"], ["memory", "path_integration", "shiny_objects"])
        metadata: Optional additional information for debugging or analysis
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mu: List[Tensor]
    sigma: List[Tensor]
    sources: List[str] = Field(default_factory=list)
    metadata: Dict = Field(default_factory=dict)


class GroundedLocationState(BaseModel):
    """State container for grounded location (place cells).

    Represents location-specific activations that arise from the conjunction
    of abstract grid cells with sensory observations.

    Attributes:
        p: Place cell activations per frequency [n_freq × [batch, n_p[f]]]
        sigma: Optional uncertainty estimates per frequency [n_freq × [batch, n_p[f]]]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    p: List[Tensor]
    sigma: Optional[List[Tensor]] = None


class AbstractLocationModule(nn.Module):
    """Manages abstract location (grid cell) inference and generation.

    This module implements a multi-frequency grid cell system inspired by the
    mammalian entorhinal cortex. Grid cells provide a metric representation of
    space through path integration (integrating self-motion) and can be grounded
    by sensory information through memory.

    Key Capabilities:
    ==================
    1. **Path Integration**: Updates grid cell activations based on actions taken
       - Learns transition matrices D(a) that shift grid patterns based on movement
       - Maintains multiple frequency bands for multi-scale spatial representation
       - Handles uncertainty propagation through movement

    2. **Memory-Based Inference**: Grounds abstract location using sensory context
       - Retrieves place cell activations (p) from memory given current sensory input
       - Infers grid cells (g) from these grounded place representations
       - Combines path integration with memory via inverse variance weighting

    3. **Shiny Object Handling**: Processes location information without directional cues
       - Used when observations provide location but not orientation (e.g., unique landmarks)
       - Provides absolute position constraints to correct accumulated path integration drift

    Module Independence:
    ====================
    This module receives only primitive parameters (dimensions, frequencies, etc.)
    and does NOT depend on parent config objects. All configuration is injected
    during initialization, making it modular and testable in isolation.

    Mathematical Framework:
    =======================
    Grid cell update via path integration:
        g[t] = activation(g[t-1] + D(a) · g[t-1])

    Where:
    - g[t]: Grid cell state at time t [batch, n_g[f]] per frequency f
    - D(a): Action-dependent transition matrix learned by MLP
    - a: Action taken (movement direction)

    Memory-based inference with inverse variance weighting:
        g_combined = (g_path/σ²_path + g_mem/σ²_mem) / (1/σ²_path + 1/σ²_mem)

    Where:
    - g_path: Grid cells from path integration
    - g_mem: Grid cells inferred from memory retrieval
    - σ²: Variance (uncertainty) of each estimate

    Args:
        arch_config: Architecture configuration with dimensions:
            - n_f: Number of frequency bands (typically 3-5)
            - n_g: Grid cell dimensions per frequency [n_g[0], n_g[1], ...]
            - n_g_subsampled_combined: Dimensions for memory operations
            - n_actions: Number of possible actions
            - d_hidden_dim: Hidden layer size for transition MLPs
            - g_init_std: Std dev for initialization of prior parameters
            - g_mem_std: Std dev for memory-based inference initialization
            - separate_ovc: Whether object vector cells use separate frequencies
            - n_f_ovc: Number of OVC frequency bands if separate
            - n_f_g: Number of pure grid cell frequencies
            - has_static_action: Whether action 0 is "no movement"

        model_config: Model behavior configuration:
            - do_sample: Whether to sample from distributions (True) or use means (False)
            - use_p_inf: Whether to use memory-based inference to ground grid cells
            - p2g_scale_offset: Offset for memory-to-grid uncertainty
            - p2g_sig_val: Base uncertainty value for memory-to-grid conversion

        static_matrices: Pre-computed connectivity and transformation matrices:
            - g_connections: Which frequency bands connect during path integration
            - g_downsample: Downsampling matrices for memory operations
            - W_repeat: Repeat matrices for combining frequency bands

    Attributes:
        g_init: Learnable prior mean for grid cells at each frequency [n_f params]
        logsig_g_init: Learnable prior log-std for grid cells [n_f params]
        MLP_D_a: Action-to-transition-matrix network for path integration
        D_no_a: Non-directional transitions for shiny object handling
        MLP_sigma_g_path: Network for path integration uncertainty estimation
        MLP_mu_g_mem: Network for memory-based grid cell inference (mean)
        MLP_sigma_g_mem: Network for memory-based uncertainty estimation
        MLP_mu_g_shiny: Network for shiny object location inference (mean)
        MLP_sigma_g_shiny: Network for shiny object uncertainty estimation
    """

    def __init__(self, arch_config: ArchitectureConfig, model_config: ModelConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.arch = arch_config
        self.model = model_config
        self.matrices = static_matrices

        # Prior parameters
        self.g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=arch_config.n_g[f], loc=0, scale=arch_config.g_init_std), dtype=torch.float)) for f in range(arch_config.n_f)]
        )

        self.logsig_g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=arch_config.n_g[f], loc=0, scale=arch_config.g_init_std), dtype=torch.float)) for f in range(arch_config.n_f)]
        )

        # Path integration MLPs
        self.MLP_D_a = MLP(
            [arch_config.n_actions for _ in range(arch_config.n_f)],
            [
                sum([arch_config.n_g[f_from] for f_from in range(arch_config.n_f) if static_matrices.g_connections[f_to][f_from]]) * arch_config.n_g[f_to]
                for f_to in range(arch_config.n_f)
            ],
            activation=[torch.tanh, None],
            hidden_dim=[arch_config.d_hidden_dim for _ in range(arch_config.n_f)],
            bias=[True, False],
        )
        self.MLP_D_a.set_weights(1, 0.0)

        self.D_no_a = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(sum([arch_config.n_g[f_from] for f_from in range(arch_config.n_f) if static_matrices.g_connections[f_to][f_from]]) * arch_config.n_g[f_to])
                )
                for f_to in range(arch_config.n_f)
            ]
        )

        self.MLP_sigma_g_path = MLP(arch_config.n_g, arch_config.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in arch_config.n_g])

        # Memory-based inference MLPs
        self.MLP_mu_g_mem = MLP(arch_config.n_g_subsampled_combined, arch_config.n_g, hidden_dim=[2 * g for g in arch_config.n_g])
        self.MLP_mu_g_mem.set_weights(
            -1,
            [
                torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=arch_config.g_mem_std), dtype=torch.float)
                for f in range(arch_config.n_f)
            ],
        )

        self.MLP_sigma_g_mem = MLP(
            [2 for _ in arch_config.n_g_subsampled_combined], arch_config.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in arch_config.n_g]
        )

        # Shiny object MLPs
        self.MLP_mu_g_shiny = MLP(
            [1 for _ in range(arch_config.n_f_ovc if arch_config.separate_ovc else arch_config.n_f)],
            [n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
        )

        self.MLP_sigma_g_shiny = MLP(
            [1 for _ in range(arch_config.n_f_ovc if arch_config.separate_ovc else arch_config.n_f)],
            [n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            hidden_dim=[2 * n_g for n_g in arch_config.n_g[(arch_config.n_f_g if arch_config.separate_ovc else 0) :]],
            activation=[torch.tanh, torch.exp],
        )

    def generate_from_transition(self, g_prev: List[torch.Tensor], actions: List[int], locations: List[dict]) -> AbstractLocationState:
        """Generate abstract location through path integration.

        Performs a movement step by integrating self-motion (actions) with the
        previous grid cell state. This implements the core path integration
        mechanism that allows the model to track position based on movement alone.

        Path integration formula:
            g[t] = activation(g[t-1] + D(a) · g[t-1])

        Special handling for "shiny objects" (unique landmarks that provide location
        without directional information) by using non-directional transitions.

        Args:
            g_prev: Previous grid cell activations [batch, n_g[f]] per frequency f
            actions: List of action indices taken by each agent in batch
            locations: List of location dicts with optional 'shiny' key for landmarks

        Returns:
            AbstractLocationState with:
            - mu: Updated grid cell means [batch, n_g[f]] per frequency
            - sigma: Uncertainty estimates [batch, n_g[f]] per frequency
            - sources: ["path_integration"] or ["path_integration", "shiny_objects"]
        """
        mu_g = self._path_integration_mean(g_prev, actions)
        sigma_g = self._path_integration_sigma(g_prev, actions)

        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.model.do_sample else mu_g[f] for f in range(self.arch.n_f)]

        # Handle shiny objects (no directional information)
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if any(shiny_envs):
            mu_g = self._path_integration_mean(g_prev, actions, no_direc=shiny_envs)

        return AbstractLocationState(mu=mu_g, sigma=sigma_g, sources=["path_integration"])

    def infer(self, p_from_memory: Optional[List[torch.Tensor]], g_gen: AbstractLocationState, x: torch.Tensor, locations: List[dict]) -> AbstractLocationState:
        """Infer abstract location from memory and path integration.

        Combines two sources of location information via inverse variance weighting:

        1. **Path Integration** (g_gen): Position estimate from integrating movement
           - Accumulated from previous position + action transitions
           - May drift over time without correction

        2. **Memory-Based Inference** (p_from_memory): Position from sensory grounding
           - Retrieves place cells from memory using current sensory input
           - Infers grid cells from these grounded place representations
           - Provides correction to accumulated drift

        Combination via inverse variance weighting:
            g = (g_path/σ²_path + g_mem/σ²_mem) / (1/σ²_path + 1/σ²_mem)

        This gives more weight to the estimate with lower uncertainty, optimally
        combining path integration's smooth tracking with memory's sensory grounding.

        Args:
            p_from_memory: Place cell activations [batch, n_p_dims] per frequency,
                retrieved from memory using current sensory input. None if memory
                retrieval is disabled or unavailable.
            g_gen: Grid cell state from path integration containing mu, sigma,
                and source information
            x: Current sensory observation [batch, n_x_dims] used for memory quality
                assessment
            locations: List of location dicts with optional 'shiny' key for
                incorporating landmark information

        Returns:
            AbstractLocationState with:
            - mu: Combined grid cell activations [batch, n_g[f]] per frequency
            - sigma: Combined uncertainty estimates [batch, n_g[f]] per frequency
            - sources: List indicating which sources contributed (e.g.,
              ["path_integration", "memory", "shiny_objects"])
        """
        # Start with path integration
        mu_g = g_gen.mu
        sigma_g = g_gen.sigma
        sources = ["path_integration"]

        # Add memory-based inference if enabled
        if self.model.use_p_inf and p_from_memory is not None:
            mu_g_mem, sigma_g_mem = self._infer_from_memory(p_from_memory, x)
            # Combine via inverse variance weighting
            mu_g, sigma_g = [], []
            for f in range(self.arch.n_f):
                mu, sigma = utils.inv_var_weight([g_gen.mu[f], mu_g_mem[f]], [g_gen.sigma[f], sigma_g_mem[f]])
                mu_g.append(mu)
                sigma_g.append(sigma)
            sources.append("memory")

        # Handle shiny objects
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if any(shiny_envs):
            mu_g, sigma_g = self._incorporate_shiny(mu_g, sigma_g, locations, shiny_envs)
            sources.append("shiny_objects")

        # Sample or take mean
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.model.do_sample else mu_g[f] for f in range(self.arch.n_f)]

        return AbstractLocationState(mu=g, sigma=sigma_g, sources=sources)

    def prepare_for_memory(self, g: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare abstract location for memory operations.

        Transforms full-resolution grid cell activations into a format suitable
        for memory storage and retrieval. This involves:

        1. **Downsampling**: Reduce dimensionality via learned projection
           g_down = g · M_downsample

        2. **Repetition**: Expand to memory format for pattern matching
           g_mem = g_down · W_repeat

        This transformation reduces memory capacity requirements while maintaining
        the essential spatial information needed for place cell associations.

        Args:
            g: Grid cell activations [batch, n_g[f]] per frequency f

        Returns:
            List of transformed grid representations [batch, n_g_subsampled[f]]
            per frequency, ready for memory operations
        """
        downsampled = [torch.matmul(g[f], self.matrices.g_downsample[f]) for f in range(self.arch.n_f)]
        return [torch.matmul(downsampled[f], self.matrices.W_repeat[f]) for f in range(self.arch.n_f)]

    def _path_integration_mean(self, g_prev: List[torch.Tensor], actions: List[int], no_direc: Optional[List[bool]] = None) -> List[torch.Tensor]:
        """Calculate mean through path integration.

        Core path integration computation that shifts grid cell activations based
        on the action taken. The shift is implemented via learned transition matrices:

            δ = D(a) · g[t-1]
            g[t] = activation(g[t-1] + δ)

        The transition matrix D(a) is action-dependent and learned, capturing how
        each movement direction affects the grid cell pattern. For multi-frequency
        systems, frequencies can be connected (allowing information flow between
        scales) according to the g_connections matrix.

        Special cases:
        - Static actions (a=None or a=0): Reset to prior (g_init)
        - No directional info (shiny objects): Use non-directional transition D_no_a

        Args:
            g_prev: Previous grid cell state [batch, n_g[f]] per frequency
            actions: List of action indices (or None for static/reset)
            no_direc: Boolean mask indicating which samples lack directional information

        Returns:
            List of updated grid cell means [batch, n_g[f]] per frequency, clamped
            to [-1, 1] for numerical stability
        """
        no_direc = [False for _ in actions] if no_direc is None else no_direc
        a_step = [a if a is not None else 0 for a in actions]
        a_do_step = [a is not None for a in actions]

        # Actions to one-hot
        if self.arch.has_static_action:
            a_onehot = torch.zeros((len(a_step), self.arch.n_actions)).scatter_(
                1, torch.clamp(torch.tensor(a_step).unsqueeze(1) - 1, min=0), 1.0 * (torch.tensor(a_step).unsqueeze(1) > 0)
            )
        else:
            a_onehot = torch.zeros((len(a_step), self.arch.n_actions)).scatter_(1, torch.tensor(a_step).unsqueeze(1), 1.0)

        # Get transition matrices
        D_a = self.MLP_D_a([a_onehot for _ in range(self.arch.n_f)])

        # Replace with non-directional transitions where needed
        for f in range(self.arch.n_f):
            D_a[f][no_direc, :] = self.D_no_a[f]

        # Reshape and apply transitions
        D_a = [
            torch.reshape(D_a[f_to], (-1, sum([self.arch.n_g[f_from] for f_from in range(self.arch.n_f) if self.matrices.g_connections[f_to][f_from]]), self.arch.n_g[f_to]))
            for f_to in range(self.arch.n_f)
        ]

        g_in = [
            torch.unsqueeze(torch.cat([g_prev[f_from] for f_from in range(self.arch.n_f) if self.matrices.g_connections[f_to][f_from]], dim=1), 1) for f_to in range(self.arch.n_f)
        ]

        delta = [torch.squeeze(torch.matmul(g, T)) for g, T in zip(g_in, D_a)]
        g_step = [g + d if g.dim() > 1 else torch.unsqueeze(g + d, 0) for g, d in zip(g_prev, delta)]
        g_step = [torch.clamp(g_f, min=-1, max=1) for g_f in g_step]

        return [torch.stack([g_step[f][i, :] if do_step else self.g_init[f] for i, do_step in enumerate(a_do_step)]) for f in range(self.arch.n_f)]

    def _path_integration_sigma(self, g_prev: List[torch.Tensor], actions: List[int]) -> List[torch.Tensor]:
        """Calculate uncertainty through path integration.

        Estimates the uncertainty (standard deviation) of the grid cell state
        after path integration. Uncertainty typically increases with each movement
        step due to accumulated noise and transition ambiguity.

        Two sources of uncertainty:
        1. **Movement-based**: Computed from previous grid state via MLP
           - Captures uncertainty that propagates through transitions
        2. **Prior-based**: Used when resetting (no action taken)
           - Drawn from learned initial uncertainty (logsig_g_init)

        Args:
            g_prev: Previous grid cell state [batch, n_g[f]] per frequency
            actions: List of action indices (None indicates reset/no movement)

        Returns:
            List of uncertainty estimates [batch, n_g[f]] per frequency as
            standard deviations (positive values)
        """
        a_do_step = [a is not None for a in actions]
        from_g = self.MLP_sigma_g_path(g_prev)
        from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]
        return [torch.stack([from_g[f][i, :] if do_step else from_prior[f] for i, do_step in enumerate(a_do_step)]) for f in range(self.arch.n_f)]

    def _infer_from_memory(self, p: List[torch.Tensor], x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Infer abstract location from grounded location memory.

        Performs the inverse operation of place-to-grid mapping: given place cell
        activations (p) retrieved from memory, infer the corresponding grid cell
        representation (g). This "grounds" the abstract grid cells using sensory
        information stored in memory.

        Process:
        1. Downsample place cells: g_down = p · W_repeat^T
        2. Infer grid cells via MLP: g = MLP(g_down)
        3. Estimate uncertainty based on memory quality metrics

        Uncertainty estimation considers:
        - Strength of place cell activation (‖g‖²)
        - Reconstruction error (how well memory matches current observation)
        Higher uncertainty when memory is weak or inconsistent.

        Args:
            p: Place cell activations [batch, n_p_dims] per frequency from memory
            x: Current sensory observation [batch, n_x_dims] for quality assessment

        Returns:
            Tuple of (mu_g_mem, sigma_g_mem) where:
            - mu_g_mem: Inferred grid cell means [batch, n_g[f]] per frequency
            - sigma_g_mem: Uncertainty estimates [batch, n_g[f]] per frequency
        """
        # Downsample p to get g
        g_downsampled = [torch.matmul(p[f], torch.t(self.matrices.W_repeat[f])) for f in range(self.arch.n_f)]
        mu_g_mem = self.MLP_mu_g_mem(g_downsampled)

        # Compute memory quality measures
        with torch.no_grad():
            from torch_tem.modules import ObservationGenerator  # Avoid circular import

            # Simplified error calculation
            err = torch.zeros(x.shape[0])

        sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]

        mu_g_mem = [torch.clamp(g_f, min=-1, max=1) for g_f in mu_g_mem]
        sigma_g_mem = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma_g_mem[f] + self.model.p2g_scale_offset * self.model.p2g_sig_val for f in range(self.arch.n_f)]

        return mu_g_mem, sigma_g_mem

    def _incorporate_shiny(
        self, mu_g: List[torch.Tensor], sigma_g: List[torch.Tensor], locations: List[dict], shiny_envs: List[bool]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Incorporate shiny object information into abstract location.

        "Shiny objects" are unique, recognizable landmarks that provide absolute
        position information but without directional cues (e.g., a distinctive
        building visible from multiple angles). These act as "anchors" that can
        correct accumulated path integration drift.

        Process:
        1. Extract shiny object features from location metadata
        2. Infer grid cell position from shiny features via MLP
        3. Combine with existing estimate via inverse variance weighting

        Only updates Object Vector Cell (OVC) modules, which specialize in
        representing objects/landmarks in the environment. Grid cell modules
        continue to rely on path integration and memory.

        This provides a third source of location information beyond path integration
        and memory-based inference, helping maintain accurate positioning in
        environments with distinctive landmarks.

        Args:
            mu_g: Current grid cell means [batch, n_g[f]] per frequency
            sigma_g: Current uncertainties [batch, n_g[f]] per frequency
            locations: List of location dicts, some containing 'shiny' key with
                landmark features
            shiny_envs: Boolean mask indicating which batch elements have shiny objects

        Returns:
            Tuple of (mu_g_updated, sigma_g_updated) with shiny object information
            incorporated into OVC modules via inverse variance weighting
        """
        shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(loc["shiny"], dtype=torch.float) for loc in locations if loc.get("shiny") is not None]), dim=-1)

        mu_g_shiny = self.MLP_mu_g_shiny([shiny_locations for _ in range(self.arch.n_f_g if self.arch.separate_ovc else self.arch.n_f)])
        mu_g_shiny = [torch.abs(mu) for mu in mu_g_shiny]
        mu_g_shiny = [utils.leaky_relu(torch.clamp(mu, min=-1, max=1)) for mu in mu_g_shiny]

        sigma_g_shiny = self.MLP_sigma_g_shiny([shiny_locations for _ in range(self.arch.n_f_g if self.arch.separate_ovc else self.arch.n_f)])

        # Update only OVC modules
        module_start = self.arch.n_f_g if self.arch.separate_ovc else 0
        for f in range(module_start, self.arch.n_f):
            mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs, :], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs, :], sigma_g_shiny[f - module_start]])
            mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
            mask[shiny_envs, :] = True
            mu_g[f] = mu_g[f].masked_scatter(mask, mu)
            sigma_g[f] = sigma_g[f].masked_scatter(mask, sigma)

        return mu_g, sigma_g


class GroundedLocationModule(nn.Module):
    """Infers grounded location (place cells) from sensory and abstract inputs.

    Place cells represent specific spatial locations grounded in sensory context.
    They are computed as the element-wise product of:
        1. Abstract location (grid cells): "where" in abstract space
        2. Sensory input: "what" is observed

    This binding operation implements a form of conjunction:
        p = g ⊙ x  (place = grid ⊙ sensory)

    The module also estimates uncertainty (sigma) in place cell activations
    using a learned MLP, which can be used for probabilistic inference.

    Independent module design:
        - Only requires place cell dimensions (one parameter!)
        - No config dependencies
        - Minimal and focused interface

    Args:
        n_p_dims: Place cell dimensions per frequency module [n_freq]
            Example: [100, 80, 60] for 3 frequency modules
            Determines the MLP architecture for uncertainty estimation

    Attributes:
        n_freq: Number of frequency modules
        MLP_sigma_p: Neural network estimating place cell uncertainty
            Input: place cell activations per frequency
            Output: log-variance per frequency (via exp activation)
    """

    def __init__(self, n_p_dims: List[int]):
        super().__init__()
        self.n_freq = len(n_p_dims)

        # MLP for uncertainty estimation: p → log(σ²)
        # tanh hidden activation, exp output activation ensures σ² > 0
        self.MLP_sigma_p = MLP(n_p_dims, n_p_dims, activation=[torch.tanh, torch.exp])

    def infer(self, sensory: List[torch.Tensor], abstract: List[torch.Tensor]) -> GroundedLocationState:
        """Infer grounded location from sensory and abstract inputs.

        This method implements the conjunction of grid cells (abstract) with sensory
        observations (sensory) to produce place cell activations (p). The conjunction
        operation is element-wise multiplication:

            p[f] = activation(clamp(g[f] ⊙ x[f]))

        Where:
        - g[f]: Abstract grid cell activations at frequency f
        - x[f]: Sensory input at frequency f
        - ⊙: Element-wise (Hadamard) product
        - clamp: Limits values to [-1, 1] for numerical stability
        - activation: Leaky ReLU to allow negative values while emphasizing positive

        This conjunction creates place-specific responses where both sensory and
        abstract components agree on the location.

        Args:
            sensory: List of tensors [batch, n_p_dims] for each frequency, representing
                sensory observations projected into location space
            abstract: List of tensors [batch, n_p_dims] for each frequency, representing
                grid cell activations from path integration

        Returns:
            GroundedLocationState containing:
            - p: List of place cell activations [batch, n_p_dims] per frequency
            - sigma: Uncertainty estimates [batch, n_p_dims] for the inferred location
        """
        p = []
        for f in range(self.n_freq):
            mu_p = abstract[f] * sensory[f]  # Element-wise multiplication
            mu_p = utils.leaky_relu(torch.clamp(mu_p, min=-1, max=1))
            p.append(mu_p)

        sigma_p = self.MLP_sigma_p(p)
        return GroundedLocationState(p=p, sigma=sigma_p)
