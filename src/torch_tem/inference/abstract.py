"""Abstract location inference for the Tolman–Eichenbaum Machine (TEM).

This module computes the abstract location representation (``g``) by
fusing multiple information sources with uncertainty-aware (precision)
weighting. It is the integration hub between predictive dynamics and
episodic memory:

Sources combined per frequency module f:
1) Transition prediction: ``g_gen[f]`` with uncertainty ``sigma_g_gen[f]``
2) Memory-based inference (optional): ``g_downsampled[f] -> g`` via a learned MLP
3) Salient object ("shiny") cues (optional): ``(mu_g_shiny[f], sigma_g_shiny[f])``

Fusion rule (precision-weighted mean):

        precision_i = 1 / (sigma_i^2 + eps)
        g_inf = sum_i precision_i * mu_i / sum_i precision_i

Key ideas:
- Memory influence is scheduled via an offset added to ``sigma_g_mem``
    (``p2g_scale_offset``), delaying strong reliance on memory early on.
- Memory uncertainty is predicted from simple quality indicators
    (norm and placeholder reconstruction error) per frequency.
- The output ``g_inf`` supports structural generalization while remaining
    anchored to retrieved episodic content when available.

Shapes (per frequency f):
- ``g_gen[f]``: [B, n_g[f]], ``sigma_g_gen[f]``: [B, n_g[f]]
- ``g_downsampled[f]`` (if used): [B, n_g_subsampled[f]] → mapped to ``mu_g_mem[f]``
- Returns ``g_inf[f]``: [B, n_g[f]]

This implementation follows the style/patterns of other TEM modules and
is designed to be testable with simple parameter stubs.
"""

from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, GroundedLocation, Matrix, SensoryObservation, Transition


class AbstractLocParams(Protocol):
    """Architecture parameters needed by AbstractLocInference."""

    n_f: int
    n_g: List[int]
    n_g_subsampled: List[int]
    g_init_std: float
    g_mem_std: float
    p2g_scale_offset: float
    p2g_sig_val: float
    do_sample: bool
    separate_ovc: bool
    n_f_g: int
    n_f_ovc: int


class AbstractLocInference(nn.Module):
    """Infers abstract location g from multiple sources.

    Sources:
    1. g_gen - Transition prediction (path integration)
    2. p_x -> g - Memory-based inference (only if p_x is provided)
    3. shiny objects - Direct inference from environment cues

    Combines via precision-weighted mean.

    Note:
        Memory-based inference is controlled by whether p_x is None or not,
        not by a configuration flag. This simplifies the logic and delegates
        the decision to the caller.
    """

    def __init__(self, params: AbstractLocParams, W_repeat: List[Matrix], decoder: Callable):
        """Initialize abstract location inference.

        Args:
            params: Architecture configuration (n_f, n_g, n_g_subsampled, g_init_std, g_mem_std, p2g_scale_offset, p2g_sig_val)
            W_repeat: Projection matrices for p_x -> g_downsampled transformation
            decoder: Decoder function (p -> x) for reconstruction error computation
        """
        super().__init__()
        self.n_f = params.n_f
        self.n_g = params.n_g
        self.n_g_subsampled = list(params.n_g_subsampled)
        self.p2g_scale_offset = params.p2g_scale_offset
        self.p2g_sig_val = params.p2g_sig_val
        self.do_sample = params.do_sample
        self.separate_ovc = params.separate_ovc
        self.n_f_g = params.n_f_g
        self.n_f_ovc = params.n_f_ovc

        # Store projection matrices and decoder
        self.W_repeat = W_repeat
        self.decoder = decoder

        # MLPs for memory-based g inference
        self.mlp_mu_g_mem = MLP(in_dim=self.n_g_subsampled, out_dim=self.n_g, hidden_dim=[2 * g for g in self.n_g])

        # Initialize with small random weights
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * params.g_mem_std for w in self.mlp_mu_g_mem.get_weights(-1)])
        self.mlp_sigma_g_mem = MLP(in_dim=[2] * self.n_f, out_dim=self.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.n_g])

        # MLPs for shiny object inference (object vector cells)
        n_ovc_modules = self.n_f_ovc if self.separate_ovc else self.n_f
        ovc_start = self.n_f_g if self.separate_ovc else 0
        ovc_dims = self.n_g[ovc_start:]
        self.mlp_mu_g_shiny = MLP(in_dim=[1] * n_ovc_modules, out_dim=ovc_dims, hidden_dim=[2 * g for g in ovc_dims])
        self.mlp_sigma_g_shiny = MLP(in_dim=[1] * n_ovc_modules, out_dim=ovc_dims, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in ovc_dims])

        # Learnable initial g for new environments
        self.g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * params.g_init_std) for g in self.n_g])
        self.logsig_g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * params.g_init_std) for g in self.n_g])

    def forward(self, p_x: Optional[GroundedLocation], g_gen: Transition, x: SensoryObservation, locations: List[Dict[str, Any]]) -> AbstractLocation:
        """Infer abstract location matching legacy inf_g interface.

        Args:
            p_x: Grounded location retrieved from memory using sensory input.
                 If None, memory-based inference is skipped.
            g_gen: Transition prediction as (mu_g_path, sigma_g_path)
            x: Current sensory observation for reconstruction error computation
            locations: Environment descriptors containing shiny object information

        Returns:
            g_inf: Inferred abstract location (sampled or mean depending on config)
        """
        # Start with path integration as base (Source 1: always present)
        g_transition = g_gen

        # Source 2: Memory-based inference (only if p_x is provided)
        if p_x is not None:
            g_transition = self._fuse_memory_path(g_transition, p_x, x)

        # Source 3: Shiny objects (if present in environment)
        shiny_envs = [location.get("shiny") is not None for location in locations]
        if any(shiny_envs):
            g_transition = self._fuse_shiny_signals(g_transition, locations, shiny_envs)

        # Sample or return mean based on configuration
        mu_g, sigma_g = g_transition
        if self.do_sample:
            g = [mu_g[f] + sigma_g[f] * torch.randn_like(mu_g[f]) for f in range(self.n_f)]
        else:
            g = mu_g

        return g

    def _fuse_memory_path(
        self,
        g_transition: Transition,
        p_x: GroundedLocation,
        x: SensoryObservation,
    ) -> Transition:
        """Fuse memory-based inference with current estimates.

        Implements precision-weighted fusion of abstract location estimates
        from path integration and memory retrieval.

        Args:
            g_transition: Current abstract location transition (mu_g, sigma_g)
            p_x: Grounded location retrieved from memory
            x: Sensory observation for reconstruction error

        Returns:
            g_fused: Updated abstract location transition with memory
        """
        mu_g, sigma_g = g_transition

        # Step 1: Project p_x to g_downsampled using W_repeat^T
        # Legacy: g_downsampled = [torch.matmul(p_x[f], torch.t(self.hyper['W_repeat'][f]))]
        g_downsampled = [torch.matmul(p_x[f], self.W_repeat[f].t()) for f in range(self.n_f)]

        # Step 2: Compute mu_g_mem from g_downsampled via MLP
        mu_g_mem = self.mlp_mu_g_mem(g_downsampled)

        # Step 3: Compute reconstruction error for memory quality
        # Legacy: with torch.no_grad(): x_hat, _ = self.gen_x(p_x[0]); err = utils.squared_error(x, x_hat)
        with torch.no_grad():
            x_hat, _ = self.decoder(p_x[0])
            # Squared error: (x - x_hat)^2 summed over observation dimension
            err = torch.sum((x - x_hat) ** 2, dim=1)

        # Step 4: Compute sigma_g_mem from memory quality indicators
        # Legacy: sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1)]
        quality = [torch.cat([torch.sum(g**2, dim=1, keepdim=True), err.unsqueeze(1)], dim=1) for g in mu_g_mem]
        sigma_g_mem_base = self.mlp_sigma_g_mem(quality)

        # Apply scheduling offset (legacy: sigma + p2g_scale_offset * p2g_sig_val)
        sigma_g_mem = [s + self.p2g_scale_offset * self.p2g_sig_val for s in sigma_g_mem_base]

        # Step 5: Clamp mu_g_mem to [-1, 1] (legacy stability)
        mu_g_mem = [torch.clamp(g, -1, 1) for g in mu_g_mem]

        # Step 6: Precision-weighted fusion per frequency
        mu_g_fused = []
        sigma_g_fused = []
        for f in range(self.n_f):
            mu_f, sigma_f = self._precision_weighted_mean([mu_g[f], mu_g_mem[f]], [sigma_g[f], sigma_g_mem[f]])
            mu_g_fused.append(mu_f)
            sigma_g_fused.append(sigma_f)

        return (mu_g_fused, sigma_g_fused)

    def _fuse_shiny_signals(
        self,
        g_transition: Transition,
        locations: List[Dict[str, Any]],
        shiny_envs: List[bool],
    ) -> Transition:
        """Fuse salient object signals with current estimates.

        Applies precision-weighted fusion only to environments and frequency
        modules affected by shiny object cues.

        Args:
            g_transition: Current abstract location transition (mu_g, sigma_g)
            locations: Environment descriptors with shiny information
            shiny_envs: Boolean mask indicating which environments have shiny objects

        Returns:
            g_fused: Updated abstract location transition with shiny cues
        """
        mu_g, sigma_g = g_transition

        # Step 1: Extract shiny indicators from environments with shiny objects
        shiny_locs = torch.stack([torch.tensor(loc["shiny"], dtype=torch.float, device=mu_g[0].device) for loc in locations if loc["shiny"] is not None]).unsqueeze(-1)

        # Step 2: Determine number of OVC modules
        n_ovc_modules = self.n_f_ovc if self.separate_ovc else self.n_f

        # Step 3: Compute mu_g_shiny and sigma_g_shiny via MLPs
        mu_g_shiny = self.mlp_mu_g_shiny([shiny_locs] * n_ovc_modules)
        sigma_g_shiny = self.mlp_sigma_g_shiny([shiny_locs] * n_ovc_modules)

        # Step 4: Take absolute for object vector cells (legacy: OVCs are positive)
        mu_g_shiny = [torch.abs(mu) for mu in mu_g_shiny]

        # Step 5: Apply clamp and leaky_relu (like grounded location activation)
        mu_g_shiny = self._apply_grounded_activation(mu_g_shiny)

        # Step 6: Determine which frequency modules are affected (separate_ovc config)
        module_start = self.n_f_g if self.separate_ovc else 0

        # Step 7: Precision-weighted fusion only for affected modules and environments
        # Make copies to avoid in-place modification
        mu_g_fused = [mu.clone() for mu in mu_g]
        sigma_g_fused = [sigma.clone() for sigma in sigma_g]

        for f in range(module_start, self.n_f):
            # Get shiny module index (offset by module_start)
            shiny_idx = f - module_start

            # Fuse only for environments with shiny objects
            mu_fused, sigma_fused = self._precision_weighted_mean([mu_g[f][shiny_envs, :], mu_g_shiny[shiny_idx]], [sigma_g[f][shiny_envs, :], sigma_g_shiny[shiny_idx]])

            # Step 8: Use masked_scatter to update only shiny environments
            mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
            mask[shiny_envs, :] = True
            mu_g_fused[f] = mu_g_fused[f].masked_scatter(mask, mu_fused)
            sigma_g_fused[f] = sigma_g_fused[f].masked_scatter(mask, sigma_fused)

        return (mu_g_fused, sigma_g_fused)

    def _apply_grounded_activation(self, p: List[Tensor]) -> List[Tensor]:
        """Apply grounded location activation (clamp + leaky_relu).

        This matches the legacy f_p activation for sparsity.

        Args:
            p: List of grounded location tensors

        Returns:
            Activated tensors with clamp and leaky_relu applied
        """
        return [torch.nn.functional.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p]

    def _precision_weighted_mean(self, means: List[Tensor], sigmas: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Compute precision-weighted mean of multiple estimates.

        Fuses multiple Gaussian estimates by weighting each by its precision
        (inverse variance). This is the optimal linear combination under
        Gaussian assumptions.

        Args:
            means: List of mean estimates [mu_1, mu_2, ...]
            sigmas: List of uncertainty estimates [sigma_1, sigma_2, ...]

        Returns:
            (mu_fused, sigma_fused): Combined mean and uncertainty

        Theory:
            Given multiple Gaussian estimates N(mu_i, sigma_i^2):
            precision_i = 1 / sigma_i^2
            mu_fused = sum(precision_i * mu_i) / sum(precision_i)
            sigma_fused = 1 / sqrt(sum(precision_i))
        """
        # Stack estimates along first dimension
        mus_stacked = torch.stack(means, dim=0)  # [n_sources, ...]
        sigmas_stacked = torch.stack(sigmas, dim=0)  # [n_sources, ...]

        # Compute precisions (inverse variance)
        precisions = 1.0 / (sigmas_stacked**2)  # [n_sources, ...]

        # Precision-weighted mean
        total_precision = torch.sum(precisions, dim=0)  # [...]
        weighted_mean = torch.sum(precisions * mus_stacked, dim=0) / total_precision  # [...]

        # Combined uncertainty
        combined_sigma = 1.0 / torch.sqrt(total_precision)  # [...]

        return weighted_mean, combined_sigma
