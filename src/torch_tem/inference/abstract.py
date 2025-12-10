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
- **Memory influence scheduling**: The uncertainty of memory-derived locations
    can be artificially increased by adding an offset:
    ``sigma_g_mem = sigma_base + p2g_scale_offset * p2g_sig_val``
    Higher offset → lower precision → less influence during fusion.
    This allows gradual integration of memory during training/inference.
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

from typing import Any, Dict, List, Optional, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.core.projection import ProjectionHead
from torch_tem.generation import ObservationDecoder
from torch_tem.types import AbstractLocation, GroundedLocation, SensoryObservation, Transition
from torch_tem.utils.fusion import fuse_transitions, sample_transition


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

    def __init__(self, params: AbstractLocParams, projection: ProjectionHead, decoder: ObservationDecoder):
        """Initialize abstract location inference.

        Args:
            params: Architecture configuration (n_f, n_g, n_g_subsampled, g_init_std, g_mem_std, p2g_scale_offset, p2g_sig_val)
            projection: Projection module for p_x -> g_downsampled transformation
            decoder: Decoder module (p -> x) for reconstruction error computation.
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

        # Store projection module and decoder
        self.projection = projection
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
        """Infer abstract location via multi-source Bayesian fusion.

        Fusion Pipeline:
            1. Path integration (g_gen) - always present
            2. Memory-based (p_x → g) - if memory available
            3. Shiny objects - if present in environment

        Each source contributes proportional to its precision (1/σ²).

        Args:
            p_x: Grounded location retrieved from memory using sensory input.
                 Used to compute memory-based location estimate via learned mapping.
                 If None, memory-based inference is skipped.
            g_gen: Path integration estimate from transition model.
                   Contains mean and uncertainty from previous location + action.
            x: Raw sensory observation. Used to assess memory retrieval quality
               via reconstruction error (even when p_x is None for initialization).
            locations: Environment descriptors containing shiny object information

        Returns:
            Inferred abstract location (sampled or mean depending on do_sample flag)

        Theory:
            Precision-weighted fusion combines multiple probabilistic estimates:
                precision_i = 1 / σ²_i
                g_fused = Σ(precision_i * g_i) / Σ(precision_i)
            Sources with higher uncertainty (larger σ) receive lower weight.
            Memory influence is controlled via uncertainty scheduling to enable
            gradual integration during learning.
        """
        # Collect all available estimates
        estimates = [g_gen]  # Always have path integration

        # Add memory estimate (if available)
        if memory_estimate := self._compute_memory_estimate(p_x, x):
            estimates.append(memory_estimate)

        # Add shiny object cues (if any)
        if shiny_estimate := self._compute_shiny_estimate(locations):
            estimates.append(shiny_estimate)

        # Fuse all estimates
        fused = fuse_transitions(estimates)

        # Return sampled or mean location
        return sample_transition(fused) if self.do_sample else fused.mean

    def _compute_memory_estimate(self, p_x: Optional[GroundedLocation], x: SensoryObservation) -> Optional[Transition]:
        """Compute memory-based location estimate with quality-dependent uncertainty.

        Maps retrieved grounded location p_x to abstract location g via:
            1. Project: p_x → g_downsampled (via projection.inverse_project())
            2. Upsample: g_downsampled → mu_g_mem (via MLP)
            3. Quality: reconstruction_error(x, decode(p_x)) → sigma_g_mem
            4. Schedule: sigma_g_mem += memory_offset (controls influence)

        Note:
            The projection module's inverse_project() handles the p→g transformation.
            This delegates the implementation details (mean vs sum, matrix operations)
            to the projection module, ensuring consistency across the codebase.
            The current implementation matches the original TensorFlow behavior.

        Args:
            p_x: Retrieved grounded location from memory (None if unavailable)
            x: Raw sensory observation for reconstruction error computation

        Returns:
            Memory estimate with quality-dependent uncertainty, or None if p_x unavailable

        Theory:
            Memory influence is controlled by uncertainty scheduling:
                σ_mem_effective = σ_base + scheduling_offset
            Higher offset → lower precision → less influence on fusion.
            This enables gradual memory integration during training.
        """
        # Early return if memory retrieval unavailable
        if p_x is None:
            return None

        # Step 1: Project retrieved location to downsampled abstract space
        # Use projection module's inverse_project() which implements the p→g transformation
        # matching the original TensorFlow implementation (mean over sensory dimension)
        g_downsampled = self.projection.inverse_project(p_x)

        # Step 2: Upsample to full abstract location via learned MLP
        mu_g_mem = self.mlp_mu_g_mem(g_downsampled)

        # Step 3: Assess memory quality via reconstruction error
        sigma_g_mem = self._compute_memory_uncertainty(mu_g_mem, x, p_x)

        # Step 4: Clamp to valid range (legacy stability)
        mu_g_mem = [torch.clamp(g, -1, 1) for g in mu_g_mem]

        return Transition(mean=mu_g_mem, uncertainty=sigma_g_mem)

    def _compute_memory_uncertainty(self, mu_g_mem: AbstractLocation, x: SensoryObservation, p_x: GroundedLocation) -> AbstractLocation:
        """Compute memory uncertainty from retrieval quality indicators.

        Quality indicators (per frequency):
            1. Norm: ||g||² (energy in retrieved code)
            2. Reconstruction error: ||x - decode(p_x)||² (sensory fidelity)

        Lower reconstruction error → lower uncertainty → higher precision weight

        Args:
            mu_g_mem: Memory-derived abstract location estimate
            x: Ground-truth sensory observation
            p_x: Retrieved grounded location

        Returns:
            Memory uncertainty with scheduling offset applied
        """
        # Compute reconstruction error (detached to avoid backprop)
        with torch.no_grad():
            x_hat, _ = self.decoder(p_x[0])  # Decode first frequency
            reconstruction_error = torch.sum((x - x_hat) ** 2, dim=1)

        # Concatenate quality indicators: [||g||², reconstruction_error]
        quality_features = [
            torch.cat(
                [
                    torch.sum(g**2, dim=1, keepdim=True),  # Norm indicator
                    reconstruction_error.unsqueeze(1),  # Reconstruction indicator
                ],
                dim=1,
            )
            for g in mu_g_mem
        ]

        # Predict base uncertainty from quality via MLP
        sigma_base = self.mlp_sigma_g_mem(quality_features)

        # Apply scheduling offset to control memory influence
        # Higher offset → higher sigma → lower precision → less influence
        sigma_scheduled = [sigma + self._compute_memory_scheduling() for sigma in sigma_base]

        return sigma_scheduled

    def _compute_memory_scheduling(self) -> float:
        """Compute current memory scheduling offset.

        Returns larger values during early training to reduce memory influence.
        As training progresses, offset decreases to allow stronger memory integration.

        Returns:
            Scheduling offset to add to memory uncertainty
        """
        return self.p2g_scale_offset * self.p2g_sig_val

    def _compute_shiny_estimate(self, locations: List[Dict[str, Any]]) -> Optional[Transition]:
        """Compute location estimate from salient object cues.

        Shiny objects provide direct location information from environmental features.
        Only environments with shiny objects receive this estimate, with per-environment
        masking applied during fusion.

        Args:
            locations: Environment descriptors with shiny information

        Returns:
            Shiny object estimate with per-environment masking, or None if no shiny objects

        Theory:
            Object vector cells encode salient landmark locations directly.
            This bypasses path integration, providing absolute positioning cues.
        """
        # Check which environments have shiny objects
        shiny_envs = [loc.get("shiny") is not None for loc in locations]
        if not any(shiny_envs):
            return None

        # Extract shiny indicators from environments with shiny objects
        shiny_locs = torch.stack([torch.tensor(loc["shiny"], dtype=torch.float, device=self.g_init[0].device) for loc in locations if loc["shiny"] is not None]).unsqueeze(-1)

        # Determine number of OVC modules
        n_ovc_modules = self.n_f_ovc if self.separate_ovc else self.n_f

        # Compute mu_g_shiny and sigma_g_shiny via MLPs
        mu_g_shiny = self.mlp_mu_g_shiny([shiny_locs] * n_ovc_modules)
        sigma_g_shiny = self.mlp_sigma_g_shiny([shiny_locs] * n_ovc_modules)

        # Take absolute for object vector cells (legacy: OVCs are positive)
        mu_g_shiny = [torch.abs(mu) for mu in mu_g_shiny]

        # Apply clamp and leaky_relu (like grounded location activation)
        mu_g_shiny = self._apply_grounded_activation(mu_g_shiny)

        # Build full estimate with masking for non-shiny environments
        module_start = self.n_f_g if self.separate_ovc else 0
        batch_size = len(locations)

        # Initialize with zeros for all environments and frequencies
        mu_g_full = [torch.zeros(batch_size, self.n_g[f], device=self.g_init[0].device) for f in range(self.n_f)]
        sigma_g_full = [torch.ones(batch_size, self.n_g[f], device=self.g_init[0].device) * 1e6 for f in range(self.n_f)]  # High uncertainty for non-shiny

        # Fill in shiny estimates only for affected modules
        for f in range(module_start, self.n_f):
            shiny_idx = f - module_start
            mu_g_full[f][shiny_envs, :] = mu_g_shiny[shiny_idx]
            sigma_g_full[f][shiny_envs, :] = sigma_g_shiny[shiny_idx]

        return Transition(mean=mu_g_full, uncertainty=sigma_g_full)

    def _apply_grounded_activation(self, p: List[Tensor]) -> List[Tensor]:
        """Apply grounded location activation (clamp + leaky_relu).

        This matches the legacy f_p activation for sparsity.

        Args:
            p: List of grounded location tensors

        Returns:
            Activated tensors with clamp and leaky_relu applied
        """
        return [torch.nn.functional.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p]
