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

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, GroundedLocation, Transition
from torch_tem.utils.fusion import fuse_transitions, sample_transition


class AbstractLocConfig(BaseModel):
    """Architecture parameters needed by AbstractLocInference.
    ...
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # Architecture parameters
    hidden_multiplier: int = Field(default=20, ge=1, frozen=True, description="Hidden dim = hidden_multiplier * n_o_c")

    #
    g_init_std: float = Field(default=0.5, gt=0, description="Std of initial abstract location g (before learning)")
    g_mem_std: float = Field(default=0.1, gt=0, description="Std for MLP hidden→output weights in g transition network")

    # Inference behavior
    p2g_scale_offset: float = Field(default=0.0, ge=0, description="Variance offset scaling for memory path during abstract inference (controls memory influence)")
    p2g_sig_val: float = Field(default=10000.0, ge=0, description="Base variance magnitude for memory-derived abstract location uncertainty")
    do_sample: bool = Field(default=False, description="If False, use distribution means instead of sampling (no observation noise)")


class AbstractLocInference(nn.Module):
    """Infers abstract location g from multiple sources.

    Sources:
    1. g_gen - Transition prediction (path integration)
    2. p_x -> g_mem - Memory-corrected estimate (inference mode only)
       p_x is the hippocampal pattern retrieved from SENSORY input.
       The quality of p_x (retrieval confidence) determines precision.
    3. shiny objects - Direct inference from environment cues (landmarks)

    Combines via precision-weighted mean (Bayesian fusion).

    Theory:
        In GENERATIVE mode: p_x = None, only path integration is used
        In INFERENCE mode: p_x is retrieved from memory via sensory input,
        allowing correction of path integration drift via loop closure.
    """

    def __init__(self, n_g: List[int], n_p: List[int], config: AbstractLocConfig):
        """Initialize abstract location inference.

        Args:
            ...
        """
        super().__init__()
        self._config = config

        # MLPs for memory-based g inference
        # Project p_x (grounded location from sensory retrieval) to g_mem
        self.mlp_mu_g_mem = MLP(in_dim=[sum(n_p)] * len(n_p), out_dim=n_g, hidden_dim=[config.hidden_multiplier * g for g in n_g])

        # Initialize with small random weights
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * params.g_mem_std for w in self.mlp_mu_g_mem.get_weights(-1)])
        # Uncertainty based on p_x quality (retrieval confidence)
        self.mlp_sigma_g_mem = MLP(in_dim=[1] * self.n_f, out_dim=self.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.n_g])

        # MLPs for shiny object inference (object vector cells)
        n_ovc_modules = self.n_f_ovc if self.separate_ovc else self.n_f
        ovc_start = self.n_f_g if self.separate_ovc else 0
        ovc_dims = self.n_g[ovc_start:]
        self.mlp_mu_g_shiny = MLP(in_dim=[1] * n_ovc_modules, out_dim=ovc_dims, hidden_dim=[config.hidden_multiplier * g for g in ovc_dims])
        self.mlp_sigma_g_shiny = MLP(in_dim=[1] * n_ovc_modules, out_dim=ovc_dims, activation=[torch.tanh, torch.exp], hidden_dim=[config.hidden_multiplier * g for g in ovc_dims])

        # Learnable initial g for new environments
        self.g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * config.g_init_std) for g in n_g])
        self.logsig_g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * config.g_init_std) for g in n_g])

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return len(self.n_g)

    @property
    def n_g(self) -> List[int]:
        """Number of abstract location neurons per frequency."""
        return self.mlp_mu_g_mem.out_dim

    def forward(self, g_gen: Transition, p_x: Optional[GroundedLocation], locations: List[Dict[str, Any]]) -> AbstractLocation:
        """Infer abstract location by fusing path integration with memory.

        Args:
            g_gen: Path integration prediction (always available)
            p_x: Hippocampal pattern retrieved from SENSORY input (inference only)
            locations: Environment descriptors for landmark cues

        Returns:
            Fused abstract location estimate
        """
        # Collect all available estimates
        estimates = [g_gen]  # Source 1: Path integration (always available)

        # Source 2: Memory-corrected estimate (inference mode only)
        if p_x is not None:
            memory_estimate = self._compute_memory_estimate(p_x)
            if memory_estimate is not None:
                estimates.append(memory_estimate)

        # Source 3: Landmark cues (if any)
        if shiny_estimate := self._compute_shiny_estimate(locations):
            estimates.append(shiny_estimate)

        # Fuse all estimates with precision weighting
        fused = fuse_transitions(estimates)

        # Return sampled or mean location
        return sample_transition(fused) if self.do_sample else fused.mean

    def _compute_memory_estimate(self, p_x: GroundedLocation) -> Optional[Transition]:
        """Compute memory-based location estimate from sensory retrieval.

        Process:
            1. Extract: p_x → g_mem (project hippocampal pattern to abstract location)
            2. Assess: ||p_x|| → sigma_g_mem (retrieval confidence)
            3. Schedule: sigma_g_mem += memory_offset (controls influence)

        Args:
            p_x: Hippocampal pattern retrieved from SENSORY input via attractor.
                 High ||p_x|| indicates confident retrieval (familiar pattern).
                 Low ||p_x|| indicates uncertain retrieval (novel pattern).

        Returns:
            Memory estimate with retrieval-dependent uncertainty

        Theory:
            p_x encodes "which memories were activated by this sensory input".
            Those memories have associated abstract locations g_mem.
            The quality of p_x determines how much to trust g_mem for
            correcting path integration drift (loop closure).

            Precision weighting:
                High ||p_x|| → low σ_mem → high precision → strong correction
                Low ||p_x|| → high σ_mem → low precision → weak correction
        """
        # Step 1: Project p_x to abstract location via learned MLP
        # Concatenate all frequency modules of p_x for projection
        p_x_concat = torch.cat(p_x, dim=-1)  # [B, sum(n_p)]
        mu_g_mem = self.mlp_mu_g_mem([p_x_concat] * self.n_f)  # Replicate for each frequency

        # Step 2: Assess retrieval quality from p_x strength
        sigma_g_mem = self._compute_memory_uncertainty(p_x)

        # Step 3: Clamp to valid range (stability)
        mu_g_mem = [torch.clamp(g, -1, 1) for g in mu_g_mem]

        return Transition(mean=mu_g_mem, uncertainty=sigma_g_mem)

    def _compute_memory_uncertainty(self, p_x: GroundedLocation) -> AbstractLocation:
        """Compute memory uncertainty from retrieval quality.

        Quality indicator (per frequency):
            Retrieval confidence: ||p_x||² (strength of memory activation)

        Theory:
            High ||p_x|| → familiar pattern → confident retrieval → LOW uncertainty
            Low ||p_x|| → novel pattern → uncertain retrieval → HIGH uncertainty

            This implements the precision term in Bayesian fusion:
                precision_mem = 1 / (σ_mem² + ε)

        Args:
            p_x: Hippocampal pattern from sensory retrieval

        Returns:
            Memory uncertainty per frequency module
        """
        # Compute retrieval confidence indicator: [||p_x||²] per frequency
        quality_features = [torch.sum(p_f**2, dim=1, keepdim=True) for p_f in p_x]

        # Predict base uncertainty from retrieval quality via MLP
        # High quality → low sigma, Low quality → high sigma
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
