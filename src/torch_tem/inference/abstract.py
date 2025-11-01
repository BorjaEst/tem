"""Abstract location inference for torch_tem package."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.config.facets import AbstractInferenceParams
from torch_tem.core.mlp import MLP


class AbstractLocationInference(nn.Module):
    """Infers abstract location g from multiple sources.

    Sources:
    1. g_gen - Transition prediction
    2. p_x -> g - Memory-based inference (if use_p_inf)
    3. g_shiny - Shiny object signals (if present)

    Combines via precision-weighted mean.
    """

    def __init__(self, params: AbstractInferenceParams):
        """Initialize abstract location inference.

        Args:
            params: Configuration satisfying AbstractInferenceParams protocol
        """
        super().__init__()
        self.n_f = params.n_f_calculated
        self.n_g = params.n_g_calculated
        self.n_g_subsampled = list(params.n_g_subsampled_combined)
        self.use_p_inf = params.use_p_inf

        # MLPs for memory-based g inference
        self.mlp_mu_g_mem = MLP(in_dim=self.n_g_subsampled, out_dim=self.n_g, hidden_dim=[2 * g for g in self.n_g])
        # Initialize with small random weights
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * params.g_mem_std for w in self.mlp_mu_g_mem.get_weights(-1)])

        self.mlp_sigma_g_mem = MLP(in_dim=[2] * self.n_f, out_dim=self.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.n_g])

        # Learnable initial g for new environments
        self.g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * params.g_init_std) for g in self.n_g])
        self.logsig_g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * params.g_init_std) for g in self.n_g])

    def forward(
        self, g_gen: List[Tensor], sigma_g_gen: List[Tensor], p_x: Optional[List[Tensor]], shiny_signals: Optional[Tuple[List[Tensor], List[Tensor]]], p2g_scale_offset: float
    ) -> List[Tensor]:
        """Infer g via precision-weighted mean of sources.

        Args:
            g_gen: Transition prediction
            sigma_g_gen: Transition uncertainty
            p_x: Memory retrieval from sensory (if use_p_inf)
            shiny_signals: (mu_g_shiny, sigma_g_shiny) if present
            p2g_scale_offset: Schedule for p->g influence

        Returns:
            g_inf: Inferred abstract location
        """
        # Source 1: Transition
        sources_mu = [g_gen]
        sources_sigma = [sigma_g_gen]

        # Source 2: Memory (if use_p_inf)
        if self.use_p_inf and p_x is not None:
            # Compute mean from memory
            mu_g_mem = self.mlp_mu_g_mem(p_x)

            # Compute uncertainty based on memory quality
            memory_quality = self._compute_memory_quality(p_x, mu_g_mem)
            sigma_g_mem = self.mlp_sigma_g_mem(memory_quality)

            sources_mu.append(mu_g_mem)
            sources_sigma.append(self._scale_sigma(sigma_g_mem, p2g_scale_offset))

        # Source 3: Shiny
        if shiny_signals is not None:
            mu_g_shiny, sigma_g_shiny = shiny_signals
            sources_mu.append(mu_g_shiny)
            sources_sigma.append(sigma_g_shiny)

        # Precision-weighted mean
        return self._precision_weighted_mean(sources_mu, sources_sigma)

    def _compute_memory_quality(self, p_x: List[Tensor], mu_g_mem: List[Tensor]) -> List[Tensor]:
        """Compute memory quality indicators for uncertainty estimation.

        Returns a 2D signal per frequency indicating:
        - Norm of inferred abstract location (good memories have similar norms)
        - Reconstruction error (not yet implemented, placeholder with zeros)

        Args:
            p_x: Projected grounded location per frequency
            mu_g_mem: Inferred abstract location mean per frequency

        Returns:
            Quality signals [n_f] of [B, 2]
        """
        quality_signals = []

        for f in range(self.n_f):
            # Signal 1: Vector norm of inferred abstract location
            g_norm = torch.sum(mu_g_mem[f] ** 2, dim=1, keepdim=True)

            # Signal 2: Reconstruction error (placeholder - would need decoder)
            # In original: err = squared_error(x, x_hat) where x_hat = gen_x(p_x[0])
            # For now, use zeros as we don't have x available here
            batch_size = p_x[f].shape[0]
            recon_error = torch.zeros(batch_size, 1, device=p_x[f].device)

            quality_signals.append(torch.cat([g_norm, recon_error], dim=1))

        return quality_signals

    def _precision_weighted_mean(self, means: List[List[Tensor]], sigmas: List[List[Tensor]]) -> List[Tensor]:
        """Combine sources via precision weighting.

        precision_i = 1 / sigma_i^2
        g_inf = sum(precision_i * mu_i) / sum(precision_i)

        Args:
            means: List of mean lists (one per source)
            sigmas: List of sigma lists (one per source)

        Returns:
            g_inf: Precision-weighted mean
        """
        g_inf = []
        for f in range(self.n_f):
            precisions = [1.0 / (sigma[f] ** 2 + 1e-8) for sigma in sigmas]
            weighted_sum = sum(p * mu[f] for p, mu in zip(precisions, means))
            precision_sum = sum(precisions)
            g_inf.append(weighted_sum / precision_sum)
        return g_inf

    def _scale_sigma(self, sigma: List[Tensor], offset: float) -> List[Tensor]:
        """Scale sigma for scheduling p->g influence.

        Args:
            sigma: Uncertainty estimates
            offset: Scheduling offset value

        Returns:
            Scaled sigma
        """
        return [s + offset for s in sigma]
