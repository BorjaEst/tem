"""Abstract location inference for the Tolman–Eichenbaum Machine (TEM).

This module computes the abstract location representation (``g``) by
fusing multiple information sources with uncertainty-aware (precision)
weighting. It is the integration hub between predictive dynamics and
episodic memory:

Sources combined per frequency module f:
1) Transition prediction: ``g_gen[f]`` with uncertainty ``sigma_g_gen[f]``
2) Memory-based inference (optional): ``p_x[f] -> g`` via a learned MLP
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
- ``p_x[f]`` (if used): [B, n_g_subsampled[f]] → mapped to ``mu_g_mem[f]``
- Returns ``g_inf[f]``: [B, n_g[f]]

This implementation follows the style/patterns of other TEM modules and
is designed to be testable with simple parameter stubs.
"""

from typing import List, Optional, Protocol, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP


class ArchitectureParams(Protocol):
    """Architecture parameters needed by AbstractLocationInference."""

    n_f: int
    n_g: List[int]
    n_g_subsampled_combined: List[int]
    g_init_std: float
    g_mem_std: float


class InferenceParams(Protocol):
    """Inference parameters needed by AbstractLocationInference."""

    use_p_inf: bool


class AbstractLocationInference(nn.Module):
    """Infers abstract location g from multiple sources.

    Sources:
    1. g_gen - Transition prediction
    2. p_x -> g - Memory-based inference (if use_p_inf)
    3. g_shiny - Shiny object signals (if present)

    Combines via precision-weighted mean.
    """

    def __init__(self, arch_params: ArchitectureParams, inf_params: InferenceParams):
        """Initialize abstract location inference.

        Args:
            arch_params: Architecture configuration (n_f, n_g, n_g_subsampled_combined, g_init_std, g_mem_std)
            inf_params: Inference configuration (use_p_inf)
        """
        super().__init__()
        self.n_f = arch_params.n_f
        self.n_g = arch_params.n_g
        self.n_g_subsampled = list(arch_params.n_g_subsampled_combined)
        self.use_p_inf = inf_params.use_p_inf

        # MLPs for memory-based g inference
        self.mlp_mu_g_mem = MLP(in_dim=self.n_g_subsampled, out_dim=self.n_g, hidden_dim=[2 * g for g in self.n_g])
        # Initialize with small random weights
        self.mlp_mu_g_mem.set_weights(-1, [torch.randn_like(w) * arch_params.g_mem_std for w in self.mlp_mu_g_mem.get_weights(-1)])

        self.mlp_sigma_g_mem = MLP(in_dim=[2] * self.n_f, out_dim=self.n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in self.n_g])

        # Learnable initial g for new environments
        self.g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * arch_params.g_init_std) for g in self.n_g])
        self.logsig_g_init = nn.ParameterList([nn.Parameter(torch.randn(g) * arch_params.g_init_std) for g in self.n_g])

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


if __name__ == "__main__":
    """Minimal usage example (smoke test).

    Demonstrates precision-weighted fusion of transition and memory paths.

    This example builds a lightweight parameter stub implementing the
    fields accessed by ``AbstractLocationInference`` and runs a forward
    pass with synthetic inputs.
    """
    import types

    # Configuration stub with required fields
    params = types.SimpleNamespace(
        n_f=2,
        n_g=[10, 8],  # abstract dims per frequency
        n_g_subsampled_combined=[6, 5],  # downsampled dims per frequency
        use_p_inf=True,  # enable memory path
        g_mem_std=0.01,  # small init for memory MLP
        g_init_std=0.1,  # init scale for learnable g_init
    )

    model = AbstractLocationInference(params)

    # Synthetic inputs
    B = 3
    n_f = params.n_f
    n_g = params.n_g
    n_g_sub = params.n_g_subsampled_combined

    # Transition prediction and its uncertainty
    g_gen = [torch.randn(B, n_g[f]) for f in range(n_f)]
    sigma_g_gen = [torch.exp(torch.randn(B, n_g[f])) for f in range(n_f)]

    # Memory-based input (projected from p-space to reduced g-space)
    p_x = [torch.randn(B, n_g_sub[f]) for f in range(n_f)]

    # No shiny cues in this minimal example
    shiny = None

    # Schedule offset: larger values down-weight memory early in training
    p2g_scale_offset = 0.1

    with torch.no_grad():
        g_inf = model(g_gen, sigma_g_gen, p_x, shiny, p2g_scale_offset)

    print("AbstractLocationInference Example")
    print("=" * 72)
    print(f"Frequencies: {n_f}")
    for f in range(n_f):
        print(f"  f={f}: g_gen {tuple(g_gen[f].shape)} | " f"sigma {tuple(sigma_g_gen[f].shape)} | " f"p_x {tuple(p_x[f].shape)} -> g_inf {tuple(g_inf[f].shape)}")
    print("✓ Forward pass completed.")
