"""Location generator for torch_tem package.

This module implements the generative path from abstract locations (g) to grounded
locations (p) using associative memory retrieval. It forms the bridge between the
agent's internal abstract representation (grid cells) and the concrete spatial
representation (place cells) that can be decoded to sensory observations.

The LocationGenerator uses Hebbian memory matrices to retrieve place cell activity
patterns from grid cell patterns via attractor dynamics. This process simulates how
the hippocampus might retrieve specific place representations given entorhinal
cortex grid cell input.
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.core.mlp import MLP
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


class LocationGeneratorParams(Protocol):
    """Minimal interface for LocationGenerator.

    Dependencies: n_f, n_p, do_sample
    Complexity: Low (3 parameters)
    """

    do_sample: bool
    n_f: int
    n_p: List[int]


class LocationGenerator(nn.Module):
    """Generates grounded locations (p) from abstract locations (g) via associative memory.

    This component implements the g→p generative pathway in the TEM architecture,
    retrieving hippocampal place cell representations from entorhinal grid cell
    patterns using learned Hebbian associations.

    The generator supports two operational modes:
    1. Deterministic: Returns mean retrieved location (do_sample=False)
    2. Stochastic: Samples from learned uncertainty distribution (do_sample=True)

    Architecture:
        - Memory retrieval via attractor dynamics
        - Optional uncertainty estimation via MLP (when sampling enabled)
        - Supports separate inference/generative memory networks

    Attributes:
        n_f: Number of frequency modules
        n_p: List of place cell dimensions per frequency module
        memory: Hebbian memory storage (M_inf and/or M_gen)
        attractor: Iterative retrieval mechanism
        do_sample: Whether to add learned uncertainty noise
        mlp_sigma_p: MLP for uncertainty estimation (only if do_sample=True)
    """

    def __init__(self, params: LocationGeneratorParams, memory: MemoryStorage, attractor: AttractorDynamics):
        """Initialize location generator.

        Sets up the g→p generative pathway with memory-based retrieval and
        optional uncertainty modeling.

        Args:
            params: Configuration parameters providing:
                - n_f: Number of frequency modules
                - n_p: Place cell dimensions per frequency
                - do_sample: Whether to enable stochastic sampling
            memory: Hebbian memory storage containing learned g-p associations
            attractor: Iterative attractor mechanism for memory retrieval

        Note:
            The uncertainty MLP (mlp_sigma_p) is only created when do_sample=True,
            reducing parameters for deterministic inference.
        """
        super().__init__()
        # Store dimensions from config
        self.n_f = params.n_f
        self.n_p = params.n_p

        # Store component references
        self.memory = memory
        self.attractor = attractor
        self.do_sample = params.do_sample

        # Create uncertainty estimation network (only for stochastic mode)
        # Uses tanh→exp activations to ensure positive standard deviations
        if self.do_sample:
            self.mlp_sigma_p = MLP(
                in_dim=params.n_p,
                out_dim=params.n_p,
                activation=[torch.tanh, torch.exp],
                hidden_dim=[2 * p for p in params.n_p],
            )

    def generate(self, g: List[Tensor], for_inference: bool = False) -> List[Tensor]:
        """Generate grounded location (p) from abstract location (g) via memory retrieval.

        Implements the core g→p transformation using Hebbian associative memory:
        1. Concatenate multi-frequency g into flat query vector
        2. Select appropriate memory matrix (M_inf or M_gen)
        3. Retrieve p via iterative attractor dynamics
        4. Optionally add learned uncertainty noise

        Args:
            g: Abstract location (downsampled grid cells) as list of [n_f] tensors,
               each with shape [B, n_g_subsampled[f]]
            for_inference: If True, use inference memory (M_inf); if False, use
                          generative memory (M_gen). Ignored if common_memory=True.

        Returns:
            p: Retrieved grounded location as list of [n_f] tensors,
               each with shape [B, n_p[f]]. If do_sample=True, includes
               learned uncertainty noise; otherwise returns deterministic mean.

        Mathematical Operation:
            p_mu = AttractorDynamics(g, M)
            p = p_mu + σ(p_mu) * ε  if do_sample, else p_mu
            where ε ~ N(0, I) and σ is learned via MLP
        """
        # Concatenate all frequency modules into single query vector
        # Shape: [B, sum(n_g_subsampled)]
        g_flat = torch.cat(g, dim=1)

        # Select memory matrix based on network mode
        M = self.memory.get_memory(for_inference=for_inference)

        # Retrieve place cell activity via attractor dynamics
        # This iteratively refines the retrieval using: p_t+1 = f(M @ p_t + g)
        p_flat_mu = self.attractor.retrieve(g_flat, M, for_inference=for_inference)

        # Split concatenated result back to per-frequency structure
        p_mu = self._split_to_frequencies(p_flat_mu)

        # Add uncertainty if sampling enabled
        if self.do_sample:
            # Estimate learned uncertainty (standard deviation per dimension)
            p_sigma = self.mlp_sigma_p(p_mu)
            # Sample: p = μ + σ * ε where ε ~ N(0,1)
            p = [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(p_mu, p_sigma)]
        else:
            p = p_mu

        return p

    def _split_to_frequencies(self, p_flat: Tensor) -> List[Tensor]:
        """Split concatenated place cell vector back into per-frequency structure.

        Helper method to reconstruct the hierarchical frequency organization after
        memory retrieval. Each frequency module has different dimensionality based
        on its spatial scale.

        Args:
            p_flat: Concatenated grounded location with shape [B, sum(n_p)]

        Returns:
            p: List of [n_f] tensors, each with shape [B, n_p[f]]

        Example:
            If n_p = [100, 80, 60], n_f = 3:
            p_flat [B, 240] → [p[0] [B,100], p[1] [B,80], p[2] [B,60]]
        """
        p = []
        start_idx = 0
        for f in range(self.n_f):
            end_idx = start_idx + self.n_p[f]
            p.append(p_flat[:, start_idx:end_idx])
            start_idx = end_idx
        return p

    def forward(self, g: List[Tensor], for_inference: bool = False) -> List[Tensor]:
        """Forward pass (alias for generate).

        PyTorch convention: defines the computation performed at every call.
        Simply delegates to generate() for consistent interface.

        Args:
            g: Abstract location (downsampled grid cells)
            for_inference: Whether to use inference or generative memory

        Returns:
            p: Retrieved grounded location (place cells)
        """
        return self.generate(g, for_inference)


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Example demonstrating LocationGenerator for g→p memory-based retrieval.

    Shows how the generator retrieves grounded locations (place cells) from
    abstract locations (grid cells) using learned Hebbian memory associations.
    """
    from torch_tem.config import Parameters
    from torch_tem.memory.attractor import AttractorDynamics
    from torch_tem.memory.storage import MemoryStorage

    # Create configuration with 3 frequency modules
    params = Parameters(
        n_g_subsampled=[10, 10, 10],  # Grid cells per frequency (downsampled)
        n_x_c=5,  # Compressed sensory dimensions
        n_f_g=3,  # Number of frequency modules
        f_initial=[0.99, 0.5, 0.1],  # Frequency scales (high to low)
        eta=0.3,  # Remembering rate
        lambda_=0.95,  # Forgetting rate
        kappa=0.8,  # Attractor decay (retrieval stability)
        do_sample=False,  # Deterministic (no sampling noise)
        use_p_inf=False,  # Only generative memory (no inference memory)
        common_memory=True,  # Single memory matrix
    )

    # Initialize components
    memory = MemoryStorage(params)
    attractor = AttractorDynamics(params)
    generator = LocationGenerator(params, memory, attractor)

    # Train memory with random patterns (simulating spatial experience)
    n_p_total = sum(params.n_p)
    batch_size = 4

    for step in range(30):
        p_patterns = torch.randn(batch_size, n_p_total).softmax(dim=1)
        memory.update(p_patterns, p_patterns, eta=params.eta, lamb=params.lambda_)

    # Generate from abstract location query
    g_test = [torch.randn(2, params.n_p[f]).softmax(dim=1) for f in range(params.n_f)]

    with torch.no_grad():
        p_generated = generator.generate(g_test, for_inference=False)

    print("Generated grounded locations:")
    print(f"  Input: {len(g_test)} frequency modules")
    print(f"  Output: {len(p_generated)} frequency modules")
    print(f"  Dimensions: {params.n_p}")
    print(f"  Mean activation: {torch.cat(p_generated, dim=1).mean():.4f}")

    # Compare deterministic vs stochastic modes
    with torch.no_grad():
        p_det_1 = generator.generate(g_test, for_inference=False)
        p_det_2 = generator.generate(g_test, for_inference=False)

    det_diff = torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(p_det_1, p_det_2)]).mean()

    params_stoch = params.model_copy(update={"do_sample": True})
    gen_stoch = LocationGenerator(params_stoch, memory, attractor)

    with torch.no_grad():
        p_stoch_1 = gen_stoch.generate(g_test, for_inference=False)
        p_stoch_2 = gen_stoch.generate(g_test, for_inference=False)

    stoch_diff = torch.stack([torch.norm(p1 - p2) for p1, p2 in zip(p_stoch_1, p_stoch_2)]).mean()

    print(f"\nMode comparison:")
    print(f"  Deterministic: {det_diff:.6f} (reproducible)")
    print(f"  Stochastic: {stoch_diff:.6f} (adds learned uncertainty)")
