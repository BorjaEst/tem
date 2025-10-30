#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration classes for the TEM model.
Replaces dictionary-based configuration with typed dataclasses.
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class ArchitectureConfig:
    """Neural architecture dimensions and structure."""

    # Sensory dimensions
    n_x: int = 45  # Sensory observation neurons
    n_x_c: int = 10  # Compressed sensory neurons

    # Module organization
    n_g_subsampled: List[int] = field(default_factory=lambda: [10, 10, 8, 6, 6])
    f_initial: List[float] = field(default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01])

    # Object vector cells
    n_ovc: List[int] = field(default_factory=list)
    separate_ovc: bool = False

    # Action space
    n_actions: int = 4
    has_static_action: bool = True

    # Network sizes
    d_hidden_dim: int = 20
    g_init_std: float = 0.5
    g_mem_std: float = 0.1

    @property
    def n_f_g(self) -> int:
        """Number of grid cell frequency modules."""
        if not self.n_ovc or not self.separate_ovc:
            return len(self.n_g_subsampled)
        return len(self.n_g_subsampled)

    @property
    def n_f_ovc(self) -> int:
        """Number of object vector cell frequency modules."""
        if not self.n_ovc or not self.separate_ovc:
            return 0
        return len(self.n_ovc)

    @property
    def n_f(self) -> int:
        """Total number of frequency modules."""
        return len(self.n_g_subsampled_combined)

    @property
    def n_g_subsampled_combined(self) -> List[int]:
        """Combined grid cell and OVC neurons."""
        if not self.n_ovc:
            return self.n_g_subsampled
        if self.separate_ovc:
            return self.n_g_subsampled + self.n_ovc
        return [grid + ovc for grid, ovc in zip(self.n_g_subsampled, self.n_ovc)]

    @property
    def n_g(self) -> List[int]:
        """Abstract location neurons per frequency (3x subsampled)."""
        return [3 * g for g in self.n_g_subsampled_combined]

    @property
    def n_x_f(self) -> List[int]:
        """Filtered sensory neurons per frequency."""
        return [self.n_x_c for _ in range(self.n_f)]

    @property
    def n_p(self) -> List[int]:
        """Grounded location neurons per frequency."""
        return [g * x for g, x in zip(self.n_g_subsampled_combined, self.n_x_f)]

    @property
    def f_initial_extended(self) -> List[float]:
        """Frequencies extended with OVC modules."""
        if self.separate_ovc and self.n_ovc:
            return self.f_initial + self.f_initial[0 : self.n_f_ovc]
        return self.f_initial


@dataclass
class MemoryConfig:
    """Hebbian memory system parameters."""

    # Hebbian learning rates
    lambda_: float = 0.9999  # Forgetting rate
    eta: float = 0.5  # Remembering rate
    kappa: float = 0.8  # Retrieval decay

    # Attractor dynamics
    i_attractor: int = 5  # Attractor iterations

    # Memory architecture
    common_memory: bool = False  # Shared generative/inference memory
    use_hierarchical: bool = True  # Hierarchical connections

    # Early stopping per frequency (computed from architecture)
    i_attractor_max_freq_inf: List[int] = field(default_factory=list)
    i_attractor_max_freq_gen: List[int] = field(default_factory=list)


@dataclass
class ModelConfig:
    """High-level model behavior configuration."""

    do_sample: bool = False  # Sample vs mean
    use_p_inf: bool = True  # Use memory-based inference

    # Training-related parameters
    p2g_scale_offset: float = 0.0  # Variance offset scaling
    p2g_sig_val: float = 10000.0  # Variance offset value


@dataclass
class StaticMatrices:
    """Pre-computed static matrices for efficient computation."""

    W_repeat: List[torch.Tensor]  # For outer product (g side)
    W_tile: List[torch.Tensor]  # For outer product (x side)
    g_downsample: List[torch.Tensor]  # Downsample g to g_subsampled
    two_hot_table: List[torch.Tensor]  # One-hot to two-hot compression
    p_update_mask: torch.Tensor  # Hierarchical memory mask
    p_retrieve_mask_inf: List[torch.Tensor]  # Retrieval masks (inference)
    p_retrieve_mask_gen: List[torch.Tensor]  # Retrieval masks (generative)
    g_connections: List[List[bool]]  # Grid cell connectivity

    @staticmethod
    def create(arch_config: ArchitectureConfig, mem_config: MemoryConfig) -> "StaticMatrices":
        """Create static matrices from configuration."""
        from scipy.special import comb

        n_f = arch_config.n_f
        n_g_subsampled = arch_config.n_g_subsampled_combined
        n_x_f = arch_config.n_x_f
        n_p = arch_config.n_p
        n_g = arch_config.n_g
        f_initial = arch_config.f_initial_extended

        # W_repeat and W_tile for outer products
        W_repeat = [torch.tensor(np.kron(np.eye(n_g_subsampled[f]), np.ones((1, n_x_f[f]))), dtype=torch.float) for f in range(n_f)]
        W_tile = [torch.tensor(np.kron(np.ones((1, n_g_subsampled[f])), np.eye(n_x_f[f])), dtype=torch.float) for f in range(n_f)]

        # Downsample matrices
        g_downsample = [
            torch.cat([torch.eye(n_g_subsampled[f], dtype=torch.float), torch.zeros((n_g[f] - n_g_subsampled[f], n_g_subsampled[f]), dtype=torch.float)]) for f in range(n_f)
        ]

        # Two-hot compression table
        two_hot_table = [[0] * (arch_config.n_x_c - 2) + [1] * 2]
        for i in range(1, min(int(comb(arch_config.n_x_c, 2)), arch_config.n_x)):
            code = two_hot_table[-1].copy()
            swap = [index for index in range(len(code) - 1, -1, -1) if code[index : index + 2] == [0, 1]][0]
            code[swap : swap + 2] = [1, 0]
            if swap + 2 < len(code) and code[swap + 2] == 1:
                code[swap + 2 :] = code[: swap + 1 : -1]
            two_hot_table.append(code)
        two_hot_table = [torch.tensor(code, dtype=torch.float) for code in two_hot_table]

        # Hierarchical memory update mask
        p_update_mask = torch.zeros((sum(n_p), sum(n_p)), dtype=torch.float)
        n_p_cumsum = np.cumsum(np.concatenate(([0], n_p)))

        for f_from in range(n_f):
            for f_to in range(n_f):
                if f_from > arch_config.n_f_g or f_to > arch_config.n_f_g:
                    if f_from > arch_config.n_f_g and f_to > arch_config.n_f_g:
                        if f_initial[f_from] <= f_initial[f_to]:
                            p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
                    else:
                        p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0
                else:
                    if f_initial[f_from] <= f_initial[f_to]:
                        p_update_mask[n_p_cumsum[f_from] : n_p_cumsum[f_from + 1], n_p_cumsum[f_to] : n_p_cumsum[f_to + 1]] = 1.0

        # Retrieval masks for early stopping
        i_attractor_max_freq_inf = [mem_config.i_attractor for _ in range(n_f)]
        i_attractor_max_freq_gen = [mem_config.i_attractor - freq_nr for freq_nr in range(arch_config.n_f_g)] + [mem_config.i_attractor for _ in range(arch_config.n_f_ovc)]

        p_retrieve_mask_inf = [torch.zeros(sum(n_p)) for _ in range(mem_config.i_attractor)]
        p_retrieve_mask_gen = [torch.zeros(sum(n_p)) for _ in range(mem_config.i_attractor)]

        for mask, max_iters in zip([p_retrieve_mask_inf, p_retrieve_mask_gen], [i_attractor_max_freq_inf, i_attractor_max_freq_gen]):
            for f, max_i in enumerate(max_iters):
                for i in range(max_i):
                    mask[i][n_p_cumsum[f] : n_p_cumsum[f + 1]] = 1.0

        # Grid cell connectivity
        g_connections = [
            [f_initial[f_from] <= f_initial[f_to] for f_from in range(arch_config.n_f_g)] + [False for _ in range(arch_config.n_f_ovc)] for f_to in range(arch_config.n_f_g)
        ]
        g_connections += [
            [False for _ in range(arch_config.n_f_g)] + [f_initial[f_from] <= f_initial[f_to] for f_from in range(arch_config.n_f_g, n_f)] for f_to in range(arch_config.n_f_g, n_f)
        ]

        return StaticMatrices(
            W_repeat=W_repeat,
            W_tile=W_tile,
            g_downsample=g_downsample,
            two_hot_table=two_hot_table,
            p_update_mask=p_update_mask,
            p_retrieve_mask_inf=p_retrieve_mask_inf,
            p_retrieve_mask_gen=p_retrieve_mask_gen,
            g_connections=g_connections,
        )
