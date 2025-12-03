"""Pattern generation for memory training and generation examples.

This module provides utilities for generating synthetic training patterns
used in memory-based generation examples. It produces realistic place cell
and grid cell activity patterns for training Hebbian memory networks.

Includes both random pattern generators and oscillatory pattern generators
for simulating grid cell dynamics.
"""

from typing import List, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor

from ..types import AbstractLocation, Transition, Vector


class PatternGeneratorParams(Protocol):
    """Protocol for pattern generation configuration.

    Components implementing this protocol provide the necessary parameters
    for generating synthetic training patterns for memory networks.
    """

    n_g: List[int]
    n_p: List[int]
    n_x_c: int


class PlaceCellPatternGenerator:
    """Generates synthetic place cell activity patterns for memory training.

    Creates realistic hippocampal place cell patterns with:
    - Sparse activation (few cells active at once)
    - Smooth probability distributions
    - Biological variability via noise
    - Optional location-specific structure

    These patterns simulate the place cell activity that would be observed
    during spatial navigation, suitable for training Hebbian memory matrices.
    """

    def __init__(
        self,
        params: PatternGeneratorParams,
        sparsity: float = 0.1,
        noise_scale: float = 0.2,
    ):
        """Initialize place cell pattern generator.

        Args:
            params: Configuration providing n_p
            sparsity: Target sparsity level (0.0-1.0, lower = sparser)
            noise_scale: Scale of Gaussian noise for variability
        """
        self.params = params
        self.sparsity = sparsity
        self.noise_scale = noise_scale
        self.n_p_total = sum(params.n_p)

    def generate(self, batch_size: int) -> Vector:
        """Generate batch of place cell activity patterns.

        Creates sparse, normalized activity patterns representing
        hippocampal place cell responses.

        Args:
            batch_size: Number of patterns to generate

        Returns:
            Tensor: [batch_size, n_p_total] place cell patterns
        """
        # Generate sparse base patterns
        patterns = torch.randn(batch_size, self.n_p_total)

        # Apply sparsity via thresholding
        threshold = torch.quantile(patterns, 1.0 - self.sparsity, dim=1, keepdim=True)
        patterns = torch.where(patterns > threshold, patterns, torch.zeros_like(patterns))

        # Add noise for biological variability
        patterns = patterns + self.noise_scale * torch.randn_like(patterns)

        # Normalize to valid probability distribution
        patterns = torch.relu(patterns)  # Ensure non-negative
        patterns = patterns / (patterns.sum(dim=1, keepdim=True) + 1e-8)

        return patterns

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.7) -> Vector:
        """Generate temporally smooth sequence of place cell patterns.

        Creates sequences where adjacent timepoints have similar activity,
        simulating continuous spatial navigation.

        Args:
            n_steps: Number of timesteps
            batch_size: Batch size
            temporal_smoothness: Smoothing factor (0.0-1.0, higher = smoother)

        Returns:
            Tensor: [n_steps, batch_size, n_p_total] pattern sequence
        """
        sequences = []
        prev_pattern = self.generate(batch_size)

        for _ in range(n_steps):
            # Generate new pattern
            new_pattern = self.generate(batch_size)

            # Smooth with previous pattern
            pattern = temporal_smoothness * prev_pattern + (1 - temporal_smoothness) * new_pattern

            # Renormalize
            pattern = pattern / (pattern.sum(dim=1, keepdim=True) + 1e-8)

            sequences.append(pattern)
            prev_pattern = pattern

        return torch.stack(sequences, dim=0)


class GridCellPatternGenerator:
    """Generates synthetic grid cell activity patterns.

    Creates realistic entorhinal grid cell patterns with:
    - Frequency-specific dimensions
    - Random phase offsets for diversity
    - Normalized probability distributions
    - Optional temporal structure

    These patterns simulate grid cell activity that would drive
    place cell responses in the hippocampus.
    """

    def __init__(self, params: PatternGeneratorParams, noise_scale: float = 0.1):
        """Initialize grid cell pattern generator.

        Args:
            params: Configuration providing n_g
            noise_scale: Scale of Gaussian noise for variability
        """
        self.params = params
        self.noise_scale = noise_scale
        self.n_f = len(params.n_g)

    def generate(self, batch_size: int) -> AbstractLocation:
        """Generate batch of grid cell patterns (one per frequency).

        Creates normalized grid cell activity patterns for each
        frequency module.

        Args:
            batch_size: Number of patterns to generate

        Returns:
            List of [batch_size, n_g[f]] tensors
        """
        patterns = []

        for f in range(self.n_f):
            # Generate random pattern for this frequency
            pattern = torch.randn(batch_size, self.params.n_g[f])

            # Add noise
            pattern = pattern + self.noise_scale * torch.randn_like(pattern)

            # Normalize to probability distribution
            pattern = pattern.softmax(dim=1)

            patterns.append(pattern)

        return patterns

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.8) -> List[AbstractLocation]:
        """Generate temporally smooth sequences of grid cell patterns.

        Creates sequences with temporal continuity for each frequency module.

        Args:
            n_steps: Number of timesteps
            batch_size: Batch size
            temporal_smoothness: Smoothing factor (0.0-1.0, higher = smoother)

        Returns:
            List of [n_steps, batch_size, n_g[f]] tensors
        """
        sequences = [[] for _ in range(self.n_f)]
        prev_patterns = self.generate(batch_size)

        for _ in range(n_steps):
            # Generate new patterns
            new_patterns = self.generate(batch_size)

            # Smooth with previous patterns
            patterns = []
            for f in range(self.n_f):
                pattern = temporal_smoothness * prev_patterns[f] + (1 - temporal_smoothness) * new_patterns[f]
                # Renormalize
                pattern = pattern.softmax(dim=1)
                patterns.append(pattern)
                sequences[f].append(pattern)

            prev_patterns = patterns

        # Stack sequences
        return [torch.stack(seq, dim=0) for seq in sequences]


class OscillatoryGridParams(Protocol):
    """Protocol for oscillatory grid cell generation configuration.

    Components implementing this protocol provide the necessary parameters
    for generating oscillatory grid cell patterns during testing and examples.

    This protocol is compatible with Parameters instances from torch_tem.config.parameters.
    """

    n_g: List[int]
    f_extended: List[float]


class OscillatoryGridGenerator:
    """Generates oscillatory grid cell activity patterns.

    Creates oscillating patterns with frequency-dependent dynamics to simulate
    grid cell responses during spatial navigation. Useful for testing inference
    components without requiring full TEM training.

    The generator creates sinusoidal patterns with:
    - Primary oscillation at specified frequency
    - Secondary harmonic for richer dynamics
    - Gaussian noise for biological realism
    - Random phase offsets for variation across cells
    """

    def __init__(
        self,
        params: OscillatoryGridParams,
        walk_length: int,
        batch_size: int = 1,
        time_scale: float = 10.0,
        harmonic_weight: float = 0.3,
        noise_scale: float = 0.2,
        sigma_scale: float = 0.1,
    ):
        """Initialize oscillatory grid generator.

        Args:
            params: Configuration providing n_g, f_extended
            walk_length: Number of timesteps to generate
            batch_size: Batch size for generation
            time_scale: Time scaling factor for oscillations
            harmonic_weight: Weight for second harmonic component (0.0-1.0)
            noise_scale: Scale of Gaussian noise added to patterns
            sigma_scale: Scale for uncertainty estimation (sigma_g)
        """
        self.params = params
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.time_scale = time_scale
        self.harmonic_weight = harmonic_weight
        self.noise_scale = noise_scale
        self.sigma_scale = sigma_scale

        # Validate matching lengths
        if len(self.params.n_g) != len(self.params.f_extended):
            raise ValueError(f"n_g length ({len(self.params.n_g)}) must match " f"f_extended length ({len(self.params.f_extended)})")

    def generate(self) -> List[Transition]:
        """Generate oscillatory grid cell activity patterns.

        Creates oscillating patterns with frequency-dependent dynamics to simulate
        grid cell responses during spatial navigation.

        Returns:
            List of T timesteps, each containing a Transition tuple (g_gen, sigma_g)
            where g_gen and sigma_g are List[n_f] of [B, n_g[f]]
        """
        n_f = len(self.params.n_g)

        # Generate patterns per frequency: [T, B, n_g[f]]
        g_per_freq = []
        sigma_per_freq = []

        for f in range(n_f):
            # Create time axis scaled by frequency
            t = torch.linspace(0, self.time_scale * self.params.f_extended[f], self.walk_length)
            t = t.unsqueeze(1).unsqueeze(2)  # [T, 1, 1]

            # Random phase offsets per cell
            phases = torch.randn(1, self.batch_size, self.params.n_g[f]) * 2 * np.pi  # [1, B, n_g[f]]

            # Combine primary and harmonic oscillations
            pattern = torch.sin(t + phases)
            pattern = pattern + self.harmonic_weight * torch.sin(2 * t + phases * 0.5)

            # Add Gaussian noise for biological realism
            pattern = pattern + self.noise_scale * torch.randn_like(pattern)

            g_per_freq.append(pattern)  # [T, B, n_g[f]]

            # Generate constant uncertainty
            sigma = torch.ones_like(pattern) * self.sigma_scale
            sigma_per_freq.append(sigma)

        # Reorganize to List[T] of Transition
        history = []
        for t in range(self.walk_length):
            g_t = [g_per_freq[f][t] for f in range(n_f)]  # List[n_f] of [B, n_g[f]]
            sigma_t = [sigma_per_freq[f][t] for f in range(n_f)]
            history.append((g_t, sigma_t))

        return history

    def generate_batch(self) -> List[Transition]:
        """Generate single batch of oscillatory grid patterns.

        Convenience method for generating one batch.

        Returns:
            List of T timesteps, each containing a Transition tuple
        """
        return self.generate()


class PairedPatternGenerator:
    """Generates paired (g, p) patterns for memory training.

    Creates associated grid cell and place cell patterns that can be
    used to train Hebbian memory matrices. The patterns have realistic
    structure and optional correlation.
    """

    def __init__(
        self,
        params: PatternGeneratorParams,
        correlation: float = 0.3,
        place_sparsity: float = 0.1,
        noise_scale: float = 0.15,
    ):
        """Initialize paired pattern generator.

        Args:
            params: Configuration providing n_g and n_p
            correlation: Correlation strength between g and p (0.0-1.0)
            place_sparsity: Sparsity for place cell patterns
            noise_scale: Noise scale for both pattern types
        """
        self.params = params
        self.correlation = correlation
        self.grid_gen = GridCellPatternGenerator(params, noise_scale)
        self.place_gen = PlaceCellPatternGenerator(params, place_sparsity, noise_scale)

    def generate(self, batch_size: int) -> Tuple[AbstractLocation, Vector]:
        """Generate paired (g, p) patterns.

        Creates associated grid and place cell patterns with optional
        correlation structure.

        Args:
            batch_size: Number of pattern pairs to generate

        Returns:
            Tuple of:
                - List of [batch_size, n_g[f]] grid cell patterns
                - [batch_size, n_p_total] place cell patterns
        """
        # Generate grid patterns
        g_patterns = self.grid_gen.generate(batch_size)

        # Generate base place patterns
        p_patterns = self.place_gen.generate(batch_size)

        # Add correlation by mixing place patterns with flattened grid patterns
        if self.correlation > 0.0:
            # Flatten grid patterns
            g_flat = torch.cat(g_patterns, dim=1)  # [B, sum(n_g)]

            # Project to place cell space (simple linear combination)
            n_p_total = self.params.n_p
            if g_flat.shape[1] < sum(n_p_total):
                # Repeat and truncate
                g_expanded = g_flat.repeat(1, (sum(n_p_total) // g_flat.shape[1]) + 1)
                g_expanded = g_expanded[:, : sum(n_p_total)]
            else:
                # Average pooling
                g_expanded = torch.nn.functional.adaptive_avg_pool1d(g_flat.unsqueeze(1), sum(n_p_total)).squeeze(1)

            # Mix patterns
            p_patterns = self.correlation * g_expanded + (1 - self.correlation) * p_patterns

            # Renormalize
            p_patterns = torch.relu(p_patterns)
            p_patterns = p_patterns / (p_patterns.sum(dim=1, keepdim=True) + 1e-8)

        return g_patterns, p_patterns

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.7) -> Tuple[List[AbstractLocation], Vector]:
        """Generate temporally smooth sequences of paired patterns.

        Creates correlated sequences of grid and place cell patterns
        with temporal continuity.

        Args:
            n_steps: Number of timesteps
            batch_size: Batch size
            temporal_smoothness: Smoothing factor (0.0-1.0)

        Returns:
            Tuple of:
                - List of [n_steps, batch_size, n_g[f]] grid sequences
                - [n_steps, batch_size, n_p_total] place sequence
        """
        g_sequences = self.grid_gen.generate_sequence(n_steps, batch_size, temporal_smoothness)
        p_sequence = self.place_gen.generate_sequence(n_steps, batch_size, temporal_smoothness)

        return g_sequences, p_sequence


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Example demonstrating pattern generation for memory training."""
    from pydantic import BaseModel

    # Create minimal config implementing PatternGeneratorParams protocol
    class ExampleConfig(BaseModel):
        n_g: List[int] = [30, 25, 20]
        n_p: List[int] = [240, 200, 160]
        n_x_c: int = 8

    print("=" * 80)
    print("Pattern Generation Example")
    print("=" * 80)

    config = ExampleConfig()
    batch_size = 4
    n_steps = 50

    print(f"\nConfiguration:")
    print(f"  Grid dimensions: {config.n_g}")
    print(f"  Place dimensions: {config.n_p}")
    print(f"  Batch size: {batch_size}")

    # Example 1: Place cell patterns
    print(f"\n{'-'*80}")
    print("1. Place Cell Pattern Generation")
    print(f"{'-'*80}")
    place_gen = PlaceCellPatternGenerator(config, sparsity=0.15)
    p_batch = place_gen.generate(batch_size)
    print(f"  Generated: {p_batch.shape}")
    print(f"  Sparsity: {(p_batch > 0.01).float().mean():.2%}")
    print(f"  Mean activation: {p_batch.mean():.6f}")

    # Example 2: Grid cell patterns
    print(f"\n{'-'*80}")
    print("2. Grid Cell Pattern Generation")
    print(f"{'-'*80}")
    grid_gen = GridCellPatternGenerator(config)
    g_batch = grid_gen.generate(batch_size)
    print(f"  Generated: {len(g_batch)} frequency modules")
    for f, g_f in enumerate(g_batch):
        print(f"    Frequency {f}: shape={g_f.shape}, mean={g_f.mean():.6f}")

    # Example 3: Paired patterns
    print(f"\n{'-'*80}")
    print("3. Paired (g, p) Pattern Generation")
    print(f"{'-'*80}")
    paired_gen = PairedPatternGenerator(config, correlation=0.4)
    g_paired, p_paired = paired_gen.generate(batch_size)
    print(f"  Grid patterns: {len(g_paired)} modules")
    print(f"  Place patterns: {p_paired.shape}")
    print(f"  Grid dimensions: {[g.shape for g in g_paired]}")
    print(f"  Total grid cells: {sum(g.shape[1] for g in g_paired)}")
    print(f"  Total place cells: {p_paired.shape[1]}")

    # Example 4: Temporal sequences
    print(f"\n{'-'*80}")
    print("4. Temporal Sequence Generation")
    print(f"{'-'*80}")
    g_seq, p_seq = paired_gen.generate_sequence(n_steps, batch_size, temporal_smoothness=0.8)
    print(f"  Grid sequences: {len(g_seq)} modules")
    for f, g_f_seq in enumerate(g_seq):
        print(f"    Frequency {f}: shape={g_f_seq.shape}")
    print(f"  Place sequence: {p_seq.shape}")

    # Measure temporal smoothness
    p_diff = torch.diff(p_seq, dim=0).norm(dim=2).mean()
    print(f"  Temporal smoothness (avg diff): {p_diff:.4f}")

    print(f"\n{'='*80}")
    print("Pattern generation examples complete!")
    print(f"{'='*80}")
