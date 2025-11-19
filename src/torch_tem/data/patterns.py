"""Pattern generation for memory training and generation examples.

This module provides utilities for generating synthetic training patterns
used in memory-based generation examples. It produces realistic place cell
and grid cell activity patterns for training Hebbian memory networks.
"""

from typing import List, Protocol, Tuple

import torch
from torch import Tensor


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

    def generate(self, batch_size: int) -> Tensor:
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

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.7) -> Tensor:
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

    def generate(self, batch_size: int) -> List[Tensor]:
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

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.8) -> List[Tensor]:
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

    def generate(self, batch_size: int) -> Tuple[List[Tensor], Tensor]:
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

    def generate_sequence(self, n_steps: int, batch_size: int, temporal_smoothness: float = 0.7) -> Tuple[List[Tensor], Tensor]:
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
