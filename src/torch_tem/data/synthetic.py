"""Synthetic grid cell activity generation for testing and examples."""

from typing import List, Protocol

import numpy as np
import torch
from torch import Tensor


class SyntheticGridParams(Protocol):
    """Protocol for synthetic grid cell generation configuration.

    Components implementing this protocol provide the necessary parameters
    for generating synthetic grid cell patterns during testing and examples.

    This protocol is compatible with Parameters instances from torch_tem.config.parameters.
    """

    n_g_calculated: List[int]
    f_initial_extended: List[float]


class SyntheticGridGenerator:
    """Generates synthetic grid cell activity patterns.

    Creates oscillating patterns with frequency-dependent dynamics to simulate
    grid cell responses during spatial navigation. Useful for testing inference
    components without requiring full TEM training.

    The generator creates sinusoidal patterns with:
    - Primary oscillation at specified frequency
    - Secondary harmonic for richer dynamics
    - Gaussian noise for biological realism
    - Random phase offsets for variation across cells
    """

    def __init__(self, params: SyntheticGridParams, walk_length: int, batch_size: int = 1, time_scale: float = 10.0, harmonic_weight: float = 0.3, noise_scale: float = 0.2):
        """Initialize synthetic grid generator.

        Args:
            params: Configuration providing n_g_calculated, f_initial_extended
            walk_length: Number of timesteps to generate
            batch_size: Batch size for generation
            time_scale: Time scaling factor for oscillations
            harmonic_weight: Weight for second harmonic component (0.0-1.0)
            noise_scale: Scale of Gaussian noise added to patterns
        """
        self.params = params
        self.walk_length = walk_length
        self.batch_size = batch_size
        self.time_scale = time_scale
        self.harmonic_weight = harmonic_weight
        self.noise_scale = noise_scale

        # Validate matching lengths
        if len(self.params.n_g_calculated) != len(self.params.f_initial_extended):
            raise ValueError(f"n_g_calculated length ({len(self.params.n_g_calculated)}) must match " f"f_initial_extended length ({len(self.params.f_initial_extended)})")

    def generate(self) -> List[List[Tensor]]:
        """Generate synthetic grid cell activity patterns.

        Creates oscillating patterns with frequency-dependent dynamics to simulate
        grid cell responses during spatial navigation.

        Returns:
            List of T timesteps, each containing a list of n_f tensors [B, n_g[f]]
            This matches the standard convention: List[T] of List[n_f] of [B, n_g[f]]
        """
        n_f = len(self.params.n_g_calculated)

        # Generate patterns per frequency: [T, B, n_g[f]]
        g_per_freq = []
        for f in range(n_f):
            # Create time axis scaled by frequency
            t = torch.linspace(0, self.time_scale * self.params.f_initial_extended[f], self.walk_length)
            t = t.unsqueeze(1).unsqueeze(2)  # [T, 1, 1]

            # Random phase offsets per cell
            phases = torch.randn(1, self.batch_size, self.params.n_g_calculated[f]) * 2 * np.pi  # [1, B, n_g[f]]

            # Combine primary and harmonic oscillations
            pattern = torch.sin(t + phases)
            pattern = pattern + self.harmonic_weight * torch.sin(2 * t + phases * 0.5)

            # Add Gaussian noise for biological realism
            pattern = pattern + self.noise_scale * torch.randn_like(pattern)

            g_per_freq.append(pattern)  # [T, B, n_g[f]]

        # Reorganize to List[T] of List[n_f] of [B, n_g[f]]
        g_history = []
        for t in range(self.walk_length):
            g_t = [g_per_freq[f][t] for f in range(n_f)]  # List[n_f] of [B, n_g[f]]
            g_history.append(g_t)

        return g_history

    def generate_batch(self) -> List[List[Tensor]]:
        """Generate single batch of synthetic grid patterns.

        Convenience method for generating one batch.

        Returns:
            List of T timesteps, each with n_f tensors [B, n_g[f]]
        """
        return self.generate()


# Example usage for standalone testing
if __name__ == "__main__":
    from pydantic import BaseModel

    # Create a minimal config implementing SyntheticGridParams protocol
    class ExampleConfig(BaseModel):
        n_g_calculated: List[int] = [12, 10, 8]
        f_initial_extended: List[float] = [0.1, 0.3, 0.9]

    print("=" * 80)
    print("Synthetic Grid Cell Generator Example")
    print("=" * 80)

    # Configuration
    config = ExampleConfig()
    walk_length = 50
    batch_size = 2

    print(f"\nConfiguration:")
    print(f"  Steps: {walk_length}")
    print(f"  Frequencies: {config.f_initial_extended}")
    print(f"  Grid dimensions: {config.n_g_calculated}")
    print(f"  Batch size: {batch_size}")

    # Generate patterns
    generator = SyntheticGridGenerator(config, walk_length=walk_length, batch_size=batch_size)
    g_history = generator.generate()

    print(f"\nGenerated patterns:")
    print(f"  Total timesteps: {len(g_history)}")
    print(f"  Frequencies per timestep: {len(g_history[0])}")
    for f in range(len(g_history[0])):
        g_f_shape = g_history[0][f].shape
        # Collect stats across all timesteps for this frequency
        g_f_values = torch.stack([g_history[t][f] for t in range(len(g_history))])
        print(f"  Frequency {f} (f={config.f_initial_extended[f]:.2f}): shape=[B, n_g]={g_f_shape}, " f"mean={g_f_values.mean():.4f}, std={g_f_values.std():.4f}")
