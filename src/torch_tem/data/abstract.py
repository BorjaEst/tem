"""Synthetic data generation for abstract location inference testing and examples.

This module provides utilities for generating synthetic abstract inference signals
used in abstract location inference examples. It produces realistic transition
predictions, memory signals, and shiny (salient object) signals for testing
precision-weighted fusion without requiring full TEM training.
"""

from typing import List, Optional, Protocol, Tuple

import numpy as np
import torch
from torch import Tensor

from ..types import AbstractLocation, Vector


class AbstractInferenceParams(Protocol):
    """Protocol for abstract inference synthetic data generation configuration.

    Components implementing this protocol provide the necessary parameters
    for generating synthetic abstract inference signals during testing and examples.
    """

    n_timesteps: int
    batch_size: int
    n_frequencies: int
    n_g: List[int]
    n_g_subsampled_combined: List[int]
    transition_sigma_base: float
    memory_sigma_base: float
    shiny_sigma_base: float


class TransitionPredictionGenerator:
    """Generates synthetic transition predictions with uncertainty.

    Creates realistic transition-based abstract location predictions (g_gen)
    with associated uncertainty estimates (sigma_g_gen). Simulates predictive
    dynamics with slowly varying means and time-varying confidence.

    The generator uses Ornstein-Uhlenbeck-like dynamics to create temporally
    coherent predictions that mimic real transition model outputs.
    """

    def __init__(
        self,
        params: AbstractInferenceParams,
        decay_rate: float = 0.95,
        noise_scale: float = 0.05,
        uncertainty_modulation: float = 0.3,
        uncertainty_period: float = 20.0,
    ):
        """Initialize transition prediction generator.

        Args:
            params: Configuration providing timesteps, batch_size, n_frequencies, n_g, transition_sigma_base
            decay_rate: Temporal decay for OU dynamics (0-1, higher = slower evolution)
            noise_scale: Innovation noise scale for dynamics
            uncertainty_modulation: Amplitude of uncertainty variation (0-1)
            uncertainty_period: Period of uncertainty oscillation in timesteps
        """
        self.params = params
        self.decay_rate = decay_rate
        self.noise_scale = noise_scale
        self.uncertainty_modulation = uncertainty_modulation
        self.uncertainty_period = uncertainty_period

    def generate(self) -> Tuple[List[AbstractLocation], List[AbstractLocation]]:
        """Generate synthetic transition predictions with uncertainty.

        Creates g_gen and sigma_g_gen sequences following Ornstein-Uhlenbeck
        dynamics with time-varying uncertainty to simulate prediction confidence.

        Returns:
            (g_gen_history, sigma_gen_history): Tuple of lists of length T, each
            containing lists of n_f tensors [B, n_g[f]]
        """
        T = self.params.n_timesteps
        B = self.params.batch_size
        n_f = self.params.n_frequencies
        n_g = self.params.n_g

        g_gen_history = []
        sigma_gen_history = []

        # Initialize with random state
        g_prev = [torch.randn(B, n_g[f]) for f in range(n_f)]

        for t in range(T):
            # Slowly evolving transition prediction (Ornstein-Uhlenbeck-like)
            g_gen = [self.decay_rate * g_prev[f] + self.noise_scale * torch.randn(B, n_g[f]) for f in range(n_f)]

            # Time-varying uncertainty (simulate varying prediction confidence)
            uncertainty_factor = 1.0 + self.uncertainty_modulation * torch.sin(torch.tensor(t / self.uncertainty_period))
            sigma_gen = [self.params.transition_sigma_base * uncertainty_factor * torch.ones(B, n_g[f]) for f in range(n_f)]

            g_gen_history.append(g_gen)
            sigma_gen_history.append(sigma_gen)
            g_prev = g_gen

        return g_gen_history, sigma_gen_history


class MemorySignalGenerator:
    """Generates synthetic memory-based inference signals.

    Creates realistic memory retrieval patterns (p_x) that simulate signals
    from attractor dynamics or Hebbian memory networks. Patterns have temporal
    correlation and frequency-dependent structure.
    """

    def __init__(
        self,
        params: AbstractInferenceParams,
        amplitude_base: float = 0.8,
        amplitude_modulation: float = 0.4,
        modulation_period: float = 15.0,
    ):
        """Initialize memory signal generator.

        Args:
            params: Configuration providing timesteps, batch_size, n_frequencies, n_g_subsampled_combined
            amplitude_base: Base amplitude for memory signals
            amplitude_modulation: Amplitude of temporal modulation
            modulation_period: Period of amplitude oscillation in timesteps
        """
        self.params = params
        self.amplitude_base = amplitude_base
        self.amplitude_modulation = amplitude_modulation
        self.modulation_period = modulation_period

    def generate(self) -> List[AbstractLocation]:
        """Generate synthetic memory-based signals.

        Creates p_x sequences with temporal correlation simulating memory
        retrieval patterns from attractor dynamics.

        Returns:
            p_x_history: List of length T, each containing lists of n_f
            tensors [B, n_g_sub[f]]
        """
        T = self.params.n_timesteps
        B = self.params.batch_size
        n_f = self.params.n_frequencies
        n_g_sub = self.params.n_g_subsampled_combined

        p_x_history = []

        for t in range(T):
            # Memory patterns with temporal correlation
            amplitude = self.amplitude_base + self.amplitude_modulation * np.sin(t / self.modulation_period)
            p_x = [torch.randn(B, n_g_sub[f]) * amplitude for f in range(n_f)]
            p_x_history.append(p_x)

        return p_x_history


class ShinySignalGenerator:
    """Generates synthetic salient object (shiny) signals.

    Creates intermittent strong localization cues with low uncertainty,
    simulating attention-grabbing spatial signals from salient objects
    in the environment.
    """

    def __init__(
        self,
        params: AbstractInferenceParams,
        signal_period: int = 20,
        signal_duration: int = 5,
        signal_strength: float = 2.0,
    ):
        """Initialize shiny signal generator.

        Args:
            params: Configuration providing timesteps, batch_size, n_frequencies, n_g, shiny_sigma_base
            signal_period: Timesteps between signal appearances
            signal_duration: Duration of each signal appearance in timesteps
            signal_strength: Multiplicative strength of signal (relative to noise)
        """
        self.params = params
        self.signal_period = signal_period
        self.signal_duration = signal_duration
        self.signal_strength = signal_strength

    def generate(self) -> Tuple[List[Optional[AbstractLocation]], List[Optional[AbstractLocation]]]:
        """Generate synthetic salient object signals.

        Creates intermittent strong signals (mu_shiny, sigma_shiny) with
        low uncertainty, appearing periodically to simulate salient objects.

        Returns:
            (mu_shiny_history, sigma_shiny_history): Tuple of lists of length T,
            each containing lists of n_f tensors [B, n_g[f]] when signal is
            present, or None when signal is absent
        """
        T = self.params.n_timesteps
        B = self.params.batch_size
        n_f = self.params.n_frequencies
        n_g = self.params.n_g

        mu_shiny_history = []
        sigma_shiny_history = []

        for t in range(T):
            if (t % self.signal_period) < self.signal_duration:
                # Strong signal with low uncertainty
                mu_shiny = [torch.randn(B, n_g[f]) * self.signal_strength for f in range(n_f)]
                sigma_shiny = [self.params.shiny_sigma_base * torch.ones(B, n_g[f]) for f in range(n_f)]
                mu_shiny_history.append(mu_shiny)
                sigma_shiny_history.append(sigma_shiny)
            else:
                mu_shiny_history.append(None)
                sigma_shiny_history.append(None)

        return mu_shiny_history, sigma_shiny_history


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Example demonstrating abstract inference synthetic data generation."""
    from pydantic import BaseModel

    # Create minimal config implementing AbstractInferenceParams protocol
    class ExampleConfig(BaseModel):
        n_timesteps: int = 100
        batch_size: int = 4
        n_frequencies: int = 3
        n_g: List[int] = [10, 8, 6]
        n_g_subsampled_combined: List[int] = [6, 5, 4]
        transition_sigma_base: float = 0.5
        memory_sigma_base: float = 0.3
        shiny_sigma_base: float = 0.2

    print("=" * 80)
    print("Abstract Inference Synthetic Data Generation Example")
    print("=" * 80)

    config = ExampleConfig()

    print(f"\nConfiguration:")
    print(f"  Timesteps: {config.n_timesteps}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Frequencies: {config.n_frequencies}")
    print(f"  Abstract dims: {config.n_g}")
    print(f"  Subsampled dims: {config.n_g_subsampled_combined}")

    # Example 1: Transition predictions
    print(f"\n{'-'*80}")
    print("1. Transition Prediction Generation")
    print(f"{'-'*80}")

    trans_gen = TransitionPredictionGenerator(config)
    g_gen_history, sigma_gen_history = trans_gen.generate()

    print(f"Generated {len(g_gen_history)} timesteps")
    print(f"Sample g_gen[0]:")
    for f in range(config.n_frequencies):
        print(f"  Frequency {f}: shape={g_gen_history[0][f].shape}, " f"mean={g_gen_history[0][f].mean():.4f}, std={g_gen_history[0][f].std():.4f}")
    print(f"Sample sigma_gen[0]:")
    for f in range(config.n_frequencies):
        print(f"  Frequency {f}: shape={sigma_gen_history[0][f].shape}, " f"mean={sigma_gen_history[0][f].mean():.4f}")

    # Example 2: Memory signals
    print(f"\n{'-'*80}")
    print("2. Memory Signal Generation")
    print(f"{'-'*80}")

    mem_gen = MemorySignalGenerator(config)
    p_x_history = mem_gen.generate()

    print(f"Generated {len(p_x_history)} timesteps")
    print(f"Sample p_x[0]:")
    for f in range(config.n_frequencies):
        print(f"  Frequency {f}: shape={p_x_history[0][f].shape}, " f"mean={p_x_history[0][f].mean():.4f}, std={p_x_history[0][f].std():.4f}")

    # Example 3: Shiny signals
    print(f"\n{'-'*80}")
    print("3. Shiny Signal Generation")
    print(f"{'-'*80}")

    shiny_gen = ShinySignalGenerator(config)
    mu_shiny_history, sigma_shiny_history = shiny_gen.generate()

    n_active = sum(1 for x in mu_shiny_history if x is not None)
    print(f"Generated {len(mu_shiny_history)} timesteps ({n_active} active signals)")

    # Find first active timestep
    active_idx = next(i for i, x in enumerate(mu_shiny_history) if x is not None)
    print(f"Sample mu_shiny[{active_idx}] (first active):")
    for f in range(config.n_frequencies):
        print(
            f"  Frequency {f}: shape={mu_shiny_history[active_idx][f].shape}, "
            f"mean={mu_shiny_history[active_idx][f].mean():.4f}, "
            f"std={mu_shiny_history[active_idx][f].std():.4f}"
        )

    print(f"\n{'='*80}")
    print("All generators completed successfully!")
    print(f"{'='*80}")
