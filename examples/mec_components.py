#!/usr/bin/env python3
"""MEC abstract location inference example demonstrating transition and fusion components.

This example demonstrates the Medial Entorhinal Cortex (MEC) components from torch_tem.mec
using entirely synthetic data to isolate MEC behavior from sensory processing:
- TransitionModel: Predicts next abstract location (g_gen with uncertainty)
- AbstractLocInference: Fuses multiple location estimates with precision weighting
- Hierarchical frequency modules with g_connections for information flow
- Memory-based location inference via synthetic g_downsampled patterns
- Precision-weighted Bayesian fusion of path integration and memory

The MEC pathway focuses on abstract location dynamics:
  Synthetic g_gen (transition prediction) + Optional g_downsampled (memory)
  → AbstractLocInference → g_inf (fused location)

Note: This example uses synthetic data for both grid cell patterns (g_gen) and
memory-based patterns (g_downsampled when enabled). For sensory processing (LEC),
see lec_components.py. For full TEM pipeline, see tem_inference.py.

Architecture:
-------------
    MEC TransitionModel (mec.transition.TransitionModel):
        - Input: Current location g_t [List[n_f] of [B, n_g[f]]], action a [B, n_actions]
        - Output: Predicted location g_gen with uncertainty (Transition)
        - Method: Action-conditioned dynamics via learned MLP, hierarchical connections
        - Note: In this demo, we use pre-generated synthetic transitions

    MEC AbstractLocInference (mec.abstract.AbstractLocInference):
        - Input: g_downsampled (optional memory), g_gen (transition), locations (env state)
        - Output: Fused abstract location g_inf [List[n_f] of [B, n_g[f]]]
        - Method: Precision-weighted fusion (inverse variance weighting)
        - Sources: (1) Path integration (g_gen), (2) Memory (g_downsampled), (3) Shiny objects

Usage Examples:
---------------
    # Default: 100 timesteps, 3 frequencies, memory enabled
    python examples/mec_components.py

    # Longer sequence without memory path
    python examples/mec_components.py --walk_length 200 --use_p_inf false

    # Custom frequency configuration and scheduling
    python examples/mec_components.py --f_initial "[0.95, 0.7, 0.4]" --p2g_schedule_start 3.0

    # Show plots interactively
    python examples/mec_components.py --show_plots true --save_plots false

    # Full help
    python examples/mec_components.py --help

Outputs:
--------
When save_plots=true, generates 4 visualizations in outputs/mec_components/:
    1. 01_source_contributions.png - Precision weights over time (transition vs memory)
    2. 02_uncertainty_evolution.png - Uncertainty dynamics per source
    3. 03_g_inf_evolution.png - Abstract location evolution across frequencies
    4. 04_schedule_effect.png - Memory influence scheduling (p2g offset)
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, mec, utils
from torch_tem.config import ModelConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for MEC abstract location inference example.

    Defines architecture parameters for demonstrating MEC components
    in isolation using entirely synthetic data (no sensory processing).

    Attributes:
        walk_length: Number of timesteps in the synthetic sequence
        f_initial: Base frequency values for each grid cell module
        n_g_subsampled: Number of downsampled grid cells per frequency
        use_p_inf: Whether to generate synthetic g_downsampled for memory-based inference
        p2g_schedule_start: Initial memory uncertainty offset (high = low influence)
        p2g_schedule_end: Final memory uncertainty offset (low = high influence)
        output_dir: Directory for saving visualization outputs
        show_plots: Whether to display plots interactively
        save_plots: Whether to save plots to disk
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="mec_components")

    # Sequence generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the synthetic sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")

    # Source configuration
    use_p_inf: bool = Field(default=True, description="Enable synthetic memory-based inference (g_downsampled)")

    # Scheduling
    p2g_schedule_start: float = Field(default=2.0, ge=0.0, le=5.0, description="Initial p2g scale offset (high = low memory influence)")
    p2g_schedule_end: float = Field(default=0.5, ge=0.0, le=2.0, description="Final p2g scale offset (low = high memory influence)")

    # Output
    output_dir: Path = Field(default=Path("outputs/mec_components"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the MEC abstract location inference pipeline experiment with visualizations.

    This script demonstrates the complete abstract location inference pathway:
      1. Generate synthetic grid cell patterns (g_gen with uncertainty)
      2. Initialize MEC TransitionModel and AbstractLocInference components
      3. Process patterns through the abstract inference pipeline
      4. Generate visualizations of fusion and precision weighting effects
    """
    config = ExampleConfig()

    # Create model config using ModelConfig
    model_config = ModelConfig(
        n_g_subsampled=config.n_g_subsampled,
        f_initial=config.f_initial,
        p2g_scale_offset=config.p2g_schedule_start,  # Will be updated per timestep
        p2g_sig_val=10000.0,  # Standard value for memory uncertainty magnitude
    )

    print("=" * 80)
    print("Abstract Location Inference Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Sequence length: {config.walk_length} timesteps (synthetic data)")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_g_subsampled={model_config.n_g_subsampled}")
    print(f"  Memory path (use_p_inf): {config.use_p_inf}")
    print(f"  p2g schedule: {config.p2g_schedule_start:.2f} → {config.p2g_schedule_end:.2f}")
    print()

    # =========================================================================
    # PHASE 1: Generate Synthetic Grid Cell Patterns
    # =========================================================================
    print("Phase 1: Generating synthetic grid cell patterns...")
    grid_generator = data.OscillatoryGridGenerator(model_config, config.walk_length, batch_size=1, sigma_scale=0.5)
    transition_history = grid_generator.generate()  # List[T] of Transition (g, sigma)
    print(f"  ✓ Generated {config.walk_length} timesteps of grid cell activity (g_gen)")
    print()

    # =========================================================================
    # PHASE 2: Initialize MEC Components
    # =========================================================================
    print("Phase 2: Initializing MEC components...")

    # MEC TransitionModel: Predicts next abstract location from current state
    # Generates predictions with uncertainty estimates for path integration
    # Uses hierarchical g_connections for multi-frequency interactions
    transition = mec.transition.TransitionModel(model_config)

    # MEC AbstractLocInference: Fuses multiple location estimates with precision weighting
    # Combines path integration (g_gen), memory (g_downsampled), and shiny signals
    # Implements Bayesian fusion: g_inf = Σ(precision_i × g_i) / Σ(precision_i)
    abstract = mec.abstract.AbstractLocInference(model_config)

    # MEC Projection: Projects abstract location to hippocampus (not used in this example)
    # Downsamples grid cell patterns for memory retrieval and hippocampal binding
    # Repeats to match hippocampal dimensions for conjunctive coding to improve performance
    projection = mec.projection.Projection(model_config)

    print(f"  ✓ TransitionModel: Action-based dynamics with hierarchical g_connections")
    print(f"  ✓ AbstractLocInference: Precision-weighted fusion ({model_config.n_f} frequencies)")
    print()

    # =========================================================================
    # PHASE 3: Generate Synthetic Memory Patterns (Optional)
    # =========================================================================
    if config.use_p_inf:
        print("Phase 3: Generating synthetic memory patterns (g_downsampled)...")
        # Generate synthetic g_downsampled patterns for memory-based inference
        # In a real scenario: sensory → hippocampus → memory retrieval → g_downsampled
        # Here we create random patterns that vary over time
        g_downsampled_history = [[torch.randn(1, model_config.n_g_subsampled[f]) * 0.2 for f in range(model_config.n_f)] for t in range(config.walk_length)]
        print(f"  ✓ Generated {config.walk_length} timesteps of synthetic g_downsampled patterns")
    else:
        g_downsampled_history = [None] * config.walk_length
        print("Phase 3: Skipping memory pattern generation (memory path disabled)")
    print()

    # =========================================================================
    # PHASE 4: Generate p2g Schedule
    # =========================================================================
    print("Phase 4: Setting up p2g schedule...")
    p2g_schedule = np.linspace(config.p2g_schedule_start, config.p2g_schedule_end, config.walk_length)
    print(f"  ✓ p2g schedule: {p2g_schedule[0]:.2f} → {p2g_schedule[-1]:.2f}")
    print()

    # =========================================================================
    # PHASE 5: Run MEC Inference Pipeline
    # =========================================================================
    print("Phase 5: Running MEC abstract location inference...")

    # Initialize history storage for visualization
    g_inf_history = []  # Fused abstract location over time
    g_gen_history = []  # Path integration estimates over time
    precisions_history = []  # Precision weights for each source
    sigma_history_dict = {"transition": [], "memory": []}  # Uncertainty tracking

    # Initialize previous fused state (zero for first timestep)
    g_prev = [torch.randn(1, model_config.n_g[f]) * 0.1 for f in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Update p2g_scale_offset for this timestep (scheduling)
        model_config.p2g_scale_offset = p2g_schedule[t]

        # Step 1: Path integration estimate from transition model
        # TransitionModel generates predicted location with uncertainty
        # In a real scenario: g_gen_transition = transition(g_prev, action)
        g_gen_transition = transition_history[t]  # Transition(mean, uncertainty)
        g_gen = g_gen_transition.mean  # Extract mean for storage
        sigma_g_gen = g_gen_transition.uncertainty  # Extract uncertainty

        # Step 2: Memory-based location estimate (optional)
        # AbstractLocInference can incorporate memory-retrieved patterns
        # In a real scenario: sensory → hippocampus → memory → g_downsampled
        g_downsampled = g_downsampled_history[t]  # Synthetic patterns or None

        # Step 3: Precision-weighted fusion via AbstractLocInference
        # Combines path integration (g_gen), memory (g_downsampled), and shiny signals
        # Precision weighting: precision_i = 1 / (σ_i² + ε)
        location_dicts = [{"shiny": None}]  # Single environment, no shiny objects
        g_inf = abstract(g_downsampled, g_gen_transition, location_dicts)  # List[n_f] of [1, n_g[f]]

        # Update previous fused state for next timestep
        g_prev = g_inf

        # Store history (extract batch dimension for single-trajectory storage)
        g_gen_history.append([g[0] for g in g_gen])  # List[n_f] of [1, n_g[f]] → List[n_f] of [n_g[f]]
        g_inf_history.append([g[0] for g in g_inf])  # List[n_f] of [1, n_g[f]] → List[n_f] of [n_g[f]]

        # Track precision contributions (for visualization)
        # Precision = 1 / variance, where variance = sigma^2
        precision_transition = [1.0 / (sigma.mean() ** 2 + 1e-8) for sigma in sigma_g_gen]
        if config.use_p_inf and g_downsampled is not None:
            # Memory uncertainty is controlled by p2g scheduling
            # Lower uncertainty = higher precision = more influence
            sigma_mem_base = 0.1  # Base uncertainty for memory
            sigma_mem_scheduled = sigma_mem_base + model_config.p2g_scale_offset * model_config.p2g_sig_val
            precision_memory = [1.0 / (sigma_mem_scheduled**2 + 1e-8)] * model_config.n_f
        else:
            precision_memory = [0.0] * model_config.n_f  # No memory influence

        # Normalize precisions to get relative contributions
        total_precision = [pt + pm for pt, pm in zip(precision_transition, precision_memory)]
        precisions_history.append(
            {
                "transition": [pt / (tp + 1e-8) for pt, tp in zip(precision_transition, total_precision)],
                "memory": [pm / (tp + 1e-8) for pm, tp in zip(precision_memory, total_precision)],
            }
        )

        # Store uncertainty values for visualization
        sigma_history_dict["transition"].append([s.mean().item() for s in sigma_g_gen])
        if config.use_p_inf:
            sigma_history_dict["memory"].append([sigma_mem_scheduled] * model_config.n_f)
        else:
            sigma_history_dict["memory"].append([float("inf")] * model_config.n_f)  # Infinite uncertainty = no influence

    print(f"  ✓ Processed {config.walk_length} timesteps through abstract inference pipeline")
    print()

    # =========================================================================
    # PHASE 6: Generate Visualizations
    # =========================================================================
    print("Phase 6: Generating visualizations...")

    # Plot 1: Source precision contributions
    timesteps_to_plot = [0, config.walk_length // 4, config.walk_length // 2, 3 * config.walk_length // 4, config.walk_length - 1]
    fig1 = figures.plot_source_contributions(precisions_history, timesteps_to_plot, model_config.n_f)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_source_contributions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_source_contributions.png")

    # Plot 2: Uncertainty evolution
    fig2 = figures.plot_uncertainty_evolution(sigma_history_dict, model_config.n_f)
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_uncertainty_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_uncertainty_evolution.png")

    # Plot 3: g_inf evolution
    fig3 = figures.plot_g_inf_evolution(g_inf_history, model_config.n_f)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_g_inf_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_g_inf_evolution.png")

    # Plot 4: p2g schedule effect
    fig4 = figures.plot_schedule_effect(p2g_schedule)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_schedule_effect.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_schedule_effect.png")

    print()
    print("=" * 80)
    print("Pipeline Summary:")
    print("=" * 80)
    print(f"Input: Synthetic grid cell patterns - {config.walk_length} timesteps")
    print(f"Stage 1: Path integration estimate (g_gen)")
    print(f"         List[{model_config.n_f}] of {model_config.n_g} with uncertainty σ_transition")
    print(f"Stage 2: Memory estimate (g_downsampled) {'[ENABLED]' if config.use_p_inf else '[DISABLED]'}")
    if config.use_p_inf:
        print(f"         List[{model_config.n_f}] of {model_config.n_g_subsampled} (synthetic patterns)")
    print(f"  ↓ AbstractLocInference (precision-weighted fusion)")
    print(f"    • Source 1: Path integration (g_gen with σ_transition)")
    print(f"    • Source 2: Memory (g_downsampled with σ_memory) {'[ENABLED]' if config.use_p_inf else '[DISABLED]'}")
    print(f"    • Source 3: Shiny signals [DISABLED in this demo]")
    print(f"    • Fusion: g_inf = Σ(precision_i × g_i) / Σ(precision_i)")
    print(f"    • Precision: precision_i = 1 / (σ_i² + ε)")
    print(f"    • Scheduling: σ_memory += p2g_offset × p2g_sig_val")
    print(f"Output: Fused abstract location (g_inf) - List[{model_config.n_f}] of {model_config.n_g}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
