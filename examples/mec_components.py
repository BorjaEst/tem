#!/usr/bin/env python3
"""MEC components example demonstrating grid cell dynamics and spatial inference.

This example demonstrates the complete Medial Entorhinal Cortex (MEC) pipeline:
- TransitionModel: Predicts next abstract location from current state + action
- AbstractLocInference: Fuses path integration estimates with precision weighting
- Projection: Downsamples and repeats grid cells for hippocampal binding

The MEC pipeline:
  g_t + action → TransitionModel → g_gen (prediction with σ)
  g_gen → AbstractLocInference → g_inf (fused estimate)
  g_inf → Projection → g_downsampled (for memory) + g_repeated (for hippocampus)

Note: This example uses synthetic data and random actions. For LEC sensory
processing, see lec_components.py.

Architecture:
-------------
    MEC TransitionModel (mec.transition.TransitionModel):
        - Input: Current location g_t, action a
        - Output: Predicted next location with uncertainty (g_gen, σ_g_gen)
        - Method: Action-conditioned MLP with hierarchical frequency connections

    MEC AbstractLocInference (mec.abstract.AbstractLocInference):
        - Input: g_gen (path integration with uncertainty)
        - Output: Fused abstract location g_inf [List[n_f] of [B, n_g[f]]]
        - Method: Precision-weighted fusion (inverse variance weighting)
        - Formula: g_inf = Σ(precision_i × g_i) / Σ(precision_i)

    MEC Projection (mec.projection.Projection):
        - Input: Abstract location g
        - Output: g_downsampled (for memory), g_repeated (for hippocampus)
        - Method: Learned downsampling + expansion for conjunctive coding

Usage Examples:
---------------
    # Default: 100 timesteps, 3 frequencies, memory enabled
    python examples/mec_components.py

    # Longer sequence without memory path
    python examples/mec_components.py --walk_length 200 --use_memory false

    # Custom frequency configuration
    python examples/mec_components.py --f_initial "[0.95, 0.7, 0.4]"

    # Show plots interactively
    python examples/mec_components.py --show_plots true --save_plots false

    # Full help
    python examples/mec_components.py --help

Outputs:
--------
When save_plots=true, generates 5 visualizations in outputs/mec_components/:
    1. 01_transition_predictions.png - Action-conditioned transitions across frequencies
    2. 02_uncertainty_evolution.png - Path integration uncertainty dynamics
    3. 03_g_inf_evolution.png - Abstract location (g_inf) evolution
    4. 04_projection_structure.png - Downsampling and expansion structure
    5. 05_grid_patterns.png - Grid cell activity patterns over time
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
from torch_tem.types import Transition


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
        use_memory: Whether to enable memory-based inference (requires p_x)
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
    n_actions: int = Field(default=5, ge=2, le=10, description="Number of possible actions")

    # Source configuration (memory path requires hippocampal representations, not implemented in this demo)
    use_memory: bool = Field(default=False, description="Enable memory-based inference (requires p_x)")

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
        n_actions=config.n_actions,
    )

    print("=" * 80)
    print("Abstract Location Inference Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Sequence length: {config.walk_length} timesteps (synthetic data)")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}")
    print(f"  Memory path: {config.use_memory} (requires hippocampal p_x)")
    print()

    # =========================================================================
    # PHASE 1: Generate Synthetic Actions
    # =========================================================================
    print("Phase 1: Generating synthetic action sequence...")
    # Random actions for demonstration (in real use: from policy/environment)
    actions = torch.randint(0, config.n_actions, (config.walk_length, 1))  # [T, 1]
    actions_onehot = torch.nn.functional.one_hot(actions, config.n_actions).float()  # [T, 1, n_actions]
    print(f"  ✓ Generated {config.walk_length} random actions")
    print()

    # =========================================================================
    # PHASE 2: Initialize MEC Components
    # =========================================================================
    print("Phase 2: Initializing MEC components...")

    # MEC TransitionModel: Predicts next abstract location from current state + action
    # Uses action-conditioned MLP with hierarchical connections between frequencies
    # Outputs prediction with uncertainty: (g_gen, σ_g_gen)
    transition = mec.transition.TransitionModel(model_config)

    # MEC AbstractLocInference: Fuses multiple location estimates with precision weighting
    # Combines path integration (g_gen) with optional memory and shiny signals
    # Implements Bayesian fusion: g_inf = Σ(precision_i × g_i) / Σ(precision_i)
    abstract = mec.abstract.AbstractLocInference(model_config)

    # MEC Projection: Downsamples and expands grid cells for memory and hippocampus
    # g → downsample → g_downsampled (memory indexing)
    # g_downsampled → repeat → g_repeated (hippocampal conjunction)
    projection = mec.projection.Projection(model_config)

    print(f"  ✓ TransitionModel: Action-conditioned dynamics ({config.n_actions} actions)")
    print(f"  ✓ AbstractLocInference: Precision-weighted fusion ({model_config.n_f} frequencies)")
    print(f"  ✓ Projection: Downsample {model_config.n_g} → {model_config.n_g_subsampled}")
    print()

    # =========================================================================
    # PHASE 3: Run MEC Pipeline
    # =========================================================================
    print("Phase 3: Running MEC pipeline (transition → inference → projection)...")

    # Initialize history storage for visualization
    g_inf_history = []  # Fused abstract location over time
    g_gen_history = []  # Path integration predictions over time
    g_downsampled_history = []  # Downsampled for memory
    sigma_history_dict = {"transition": []}  # Uncertainty tracking

    # Initialize with random abstract location (first timestep)
    g_prev = [torch.randn(1, model_config.n_g[f]) * 0.1 for f in range(model_config.n_f)]

    for t in range(config.walk_length):
        # Step 1: Predict next location via TransitionModel
        # g_gen = f(g_prev, action) with uncertainty σ_g_gen
        action_t = actions_onehot[t]  # [1, n_actions]
        g_gen_transition = transition(g_prev, action_t)  # Returns Transition(mean, uncertainty)
        g_gen = g_gen_transition.mean
        sigma_g_gen = g_gen_transition.uncertainty

        # Step 2: Fuse estimates via AbstractLocInference
        # In this demo: single source (path integration only)
        # In full TEM: includes memory (p_x) and shiny signals
        p_x = None  # No memory retrieval in this demo
        location_dicts = [{"shiny": None}]  # No shiny objects
        g_inf = abstract(g_gen_transition, p_x, location_dicts)

        # Step 3: Project for memory and hippocampus
        # Downsample for memory indexing, expand for hippocampal binding
        g_downsampled = projection.downsample(g_inf)

        # Update previous state for next timestep
        g_prev = g_inf

        # Store history (keep batch dimension for plotting)
        g_gen_history.append(g_gen)  # List[n_f] of [1, n_g[f]]
        g_inf_history.append(g_inf)  # List[n_f] of [1, n_g[f]]
        g_downsampled_history.append(g_downsampled)  # List[n_f] of [1, n_g_sub[f]]
        sigma_history_dict["transition"].append([s.mean() for s in sigma_g_gen])

    print(f"  ✓ Processed {config.walk_length} timesteps through MEC pipeline")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Extract frequencies for plotting
    frequencies = model_config.f_initial

    # Plot 1: Grid cell patterns from final timestep
    # plot_grid_cell_patterns expects AbstractLocation (list of tensors per frequency)
    final_g_gen = [g_gen_history[-1][f] for f in range(model_config.n_f)]  # [n_f] x [B, n_g[f]]
    fig1 = figures.plot_grid_cell_patterns(
        patterns=final_g_gen,
        frequencies=frequencies,
        title="Grid Cell Patterns at Final Timestep (t=99)",
        n_samples=4,
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_transition_predictions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_transition_predictions.png")

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

    # Plot 4: Downsampled patterns at final timestep
    final_g_downsampled = [g_downsampled_history[-1][f] for f in range(model_config.n_f)]
    fig4 = figures.plot_grid_cell_patterns(
        patterns=final_g_downsampled,
        frequencies=frequencies,
        title="Downsampled Grid Cells for Memory (t=99)",
        n_samples=4,
    )
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_projection_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_projection_structure.png")

    # Plot 5: Compare g_inf patterns at final timestep
    final_g_inf = [g_inf_history[-1][f] for f in range(model_config.n_f)]
    fig5 = figures.plot_grid_cell_patterns(
        patterns=final_g_inf,
        frequencies=frequencies,
        title="Fused Grid Cell Estimates (g_inf at t=99)",
        n_samples=4,
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_grid_patterns.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_grid_patterns.png")

    print()
    print("=" * 80)
    print("MEC Pipeline Summary")
    print("=" * 80)
    print(f"Input: Initial state g_0 + action sequence ({config.walk_length} timesteps)")
    print(f"  ↓ TransitionModel (action-conditioned dynamics)")
    print(f"Stage 1: g_gen - Predicted location List[{model_config.n_f}] of {model_config.n_g} with σ")
    print(f"  ↓ AbstractLocInference (precision-weighted fusion)")
    print(f"Stage 2: g_inf - Fused estimate List[{model_config.n_f}] of {model_config.n_g}")
    print(f"  ↓ Projection (downsample + expand)")
    print(f"Stage 3: g_downsampled - Memory indexing List[{model_config.n_f}] of {model_config.n_g_subsampled}")
    print(f"         g_repeated - Hippocampal binding (not visualized)")
    print("=" * 80)
    print()
    print("TEM Theory:")
    print("  • MEC provides spatial 'where' information (grid cells g)")
    print("  • TransitionModel: path integration via action-conditioned dynamics")
    print("  • AbstractLocInference: fuses multiple spatial estimates with uncertainty")
    print("  • Projection: prepares for memory storage and hippocampal conjunction")
    print("=" * 80)
    print()
    print(f"All {5 if config.save_plots else 0} visualizations saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
