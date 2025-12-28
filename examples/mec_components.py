#!/usr/bin/env python3
"""MEC components example demonstrating spatial processing and inference.

This example demonstrates the Medial Entorhinal Cortex (MEC) components:
- TransitionModel: Predicts next abstract location from current state + action
- AbstractLocInference: Fuses path integration estimates with precision weighting
- Projection: Downsamples and repeats grid cells for hippocampal binding

The MEC pipeline:
  Transition: (g_t, a) → g_gen (with σ)
  Inference: g_gen → g_inf
  Projection: g_inf → g_downsampled (memory) + g_repeated (hippocampus)

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
from typing import List, Optional

import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import data, figures, utils
from torch_tem.core.mec.abstract import AbstractLocConfig, AbstractLocModel
from torch_tem.core.mec.object import ObjectInference, ObjectInferenceConfig
from torch_tem.core.mec.projection import Projection, ProjectionConfig
from torch_tem.core.mec.transition import TransitionConfig, TransitionModel
from torch_tem.types import Matrix


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for MEC components example.

    Defines submodule configurations and output settings for demonstrating
    MEC spatial processing components in isolation with synthetic action sequences.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="mec_components")

    # Learning projection matrices
    learn_W_down: bool = Field(default=False, description="If True, downsampling matrices W_down are learnable")
    learn_W_repeat: bool = Field(default=False, description="If True, expansion matrices W_repeat are learnable")

    # Submodule configurations
    abstract: AbstractLocConfig = Field(default_factory=AbstractLocConfig, description="Abstract location inference configuration")
    transition: TransitionConfig = Field(default_factory=TransitionConfig, description="Transition model configuration")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig, description="Projection configuration")

    # Optional OVC extension
    ovc: Optional[ObjectInferenceConfig] = Field(default=None, description="OVC configuration. None = disabled")

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


# Model architecture; not configurable via CLI
N_G_SUBSAMPLED = [12, 10, 8]  # Downsampled grid cells per frequency (3 modules)
N_F = len(N_G_SUBSAMPLED)  # Number of frequency modules
N_G = [3 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]  # Grid cells per frequency [36, 30, 24]
N_P = [2 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]  # Place cells per frequency [24, 20, 16]
N_ACTIONS = 5  # Number of possible actions
F_INITIAL = [0.9, 0.6, 0.3]  # Initial frequency values
WALK_LENGTH = 100  # Timesteps in sequence
BATCH_SIZE = 4  # Batch size
DEVICE = torch.device("cpu")  # Change to "cuda" if GPU is available

# Build MEC down and repeat from constants
W_down = utils.create_g_downsample(N_G, N_G_SUBSAMPLED)
W_repeat = utils.create_repeat_matrices(N_G_SUBSAMPLED, N_P)


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the MEC spatial processing pipeline experiment with visualizations.

    This script demonstrates the spatial processing pathway:
      1. Generate synthetic action sequences
      2. Initialize MEC components (TransitionModel, AbstractLocModel, Projection)
      3. Process through the pipeline
      4. Generate visualizations of the complete MEC pathway
    """
    config = ExampleConfig()

    print("=" * 80)
    print("MEC Spatial Processing Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Sequence length: {WALK_LENGTH} timesteps")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Frequencies: {N_F} ({F_INITIAL[0]:.2f} to {F_INITIAL[-1]:.2f})")
    print(f"  Architecture: n_g={N_G}, n_g_sub={N_G_SUBSAMPLED}")
    print()

    # =========================================================================
    # PHASE 1: Initialize MEC Components
    # =========================================================================
    print("Phase 1: Initializing MEC components...")

    # MEC TransitionModel: Path integration via action-conditioned dynamics
    # Predicts next abstract location from current state + action
    # Outputs prediction with uncertainty: (g_gen, σ_g_gen)
    transition_model = TransitionModel(n_g=N_G, n_f_grid=N_F, n_actions=N_ACTIONS, f_initial=F_INITIAL, config=config.transition)
    print(f"  ✓ TransitionModel: Action-conditioned dynamics ({N_ACTIONS} actions)")

    # MEC AbstractLocModel: Fuses path integration with memory corrections
    # Combines g_gen (always available) with p_x (memory retrieval, optional)
    # Uses precision-weighted Bayesian fusion
    abstract_model = AbstractLocModel(n_g=N_G, W_repeat=W_repeat, config=config.abstract)
    print(f"  ✓ AbstractLocModel: Precision-weighted fusion ({N_F} frequencies)")

    # MEC Projection: Downsamples and expands grid cells for hippocampal input
    # Downsamples g_inf for memory indexing: g → g_downsampled
    # Expands g_downsampled for hippocampal conjunction: g_downsampled → g_repeated
    # Note: learn_W_down/learn_W_repeat in config are for demonstration; current implementation uses fixed matrices
    projection_model = Projection(W_down=W_down, W_repeat=W_repeat, config=config.projection)
    print(f"  ✓ Projection: {N_G} → {N_G_SUBSAMPLED} → {N_P}")

    # OVC ObjectInference (optional)
    if config.ovc is not None:
        n_g_ovc = [6, 5, 4]  # OVC grid cells per frequency (half of n_g_sub)
        ovc_model = ObjectInference(n_g=N_G, n_g_ovc=n_g_ovc, config=config.ovc)
        print(f"  ✓ ObjectInference: OVC module enabled ({N_F} frequencies)")
    else:
        ovc_model = None
        print(f"  ✓ ObjectInference: OVC module disabled")

    print()

    # =========================================================================
    # PHASE 2: Generate Synthetic Data
    # =========================================================================
    print("Phase 2: Generating synthetic action sequence...")

    # Random actions for demonstration (in real TEM: from policy/environment)
    actions = torch.randint(0, N_ACTIONS, (WALK_LENGTH, BATCH_SIZE))

    print(f"  ✓ Generated {WALK_LENGTH} random actions")
    print()

    # =========================================================================
    # PHASE 3: Run MEC Pipeline
    # =========================================================================
    print("Phase 3: Running MEC pipeline...")
    print("  Pipeline: Transition → Abstract Fusion → Projection")
    print()

    # Initialize history storage
    g_gen_transitions = []  # Transition objects (mean + uncertainty)
    g_inf_history = []  # Fused abstract locations
    g_downsampled_history = []  # Downsampled for memory

    # Initialize abstract location (first timestep)
    g_prev = [torch.randn(BATCH_SIZE, N_G[f], device=DEVICE) * 0.1 for f in range(N_F)]
    valid_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    for t in range(WALK_LENGTH):
        # === Step 1: Transition (Path Integration) ===
        # Predict next location from current state + action
        action_t = actions[t]  # [B]
        g_gen_transition = transition_model(g_prev, action_t, valid_mask)
        g_gen = g_gen_transition.mean  # List[n_f] of [B, n_g[f]]

        # === Step 2: Abstract Location Fusion ===
        # Fuse path integration with optional memory correction
        # Note: AbstractLocModel always works, p_x can be None
        p_x = None  # In this example we don't use memory path

        g_inf = abstract_model(g_gen_transition, p_x)  # List[n_f] of [B, n_g[f]]

        # === Step 3: Projection ===
        # Downsample for memory, expand for hippocampus
        g_downsampled = projection_model.downsample(g_inf)  # List[n_f] of [B, n_g_sub[f]]

        # Update state for next timestep
        g_prev = g_inf

        # Store history
        g_gen_transitions.append(g_gen_transition)
        g_inf_history.append([g.clone().detach() for g in g_inf])
        g_downsampled_history.append([g.clone().detach() for g in g_downsampled])

        # Progress reporting
        if (t + 1) % 20 == 0 or t == 0:
            avg_sigma = sum(s.mean().item() for s in g_gen_transition.uncertainty) / len(g_gen_transition.uncertainty)
            print(f"  Step {t+1}/{WALK_LENGTH}: avg uncertainty={avg_sigma:.4f}")

    print(f"  ✓ Processed {WALK_LENGTH} timesteps through MEC pipeline")
    print()

    # =========================================================================
    # PHASE 4: Generate Visualizations
    # =========================================================================
    print("Phase 4: Generating visualizations...")

    # Plot 1: Transition predictions (grid cell patterns)
    fig1 = figures.plot_grid_cell_patterns(
        patterns=g_gen_transitions[-1].mean, frequencies=F_INITIAL, title="Grid Cell Patterns from Transition Model (Final Timestep)", n_samples=min(4, BATCH_SIZE)
    )
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_transition_predictions.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_transition_predictions.png")

    # Plot 2: Uncertainty evolution
    fig2 = figures.plot_transition_uncertainty(transitions=g_gen_transitions, frequencies=F_INITIAL, title="Path Integration Uncertainty Evolution")
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_uncertainty_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 02_uncertainty_evolution.png")

    # Plot 3: g_inf evolution
    fig3 = figures.plot_g_inf_evolution(g_inf_history=g_inf_history, n_frequencies=N_F)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_g_inf_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_g_inf_evolution.png")

    # Plot 4: Projection structure (downsampled grid cells)
    fig4 = figures.plot_grid_cell_patterns(
        patterns=g_downsampled_history[-1], frequencies=F_INITIAL, title="Downsampled Grid Cells for Memory Indexing (Final Timestep)", n_samples=min(4, BATCH_SIZE)
    )
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_projection_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_projection_structure.png")

    # Plot 5: Grid temporal evolution
    # Convert history to tensor format: List[T] of List[n_f] of [B, n_g[f]]
    # to List[n_f] of [T, B, n_g[f]]
    g_inf_sequences = []
    for f in range(N_F):
        g_f_seq = torch.stack([g_inf_history[t][f] for t in range(WALK_LENGTH)])  # [T, B, n_g[f]]
        g_inf_sequences.append(g_f_seq)

    fig5 = figures.plot_grid_temporal_evolution(g_sequences=g_inf_sequences, frequencies=F_INITIAL, title="Multi-frequency Grid Cell Temporal Evolution", n_cells_per_freq=5)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_grid_patterns.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_grid_patterns.png")

    print()
    print("=" * 80)
    print("MEC Pipeline Summary")
    print("=" * 80)
    print(f"Architecture:")
    print(f"  Grid cells: {N_G} ({N_F} frequencies)")
    print(f"  Downsampled: {N_G_SUBSAMPLED}")
    print(f"  Place cells: {N_P}")
    print()
    print(f"Pipeline Flow:")
    print(f"  1. TransitionModel: (g_t, a) → (g_gen, σ_gen)")
    print(f"     - Hierarchical connections between {N_F} frequency modules")
    print(f"     - Action-conditioned dynamics via MLP")
    print()
    print(f"  2. AbstractLocModel: (g_gen, p_x?) → g_inf")
    print(f"     - Precision-weighted fusion")
    print(f"     - Memory path: DISABLED (p_x=None in this example)")
    print()
    print(f"  3. Projection: g_inf → (g_down, g_repeat)")
    print(f"     - Downsample: {N_G} → {N_G_SUBSAMPLED} (memory indexing)")
    print(f"     - Expand: {N_G_SUBSAMPLED} → {N_P} (hippocampal conjunction)")
    print()
    print("=" * 80)
    print()
    print("TEM Theory:")
    print("  • MEC provides spatial 'where' information via grid cells")
    print("  • Path integration maintains spatial coherence during navigation")
    print("  • Memory corrections reduce drift through Hebbian associations")
    print("  • Hierarchical frequencies capture multiple spatial scales")
    print("=" * 80)
    print()
    if config.save_plots:
        print(f"All 5 visualizations saved to: {config.output_dir}")
    else:
        print("Plots not saved (use --save_plots true to save)")

    # Show or close plots
    if config.show_plots:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close("all")
