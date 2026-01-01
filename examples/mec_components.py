#!/usr/bin/env python3
"""MEC components example demonstrating spatial processing and inference.

This example demonstrates the Medial Entorhinal Cortex (MEC) components:
- TransitionModel: Predicts next abstract location from current state + action
- AbstractLocModel: Fuses path integration estimates with precision weighting
- ObjectInference: Object vector cells for landmark recognition (optional)
- Projection: Downsamples and repeats grid cells for hippocampal binding

The MEC pipeline:
  Transition: (g_t, a) → g_gen (with σ)
  Abstract: (g_gen, p_x?) → g_inf
  [OVC: (g_gen, locations) → g_ovc (optional)]
  Projection: g_inf → g_downsampled (memory) + g_repeated (hippocampus)

Note: This example uses synthetic data and random actions. Transition weights are
initialized with small random values to demonstrate dynamics (untrained model).
For LEC sensory processing, see lec_components.py.

Architecture:
-------------
    MEC TransitionModel (mec.transition.TransitionModel):
        - Input: Current location g_t, action a
        - Output: Predicted next location with uncertainty (g_gen, σ_g_gen)
        - Method: Action-conditioned MLP with hierarchical frequency connections

    MEC AbstractLocModel (mec.abstract.AbstractLocModel):
        - Input: g_gen (path integration with uncertainty), optional p_x (memory)
        - Output: Fused abstract location g_inf [List[n_f] of [B, n_g[f]]]
        - Method: Precision-weighted fusion (inverse variance weighting)
        - Formula: g_inf = Σ(precision_i × g_i) / Σ(precision_i)

    MEC ObjectInference (mec.object.ObjectInference):
        - Input: g_gen (fallback), locations (environment with 'shiny' field)
        - Output: Object vector cells [List[n_f_ovc] of [B, n_g_ovc[f]]] or []
        - Method: Landmark-based OVC activation with learned MLPs

    MEC Projection (mec.projection.Projection):
        - Input: Abstract location g
        - Output: g_downsampled (for memory), g_repeated (for hippocampus)
        - Method: Learned downsampling + expansion for conjunctive coding

Usage Examples:
---------------
    # Default: 100 timesteps, 4 total modules (all grid)
    python examples/mec_components.py

    # Enable OVC in merged mode (4 grid modules with OVC portions)
    python examples/mec_components.py --ovc.n_g_ovc "[10, 10, 10, 10]"

    # Enable OVC in separate mode (3 grid + 1 separate OVC module)
    python examples/mec_components.py --ovc.n_g_ovc "[10, 10, 10]" --ovc.frequencies "[0.25]"

    # Enable sampling for stochastic dynamics
    python examples/mec_components.py --transition.do_sample true --abstract.do_sample true

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

Note: OVC (object vector cells) can be enabled with --ovc.n_g_ovc but operates
in merged mode (integrated with grid cells) and doesn't produce separate visualization
in this component-level demo.
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


# ==============================================================================
# Model Architecture - Fixed Total Constants
# ==============================================================================
# These define TOTAL module capacity; OVC config determines grid vs OVC split

N_G_SUBSAMPLED = [12, 10, 8, 10]  # Downsampled cells per frequency (4 total: max 3 grid + 1 OVC)
N_G_MAX = [3 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]  # Maximum grid cells [36, 30, 24, 30]
N_P_MAX = [2 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]  # Maximum place cells [24, 20, 16, 20]
N_ACTIONS = 5  # Number of possible actions
F_INITIAL = [0.9, 0.6, 0.3, 0.25]  # Frequency values for all modules
WALK_LENGTH = 100  # Timesteps in sequence
BATCH_SIZE = 4  # Batch size
DEVICE = torch.device("cpu")  # Change to "cuda" if GPU is available

# Build projection matrices ONCE from maximum capacity - never rebuild
# Models use resolve_dimensions() internally to compute actual n_g based on OVC config
W_down = utils.create_downsample_matrix(N_G_MAX, N_G_SUBSAMPLED)
W_repeat = utils.create_repeat_matrices(N_G_SUBSAMPLED, N_P_MAX)


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the MEC spatial processing pipeline experiment with visualizations.

    This script demonstrates the spatial processing pathway:
      1. Generate synthetic action sequences and optional landmarks
      2. Initialize MEC components (TransitionModel, AbstractLocModel, Projection, OVC)
      3. Process through the pipeline with randomized transition dynamics
      4. Generate visualizations of the complete MEC pathway
    """
    config = ExampleConfig()

    # Compute actual dimensions using resolve_dimensions()
    # Create temporary context with fixed matrices
    from dataclasses import dataclass

    from torch_tem.core.mec import MECConfig, resolve_dimensions
    from torch_tem.core.mec.object import ObjectInferenceConfig

    @dataclass
    class TempContext:
        """Temporary context for dimension resolution."""

        n_g: List[int]
        n_a: int
        f_initial: List[float]
        W_down: List[torch.Tensor]
        W_repeat: List[torch.Tensor]

    context = TempContext(
        n_g=N_G_MAX,
        n_a=N_ACTIONS,
        f_initial=F_INITIAL,
        W_down=W_down,
        W_repeat=W_repeat,
    )

    mec_config = MECConfig(
        transition=config.transition,
        abstract=config.abstract,
        projection=config.projection,
        ovc=config.ovc if config.ovc is not None else ObjectInferenceConfig(),
    )

    dims = resolve_dimensions(context, mec_config)

    # Extract computed dimensions
    n_g = dims.n_g
    n_p = dims.n_p
    n_f_grid = len(dims.n_g_grid)
    n_f_ovc_separate = len(config.ovc.frequencies) if config.ovc and config.ovc.frequencies else 0

    # Rebuild projection matrices with actual dimensions
    # (necessary when OVC changes the dimensions)
    W_down = utils.create_downsample_matrix(n_g, N_G_SUBSAMPLED)
    W_repeat = utils.create_repeat_matrices(N_G_SUBSAMPLED, n_p)

    print("=" * 80)
    print("MEC Spatial Processing Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Sequence length: {WALK_LENGTH} timesteps")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total modules: {len(n_g)} ({n_f_grid} grid + {n_f_ovc_separate} separate OVC)")
    print(f"  Frequencies: {F_INITIAL}")
    print(f"  Max capacity: n_g_max={N_G_MAX}, n_p_max={N_P_MAX}")
    print(f"  Actual dimensions: n_g={n_g}, n_p={n_p}")
    print(f"  OVC config: n_g_ovc={config.ovc.n_g_ovc if config.ovc else []}, frequencies={config.ovc.frequencies if config.ovc else []}")
    print()

    # =========================================================================
    # PHASE 1: Initialize MEC Components
    # =========================================================================
    print("Phase 1: Initializing MEC components...")

    # MEC TransitionModel: Path integration via action-conditioned dynamics
    # Predicts next abstract location from current state + action
    # Outputs prediction with uncertainty: (g_gen, σ_g_gen)
    transition_model = TransitionModel(n_g=n_g, n_f_grid=n_f_grid, n_actions=N_ACTIONS, f_initial=F_INITIAL, config=config.transition)

    # Initialize transition weights with small random values for demonstration
    # (In real TEM, these would be learned through training)
    with torch.no_grad():
        for param in transition_model.MLP_D_a.parameters():
            if param.dim() > 1:  # Weight matrices
                param.data = torch.randn_like(param) * 0.05  # Small random initialization

    print(f"  ✓ TransitionModel: Action-conditioned dynamics ({N_ACTIONS} actions)")
    print(f"    Note: Weights initialized randomly for demonstration (untrained model)")

    # MEC AbstractLocModel: Fuses path integration with memory corrections
    # Combines g_gen (always available) with p_x (memory retrieval, optional)
    # Uses precision-weighted Bayesian fusion (grid modules only)
    abstract_model = AbstractLocModel(n_g=n_g[:n_f_grid], W_repeat=W_repeat[:n_f_grid], config=config.abstract)
    print(f"  ✓ AbstractLocModel: Precision-weighted fusion ({n_f_grid} grid frequencies)")

    # MEC Projection: Downsamples and expands grid cells for hippocampal input
    # Downsamples g_inf for memory indexing: g → g_downsampled
    # Expands g_downsampled for hippocampal conjunction: g_downsampled → g_repeated
    # Note: learn_W_down/learn_W_repeat in config are for demonstration; current implementation uses fixed matrices
    projection_model = Projection(W_down=W_down, W_repeat=W_repeat, config=config.projection)
    print(f"  ✓ Projection: {n_g} → {N_G_SUBSAMPLED} → {n_p}")

    # OVC ObjectInference (optional)
    ovc_model = None
    if config.ovc is not None and config.ovc.n_g_ovc:
        ovc_model = ObjectInference(n_g=n_g, config=config.ovc)

        # Check if OVC is in merged mode (won't produce separate output)
        if not config.ovc.frequencies:
            print(f"  ✓ ObjectInference: OVC module enabled in MERGED mode (integrated with grid cells)")
            print(f"    OVC dims: {config.ovc.n_g_ovc}, allocated backwards from grid modules")
        else:
            n_g_ovc_separate = config.ovc.n_g_ovc[-len(config.ovc.frequencies) :]
            print(f"  ✓ ObjectInference: OVC module enabled in SEPARATE mode ({len(config.ovc.frequencies)} frequencies, dims={n_g_ovc_separate})")
    else:
        print(f"  ✓ ObjectInference: OVC module disabled")

    print()

    # =========================================================================
    # PHASE 2: Generate Synthetic Data
    # =========================================================================
    print("Phase 2: Generating synthetic action sequence and landmarks...")

    # Random actions for demonstration (in real TEM: from policy/environment)
    actions = torch.randint(0, N_ACTIONS, (WALK_LENGTH, BATCH_SIZE))

    # Generate landmark locations (for OVC demonstration)
    # Place a "shiny" object every 10 timesteps for half the batch
    locations_sequence = []
    for t in range(WALK_LENGTH):
        batch_locations = []
        for b in range(BATCH_SIZE):
            has_shiny = (t % 10 == 0) and (b < BATCH_SIZE // 2)  # Shiny for first half of batch
            batch_locations.append({"shiny": True if has_shiny else None})
        locations_sequence.append(batch_locations)

    n_shiny_timesteps = sum(1 for locs in locations_sequence if any(loc["shiny"] for loc in locs))

    print(f"  ✓ Generated {WALK_LENGTH} random actions")
    print(f"  ✓ Generated landmarks: {n_shiny_timesteps} timesteps with 'shiny' objects")
    print()

    # =========================================================================
    # PHASE 3: Run MEC Pipeline
    # =========================================================================
    print("Phase 3: Running MEC pipeline...")
    print(f"  Pipeline: Transition → Abstract Fusion → Projection{' → OVC' if ovc_model else ''}")
    print()

    # Initialize history storage
    g_gen_transitions = []  # Transition objects (mean + uncertainty)
    g_inf_history = []  # Fused abstract locations
    g_downsampled_history = []  # Downsampled for memory
    g_grid_history = []  # Grid cells only (for visualization)
    g_ovc_history = []  # OVC cells only (for visualization)
    landmark_timesteps = []  # Track when landmarks were present

    # Initialize abstract location (first timestep)
    # Use actual dimensions from transition model (computed via resolve_dimensions)
    g_prev = [torch.randn(BATCH_SIZE, transition_model.n_g[f], device=DEVICE) * 0.1 for f in range(len(transition_model.n_g))]
    valid_mask = torch.ones(BATCH_SIZE, dtype=torch.bool, device=DEVICE)

    for t in range(WALK_LENGTH):
        # === Step 1: Transition (Path Integration) ===
        # Predict next location from current state + action
        action_t = actions[t]  # [B]
        g_gen_transition = transition_model(g_prev, action_t, valid_mask)
        g_gen = g_gen_transition.mean  # List[n_f] of [B, n_g[f]]

        # === Step 2: Abstract Location Fusion ===
        # Fuse path integration with optional memory correction
        # Note: p_x can be None (no memory path in this example)
        p_x = None  # No memory retrieval in this demo

        g_inf = abstract_model(g_gen_transition, p_x)  # List[n_f] of [B, n_g[f]]

        # === Step 3 (Optional): Object Vector Cells ===
        # Extract OVC activity (if configured)
        if config.ovc and config.ovc.frequencies:
            # Create dummy locations (placeholder logic: every 10 steps has landmark)
            is_landmark = t % 10 == 0
            locations = [{"shiny": torch.tensor([1.0], device=DEVICE)} if is_landmark else {} for _ in range(BATCH_SIZE)]
            g_ovc = ovc_model(g_gen_transition, locations)  # List[n_f_ovc] of [B, n_g_ovc[f]]
        else:
            g_ovc = []
            is_landmark = False

        # === Step 4: Projection ===
        # Downsample for memory, expand for hippocampus
        g_downsampled = projection_model.downsample(g_inf)  # List[n_f] of [B, n_g_sub[f]]

        # Update state for next timestep
        g_prev = g_inf

        # Store history
        g_gen_transitions.append(g_gen_transition)
        g_inf_history.append([g.clone().detach() for g in g_inf])
        g_downsampled_history.append([g.clone().detach() for g in g_downsampled])

        # Store grid and OVC separately (for batch 0 only)
        g_grid_history.append([g[0].clone().detach() for g in g_inf[:n_f_grid]])
        if g_ovc:
            g_ovc_history.append(torch.cat([g[0] for g in g_ovc]).clone().detach())
        else:
            g_ovc_history.append(torch.zeros(0))
        landmark_timesteps.append(is_landmark)

        # Progress reporting
        if (t + 1) % 20 == 0 or t == 0:
            avg_sigma = sum(s.mean().item() for s in g_gen_transition.uncertainty) / len(g_gen_transition.uncertainty)
            avg_g_grid = sum(g.abs().mean().item() for g in g_inf[:n_f_grid]) / n_f_grid if n_f_grid > 0 else 0
            avg_g_ovc = sum(g.abs().mean().item() for g in g_ovc) / len(g_ovc) if g_ovc else 0
            if g_ovc:
                print(f"  Step {t+1}/{WALK_LENGTH}: avg_g_grid={avg_g_grid:.4f}, avg_g_ovc={avg_g_ovc:.4f}, avg_sigma={avg_sigma:.4f}")
            else:
                print(f"  Step {t+1}/{WALK_LENGTH}: avg_g_grid={avg_g_grid:.4f}, avg_sigma={avg_sigma:.4f}")

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

    # Plot 3: g_inf evolution (grid modules only)
    fig3 = figures.plot_g_inf_evolution(g_inf_history=g_inf_history, n_frequencies=n_f_grid)
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

    # Plot 5: Grid temporal evolution (grid modules only)
    # Convert history to tensor format: List[T] of List[n_f_grid] of [B, n_g[f]]
    # to List[n_f_grid] of [T, B, n_g[f]]
    g_inf_sequences = []
    for f in range(n_f_grid):
        g_f_seq = torch.stack([g_inf_history[t][f] for t in range(WALK_LENGTH)])  # [T, B, n_g[f]]
        g_inf_sequences.append(g_f_seq)

    fig5 = figures.plot_grid_temporal_evolution(
        g_sequences=g_inf_sequences, frequencies=F_INITIAL[:n_f_grid], title="Multi-frequency Grid Cell Temporal Evolution (Grid Modules)", n_cells_per_freq=5
    )
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_grid_patterns.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_grid_patterns.png")

    # Plot 6 & 7: OVC visualizations (only if configured)
    if config.ovc and config.ovc.frequencies:
        # Extract separate OVC dimensions (last n_f_ovc_separate elements)
        n_g_ovc_separate = config.ovc.n_g_ovc[-len(config.ovc.frequencies) :]

        fig6 = figures.plot_ovc_activation_patterns(
            ovc_history=g_ovc_history,
            landmark_timesteps=landmark_timesteps,
            n_g_ovc=n_g_ovc_separate,
            frequencies=config.ovc.frequencies,
        )
        if config.save_plots:
            fig6.savefig(config.output_dir / "06_ovc_activation.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: 06_ovc_activation.png")

        fig7 = figures.plot_ovc_landmark_correlation(
            ovc_history=g_ovc_history,
            landmark_timesteps=landmark_timesteps,
            n_g_ovc=n_g_ovc_separate,
            frequencies=config.ovc.frequencies,
        )
        if config.save_plots:
            fig7.savefig(config.output_dir / "07_ovc_landmark_correlation.png", dpi=150, bbox_inches="tight")
            print(f"  Saved: 07_ovc_landmark_correlation.png")

    n_shiny_timesteps = sum(landmark_timesteps)
    print()
    print("=" * 80)
    print("MEC Pipeline Summary")
    print("=" * 80)
    print(f"Architecture:")
    print(f"  Total modules: {len(n_g)} ({n_f_grid} grid + {n_f_ovc_separate} separate OVC)")
    print(f"  Total dimensions: n_g={n_g}, n_p={n_p}")
    print(f"  Grid cells (grid-only portion): {dims.n_g_grid}")
    print(f"  OVC cells (per module): {dims.n_g_ovc}")
    print(f"  Downsampled: {N_G_SUBSAMPLED}")
    if config.ovc and config.ovc.n_g_ovc:
        print(f"  OVC config: {config.ovc.n_g_ovc}, frequencies={config.ovc.frequencies if config.ovc.frequencies else '[] (merged)'}")
    print()
    print(f"Pipeline Flow:")
    print(f"  1. TransitionModel: (g_t, a) → (g_gen, σ_gen)")
    print(f"     - Total dimensions: {n_g}")
    print(f"     - Hierarchical connections: {n_f_grid} grid + {n_f_ovc_separate} OVC modules")
    print(f"     - Action-conditioned dynamics via MLP (random weights for demo)")
    print()
    print(f"  2. AbstractLocModel: (g_gen, p_x?) → g_inf")
    print(f"     - Grid portion only: {dims.n_g_grid}")
    print(f"     - Precision-weighted fusion")
    print(f"     - Memory path: DISABLED (p_x=None in this example)")
    print()
    if config.ovc and config.ovc.frequencies:
        print(f"  3. ObjectInference: (g_gen, locations) → g_ovc")
        print(f"     - Separate OVC modules: {n_f_ovc_separate}")
        print(f"     - Landmark-based OVC activation")
        print(f"     - Landmarks detected: {n_shiny_timesteps} timesteps")
        print()
    print(f"  {4 if config.ovc and config.ovc.frequencies else 3}. Projection: g_inf → (g_down, g_repeat)")
    print(f"     - Downsample: {n_g} → {N_G_SUBSAMPLED} (memory indexing)")
    print(f"     - Expand: {N_G_SUBSAMPLED} → {n_p} (hippocampal conjunction)")
    print()
    print("=" * 80)
    print()
    print("TEM Theory:")
    print("  • MEC provides spatial 'where' information via grid cells")
    print("  • Path integration maintains spatial coherence during navigation")
    print("  • Action-conditioned transitions predict next grid cell state")
    print("  • Hierarchical frequencies capture multiple spatial scales")
    if ovc_model:
        print("  • OVCs provide 'what' information for landmark/object recognition")
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
