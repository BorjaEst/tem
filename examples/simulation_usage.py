#!/usr/bin/env python3
"""Simulation class example demonstrating TEM model iteration over walk trajectories.

This example shows how to use the Simulation iterator to process walk data through
the TEM model, automatically managing state updates and memory formation across
timesteps. The Simulation class provides a clean interface for running inference
or generative modes sequentially.

Key Concepts:
-------------
- Simulation iterator: Automatically processes walk trajectories timestep-by-timestep
- State management: TEMState tracks LEC, MEC, and HPC pathway activations
- Memory formation: Hebbian updates occur automatically during inference
- Walk generation: Synthetic trajectories with observations and actions

TEM Processing Pipeline:
------------------------
For each timestep, the Simulation class:
    1. Extracts observation x[t] and action a[t] from walk data
    2. Processes through LEC pathway (sensory encoding)
    3. Processes through MEC pathway (abstract location update)
    4. Performs memory retrieval from hippocampus
    5. Updates memory via Hebbian plasticity
    6. Returns updated TEMState with all pathway activations

Usage Examples:
---------------
    # Default: 4x4 grid, 50 timesteps
    python examples/simulation_usage.py

    # Larger environment with longer walk
    python examples/simulation_usage.py --grid_size 6 --walk_length 100

    # Different observation mode and architecture
    python examples/simulation_usage.py --observation_mode tiled --n_o_c 12

    # Custom frequency modules
    python examples/simulation_usage.py --f_initial "[0.95, 0.7, 0.4, 0.2]"

    # Show plots interactively
    python examples/simulation_usage.py --show_plots true --save_plots false

    # Full help
    python examples/simulation_usage.py --help

Outputs:
--------
When save_plots=true, generates 5 visualizations in outputs/simulation/:
    1. 01_environment.png - Grid layout and observation mapping
    2. 02_walk_trajectory.png - Agent path through environment
    3. 03_abstract_location.png - Grid cell activity heatmaps over time
    4. 04_grounded_location.png - Place cell activity with observations/locations
    5. 05_memory_formation.png - Hebbian memory matrix development timeline
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import figures, utils
from torch_tem.core.model import Simulation, TEMConfig, TEMModel, StandardTEMContext
from torch_tem.data.environment import Environment, EnvironmentConfig
from torch_tem.data.policies import PolicyConfig, PolicyGenerator, RandomPolicyConfig
from torch_tem.data.walks import WalkGenerator


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for Simulation usage example.

    Demonstrates configurable parameters for TEM model simulation including
    environment setup, architecture choices, and output options.
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="simulation_usage")

    # Environment configuration
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig, description="Environment configuration parameters")
    policy: PolicyConfig = Field(default_factory=RandomPolicyConfig, description="Policy configuration for walk generation")
    walk_length: int = Field(default=50, ge=1, description="Number of timesteps in the generated walk")

    # TEM model architecture
    tem: TEMConfig = Field(default_factory=TEMConfig, description="TEM model configuration parameters")

    # Output configuration
    output_dir: Path = Field(default=Path("outputs/simulation"), description="Directory for plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        """Create output directory if it doesn't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v


# Model architecture; not configurable via CLI
BATCH_SIZE = 1
DEVICE = torch.device("cpu")

N_G_SUBSAMPLED = [12, 10, 8]
N_F = len(N_G_SUBSAMPLED)
N_G = [3 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]  # [36, 30, 24]
N_P = [24, 40, 16]  # Divisible by both n_g_subsampled and n_o_c
N_O_C = 8  # C(8,2)=28 > 16 observations
F_INITIAL = [0.9, 0.6, 0.3]
I_ATTRACTOR = 3
MAX_FREQ_INF = [2, 3, 3]
MAX_FREQ_GEN = [3, 3, 3]


# ==============================================================================
# Main Example
# ==============================================================================
def main():
    """Run complete Simulation example with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("TEM Simulation Usage Example")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode})")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Architecture: {len(config.f_initial)} frequency modules")
    print(f"  Hebbian rate: η={config.eta}")
    print()

    # =========================================================================
    # PHASE 1: Setup Environment and Model
    # =========================================================================
    print("Phase 1: Initializing environment and model...")

    # Create environment configuration and context
    env_config = EnvironmentConfig(config.environment)
    model_config = TEMConfig(config.tem)

    # Initialize TEM model
    context = StandardTEMContext(
        environment=Environment(env_config),
        f_initial=F_INITIAL,
        W_tile=utils.create_tiling_matrices([N_O_C] * N_F, N_P),
        W_down=utils.create_downsample_matrix(N_G, N_G_SUBSAMPLED),
        W_repeat=utils.create_repeat_matrices(N_G_SUBSAMPLED, N_P),
        mask_inference=utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_INF),
        mask_generative=utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_GEN),
        update_mask=utils.create_p_update_mask(N_P, N_F, F_INITIAL),
    )   
    model = TEMModel(context, model_config).to(DEVICE)
    print(f"  ✓ Model initialized:")
    print(f"    - Frequencies: {model_config.f_initial}")
    print(f"    - Grid cells: {model_config.n_g} (before downsampling)")
    print(f"    - Place cells: {model_config.n_p} (outer product dimensions)")

    # =========================================================================
    # PHASE 2: Generate Walk Data
    # =========================================================================
    print("\nPhase 2: Generating walk trajectory...")

    # Create environment and walk generator
    env = Environment(env_config)
    policy ...
    walk_gen = WalkGenerator(env)
    walk = walk_gen.generate_walks(n_walks=1, walk_length=config.walk_length)[0]

    print(f"  ✓ Walk generated: {len(walk)} timesteps")
    print(f"    - Observations: {walk.observations[0].shape}")
    print(f"    - Actions: {walk.actions.shape}")
    print(f"    - Locations: {len(torch.unique(walk.locations))} unique locations visited")

    # =========================================================================
    # PHASE 3: Run Simulation
    # =========================================================================
    print("\nPhase 3: Running simulation...")

    # Storage for visualization
    abstract_locations = []  # g[t] at each timestep
    grounded_locations = []  # p[t] at each timestep
    predictions = []  # x_hat[t] at each timestep
    memory_snapshots = []  # M[t] at selected timesteps

    # Run simulation iterator
    sim = Simulation(model, walk)
    for t, state in enumerate(sim):
        # Store state components for analysis (extract batch dimension [0])
        abstract_locations.append([g[0].detach().cpu() for g in state.abstract_location])
        grounded_locations.append([p[0].detach().cpu() for p in state.grounded_location])

        # Capture memory snapshots at intervals
        if t % (config.walk_length // 4) == 0:
            M_gen = state.memory[0].detach().cpu()  # memory[0] is M_gen [B, N, N]
            M_inf = state.memory[1].detach().cpu() if state.memory[1] is not None else None
            memory_snapshots.append((t, M_gen, M_inf))

        # Progress indicator
        if (t + 1) % 10 == 0 or t == 0:
            has_grounded = state.grounded_location is not None
            print(f"  Step {t+1:3d}/{config.walk_length}: " f"abstract={len(state.abstract_location)} modules, " f"grounded={'✓' if has_grounded else '✗'}")

    print(f"  ✓ Simulation complete: {config.walk_length} timesteps processed")

    # =========================================================================
    # PHASE 4: Visualizations
    # =========================================================================
    if config.save_plots or config.show_plots:
        print("\nPhase 4: Generating visualizations...")

        # Plot 1: Environment layout
        fig = figures.plot_environment_layout(env, title="Environment Layout")
        if config.save_plots:
            fig.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        print("  ✓ Plot 1: Environment layout")

        # Plot 2: Walk trajectory
        fig = figures.plot_walks(env, [walk], title="Walk Trajectory")
        if config.save_plots:
            fig.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")
        print("  ✓ Plot 2: Walk trajectory")

        # Plot 3: Abstract location evolution (grid cells)
        fig = figures.plot_abstract_location_heatmap(abstract_locations, title="Abstract Location Evolution (Grid Cells)")
        if config.save_plots:
            fig.savefig(config.output_dir / "03_abstract_location.png", dpi=150, bbox_inches="tight")
        print("  ✓ Plot 3: Abstract location evolution")

        # Plot 4: Grounded location evolution (place cells)
        # Convert walk observations to list of tensors as expected by plot function
        fig = figures.plot_grounded_location_activity(
            p_history=grounded_locations,
            observations=[walk.observations[t] for t in range(len(walk))],
            locations=walk.locations,
            frequencies=model_config.f_extended,
            n_cells_per_freq=model_config.n_p,
            title="Grounded Location Evolution (Place Cells)",
        )
        if config.save_plots:
            fig.savefig(config.output_dir / "04_grounded_location.png", dpi=150, bbox_inches="tight")
        print("  ✓ Plot 4: Grounded location evolution")

        # Plot 5: Memory formation timeline
        fig = figures.plot_memory_formation_timeline(memory_snapshots, vmin=-0.1, vmax=0.1)
        if config.save_plots:
            fig.savefig(config.output_dir / "05_memory_formation.png", dpi=150, bbox_inches="tight")
        print("  ✓ Plot 5: Memory formation timeline")

        if config.save_plots:
            print(f"\n  All plots saved to: {config.output_dir.absolute()}")

        if config.show_plots:
            plt.show()

    print("\n" + "=" * 80)
    print("Example complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
