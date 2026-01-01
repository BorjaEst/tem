#!/usr/bin/env python3
"""Complete TEM inference pipeline example combining all inference components.

This example demonstrates the full torch_tem inference pipeline using subcomponents.

Pipeline Stages (TEM Manuscript - Inference):
----------------------------------------------
1. Compress sensory: o_c = encoder(o)
2. Temporal filter: x = processor(o_c, x_prev)
3. Project to HPC: x_ = projection(x)
4. Memory retrieval: p_x = attractor(x_, M)
5. Transition: g_gen = transition(g_prev, a)
6. Abstract inference: g = abstract(g_gen, p_x)
7. Project to HPC: g_ = projection(g)
8. Grounded inference: p = grounded(g_, x_)
9. Memory update: M = storage(p_inf, p_gen, M)

Usage Examples:
---------------
    python examples/tem_inference.py
    python examples/tem_inference.py --lec.encoder.two_hot true
    python examples/tem_inference.py --show_plots false

Outputs: 7 visualizations in outputs/inference/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import figures, utils
from torch_tem.core.hpc import HPCConfig, HPCModel, HPCState
from torch_tem.core.lec import LECConfig, LECModel, LECState
from torch_tem.core.mec import MECConfig, MECModel, MECState
from torch_tem.core.model import StandardTEMContext
from torch_tem.data.environment import Environment, EnvironmentConfig
from torch_tem.data.walks import WalkGenerator


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for TEM inference pipeline example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="tem_inference")

    # Component configurations
    lec: LECConfig = Field(default_factory=LECConfig, description="LEC pathway configuration")
    mec: MECConfig = Field(default_factory=MECConfig, description="MEC pathway configuration")
    hpc: HPCConfig = Field(default_factory=HPCConfig, description="HPC pathway configuration")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference"), description="Directory for saving plots")
    show_plots: bool = Field(default=True, description="Display plots interactively")
    save_plots: bool = Field(default=True, description="Save plots to output directory")

    @field_validator("output_dir")
    @classmethod
    def create_output_dir(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v


# Model architecture; not configurable via CLI
GRID_SIZE = 4  # 4x4 grid = 16 observations
OBSERVATION_MODE = "unique"
WALK_LENGTH = 100
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

env_config = EnvironmentConfig(width=GRID_SIZE, height=GRID_SIZE, observation_mode=OBSERVATION_MODE)
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


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    config = ExampleConfig()

    print("=" * 80)
    print("Complete TEM Inference Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {GRID_SIZE}×{GRID_SIZE} grid ({OBSERVATION_MODE})")
    print(f"  Walk length: {WALK_LENGTH} timesteps")
    print(f"  Architecture: n_g={N_G}, n_p={N_P}, n_o_c={N_O_C}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk
    # =========================================================================
    print("Phase 1: Generating environment and walk...")

    walk_gen = WalkGenerator(environment=context.environment, repeat_bias=2.0)
    walk_data = walk_gen.generate_walk(walk_length=WALK_LENGTH)

    observations = walk_data.observations.unsqueeze(1).to(DEVICE)  # [T, 1, n_o]
    actions = walk_data.actions.unsqueeze(1).to(DEVICE)  # [T, 1]
    locations = walk_data.locations.unsqueeze(1).to(DEVICE)  # [T, 1]

    print(f"  ✓ Environment: {context.environment.n_locations} locations, {context.environment.n_observations} observations")
    print(f"  ✓ Walk: {WALK_LENGTH} timesteps")
    print()

    # =========================================================================
    # PHASE 2: Initialize Components
    # =========================================================================
    print("Phase 2: Initializing components...")

    # LEC model
    lec_model = LECModel(context, config.lec)
    print(f"  ✓ LEC: Encoder, Processor, Projection, Decoder")

    # MEC model
    mec_model = MECModel(context, config.mec)
    print(f"  ✓ MEC: TransitionModel, Projection, AbstractLocModel")

    # HPC model
    hpc_model = HPCModel(context, config.hpc)
    print(f"  ✓ HPC: MemoryStorage, AttractorDynamics, GroundedLocInference")
    print()

    # =========================================================================
    # PHASE 3: Initialize States
    # =========================================================================
    print("Phase 3: Initializing states...")

    lec_state = lec_model.init_state(BATCH_SIZE, DEVICE)
    mec_state = mec_model.init_state(BATCH_SIZE, DEVICE)
    hpc_state = hpc_model.init_state(BATCH_SIZE, DEVICE)

    print(f"  ✓ States initialized")
    print()

    # =========================================================================
    # PHASE 4: Inference Loop
    # =========================================================================
    print("Phase 4: Running inference...")

    x_history = []
    p_history = []
    g_history = []

    for t in range(WALK_LENGTH):
        # LEC Pathway: Process sensory input to prepare for memory retrieval
        lec_state: LECState = lec_model(observations[t], lec_state)  # Process sensory input through LEC
        x_ = lec_state.projection  # Projected sensory code for HPC retrieval
        p_x = hpc_model.retrieve(x_, for_inference=True, state=hpc_state)

        # MEC Pathway: Infer abstract location from action and previous location
        a_t = actions[t] if t > 0 else None
        mec_state: MECState = mec_model(p_x, None, a_t, mec_state)  # Infer abstract location via MEC
        g_ = mec_state.projection

        # HPC Pathway: Infer grounded location from abstract location and sensory input
        p = hpc_model.grounded(g_, x_)

        # HPC: Update memory
        updated_memory = hpc_model.update(p, p, hpc_state)
        hpc_state = HPCState(grounded_location=p, memory=updated_memory)

        # Store (remove batch dimension for plotting convenience)
        x_history.append([x_f[0].detach().cpu() for x_f in lec_state.filtered_observation])
        p_history.append([p_f[0].detach().cpu() for p_f in p])
        g_history.append([g_f[0].detach().cpu() for g_f in mec_state.abstract_location])

        if (t + 1) % 20 == 0:
            print(f"  Processed {t + 1}/{WALK_LENGTH}")

    print(f"  ✓ Complete")
    print()

    # =========================================================================
    # PHASE 5: Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")

    # Environment layout
    fig1 = figures.data.plot_environment_layout(context.environment)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")

    # Walk trajectory
    fig2 = figures.data.plot_walks(context.environment, [walk_data])
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")

    # Memory matrices
    fig3 = figures.memory.plot_memory_matrices(M_gen=hpc_state.memory[0].squeeze(0), n_p_per_freq=N_P, title="Learned Memory Structure")
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_memory_structure.png", dpi=150, bbox_inches="tight")

    # Prepare data for place cell visualization (single trajectory, no batch dimension)
    obs_no_batch = [observations[t][0] for t in range(WALK_LENGTH)]
    fig4 = figures.grounded.plot_grounded_location_activity(
        p_history=p_history, observations=obs_no_batch, locations=locations.squeeze().cpu(), frequencies=F_INITIAL, n_cells_per_freq=N_P
    )
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_place_cell_activity.png", dpi=150, bbox_inches="tight")

    fig5 = figures.grounded.plot_place_cell_dynamics(p_history=p_history, observations=obs_no_batch, frequencies=F_INITIAL, n_cells_per_freq=N_P)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_place_cell_dynamics.png", dpi=150, bbox_inches="tight")

    # Prepare data for grid cell temporal evolution (add batch dimension back for plotting helper)
    g_sequences = []
    for f in range(N_F):
        g_f_seq = torch.stack([g_history[t][f] for t in range(WALK_LENGTH)])  # [T, n_g[f]]
        g_sequences.append(g_f_seq.unsqueeze(1))  # [T, 1, n_g[f]]

    fig6 = figures.patterns.plot_grid_temporal_evolution(g_sequences=g_sequences, frequencies=F_INITIAL)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_abstract_location.png", dpi=150, bbox_inches="tight")

    fig7 = figures.memory.plot_memory_matrices(M_gen=hpc_state.memory[0].squeeze(0), n_p_per_freq=N_P, title="Learned Memory Structure (Final)")
    if config.save_plots:
        fig7.savefig(config.output_dir / "07_memory.png", dpi=150, bbox_inches="tight")

    print(f"  ✓ Saved to: {config.output_dir}")

    if config.show_plots:
        plt.show()
    else:
        plt.close("all")

    print()
    print("=" * 80)
    print("Complete")
    print("=" * 80)
