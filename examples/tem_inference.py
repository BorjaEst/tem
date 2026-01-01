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
from torch_tem.core.hpc import HPCConfig, HPCContext, HPCModel, HPCState
from torch_tem.core.lec import LECConfig, LECContext, LECModel, LECState
from torch_tem.core.mec import MECConfig, MECContext, MECModel, MECState
from torch_tem.data.environment import Environment, EnvironmentConfig
from torch_tem.data.policies import RandomPolicyConfig
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
GRID_SIZE = 5
OBSERVATION_MODE = "unique"
WALK_LENGTH = 100
BATCH_SIZE = 1
DEVICE = torch.device("cpu")

N_G_SUBSAMPLED = [12, 10, 8]
N_F = len(N_G_SUBSAMPLED)
N_G = [3 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]
N_P = [2 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]
N_O_C = 8
F_INITIAL = [0.9, 0.6, 0.3]
I_ATTRACTOR = 3
MAX_FREQ_INF = [2, 3, 3]

W_down = utils.create_downsample_matrix(N_G, N_G_SUBSAMPLED)
W_repeat = utils.create_repeat_matrices(N_G_SUBSAMPLED, N_P)
W_tile = utils.create_tiling_matrices([N_O_C] * N_F, N_P)
p_update_mask = utils.create_p_update_mask(N_P, N_F, N_F, 0, F_INITIAL)
mask_inf = utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_INF)


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

    env_config = EnvironmentConfig(width=GRID_SIZE, height=GRID_SIZE, observation_mode=OBSERVATION_MODE)
    env = Environment(env_config)
    walk_gen = WalkGenerator(env=env, policy_config=RandomPolicyConfig())
    walk_data = walk_gen.generate_walk(length=WALK_LENGTH)

    observations = walk_data.observations.unsqueeze(1).to(DEVICE)  # [T, 1, n_o]
    actions = walk_data.actions.unsqueeze(1).to(DEVICE)  # [T, 1]
    locations = walk_data.locations.unsqueeze(1).to(DEVICE)  # [T, 1]

    print(f"  ✓ Environment: {env.n_locations} locations, {env.n_observations} observations")
    print(f"  ✓ Walk: {WALK_LENGTH} timesteps")
    print()

    # =========================================================================
    # PHASE 2: Initialize Components
    # =========================================================================
    print("Phase 2: Initializing components...")

    # LEC model
    lec_context = LECContext(n_o=env.n_observations, f_initial=F_INITIAL, W_tile=W_tile)
    lec_model = LECModel(lec_context, config.lec)
    print(f"  ✓ LEC: Encoder, Processor, Projection, Decoder")

    # MEC model
    mec_context = MECContext(W_down=W_down, W_repeat=W_repeat, n_f_grid=N_F, f_initial=F_INITIAL, n_actions=env.n_actions)
    mec_model = MECModel(mec_context, config.mec)
    print(f"  ✓ MEC: TransitionModel, Projection, AbstractLocModel")

    # HPC model
    hpc_context = HPCContext(mask_inference=mask_inf, mask_generative=mask_inf, update_mask=p_update_mask)
    hpc_model = HPCModel(hpc_context, config.hpc)
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
        state_lec: LECState = lec_model(o, lec_state)  # Process sensory input through LEC
        x_ = state_lec.projection  # Projected sensory code for HPC retrieval
        p_x = hpc_model.retrieve(x_, for_inference=True, state=hpc_state)

        # MEC Pathway: Infer abstract location from action and previous location
        state_mec: MECState = mec_model(p_x, locations, a[t], mec_state)  # Infer abstract location via MEC
        g = state_mec.abstract_location
        g_ = state_mec.projection

        # HPC Pathway: Infer grounded location from abstract location and sensory input
        p = hpc_model.grounded(g_, x_)

    print(f"  ✓ Complete")
    print()

    # =========================================================================
    # PHASE 5: Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")

    fig1 = figures.data.plot_environment(env)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")

    fig2 = figures.data.plot_walk(env, locations.squeeze().cpu().numpy())
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")

    fig3 = figures.sensory.plot_temporal_filtering(x_history, F_INITIAL)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_sensory_processing.png", dpi=150, bbox_inches="tight")

    fig4 = figures.grounded.plot_place_cells(p_history, locations.squeeze().cpu().numpy())
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_place_cell_activity.png", dpi=150, bbox_inches="tight")

    mid_t = WALK_LENGTH // 2
    fig5 = figures.grounded.plot_outer_product(g_history[mid_t], x_history[mid_t], p_history[mid_t])
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_outer_product.png", dpi=150, bbox_inches="tight")

    fig6 = figures.patterns.plot_abstract_location(g_history, F_INITIAL)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_abstract_location.png", dpi=150, bbox_inches="tight")

    fig7 = figures.memory.plot_memory_matrix(hpc_state.memory[0].squeeze().cpu().numpy())
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
