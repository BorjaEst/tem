#!/usr/bin/env python3
"""Complete TEM generative pipeline example.

This example demonstrates the full torch_tem generative pipeline using subcomponents.

Pipeline Stages (TEM Manuscript - Generative):
----------------------------------------------
1. Transition: g_t = transition(g_{t-1}, a_t)
2. Project to HPC: g_ = projection(g)
3. Memory retrieval: p_g = attractor(g_, M_gen)
4. Decode sensory: x_hat = decoder(p_g)

Usage Examples:
---------------
    python examples/tem_generative.py
    python examples/tem_generative.py --mec.transition.d_hidden_dim 30
    python examples/tem_generative.py --show_plots false

Outputs: 5 visualizations in outputs/generation_location/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import figures, utils
from torch_tem.core.hpc import HPCConfig, HPCContext, HPCModel
from torch_tem.core.lec import LECConfig, LECContext, LECModel
from torch_tem.core.mec import MECConfig, MECContext, MECModel, MECState
from torch_tem.data.environment import Environment, EnvironmentConfig
from torch_tem.data.policies import RandomPolicyConfig
from torch_tem.data.walks import WalkGenerator


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for TEM generative pipeline example."""

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="tem_generative")

    # Component configurations
    lec: LECConfig = Field(default_factory=LECConfig, description="LEC pathway configuration")
    mec: MECConfig = Field(default_factory=MECConfig, description="MEC pathway configuration")
    hpc: HPCConfig = Field(default_factory=HPCConfig, description="HPC pathway configuration")

    # Output
    output_dir: Path = Field(default=Path("outputs/generation_location"), description="Directory for saving plots")
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
WALK_LENGTH_TRAIN = 50  # Training walk for memory
WALK_LENGTH_GEN = 100  # Generation walk
BATCH_SIZE = 1
DEVICE = torch.device("cpu")

N_G_SUBSAMPLED = [12, 10, 8]
N_F = len(N_G_SUBSAMPLED)
N_G = [3 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]
N_P = [2 * n_g_sub for n_g_sub in N_G_SUBSAMPLED]
N_O_C = 8
F_INITIAL = [0.9, 0.6, 0.3]
I_ATTRACTOR = 3
MAX_FREQ_GEN = [3, 3, 3]

W_down = utils.create_downsample_matrix(N_G, N_G_SUBSAMPLED)
W_repeat = utils.create_repeat_matrices(N_G_SUBSAMPLED, N_P)
W_tile = utils.create_tiling_matrices([N_O_C] * N_F, N_P)
p_update_mask = utils.create_p_update_mask(N_P, N_F, N_F, 0, F_INITIAL)
mask_gen = utils.create_p_retrieve_mask(N_P, I_ATTRACTOR, MAX_FREQ_GEN)


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    config = ExampleConfig()

    print("=" * 80)
    print("Complete TEM Generative Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {GRID_SIZE}×{GRID_SIZE} grid ({OBSERVATION_MODE})")
    print(f"  Training walk: {WALK_LENGTH_TRAIN} timesteps")
    print(f"  Generation walk: {WALK_LENGTH_GEN} timesteps")
    print(f"  Architecture: n_g={N_G}, n_p={N_P}, n_o_c={N_O_C}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Training Walk
    # =========================================================================
    print("Phase 1: Generating training walk for memory...")

    env_config = EnvironmentConfig(width=GRID_SIZE, height=GRID_SIZE, observation_mode=OBSERVATION_MODE)
    env = Environment(env_config)
    walk_gen = WalkGenerator(env=env, policy_config=RandomPolicyConfig())

    # Training walk
    train_data = walk_gen.generate_walk(length=WALK_LENGTH_TRAIN)
    train_obs = train_data.observations.unsqueeze(1).to(DEVICE)
    train_actions = train_data.actions.unsqueeze(1).to(DEVICE)
    train_locs = train_data.locations.unsqueeze(1).to(DEVICE)

    # Generation walk (actions only)
    gen_data = walk_gen.generate_walk(length=WALK_LENGTH_GEN)
    gen_actions = gen_data.actions.unsqueeze(1).to(DEVICE)
    gen_locs = gen_data.locations.unsqueeze(1).to(DEVICE)

    print(f"  ✓ Environment: {env.n_locations} locations, {env.n_observations} observations")
    print(f"  ✓ Training walk: {WALK_LENGTH_TRAIN} timesteps")
    print(f"  ✓ Generation walk: {WALK_LENGTH_GEN} actions")
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
    hpc_context = HPCContext(mask_inference=mask_gen, mask_generative=mask_gen, update_mask=p_update_mask)
    hpc_model = HPCModel(hpc_context, config.hpc)
    print(f"  ✓ HPC: MemoryStorage, AttractorDynamics, GroundedLocInference")
    print()

    # =========================================================================
    # PHASE 3: Pre-learn Memory from Training Walk
    # =========================================================================
    print("Phase 3: Pre-learning memory from training walk...")

    lec_state = lec_model.init_state(BATCH_SIZE, DEVICE)
    mec_state = mec_model.init_state(BATCH_SIZE, DEVICE)
    hpc_state = hpc_model.init_state(BATCH_SIZE, DEVICE)

    # Run inference on training walk to build memory
    for t in range(WALK_LENGTH_TRAIN):
        o_c = lec_model.encoder(train_obs[t])
        lec_state = lec_model.processor(o_c, lec_state)
        x = lec_state.x_prev
        x_proj = lec_model.projection(x)

        p_x = hpc_model.attractor(x_proj, hpc_state.memory[0])

        a_t = train_actions[t] if t > 0 else None
        transition_out = mec_model.transition(mec_state.g, a_t, mec_state.sigma_g)
        g = mec_model.abstract(transition_out, p_x)
        g_proj = mec_model.projection(g)

        p_out = hpc_model.grounded(g_proj, x_proj)
        p = p_out.location

        hpc_state = hpc_model.storage(p, p, hpc_state)
        mec_state = MECState(g=g, sigma_g=transition_out.sigma_g)

    M_gen = hpc_state.memory[0].clone()  # Save learned memory
    print(f"  ✓ Memory learned from {WALK_LENGTH_TRAIN} timesteps")
    print()

    # =========================================================================
    # PHASE 4: Generative Loop
    # =========================================================================
    print("Phase 4: Running generative pipeline...")

    # Reset state for generation
    mec_state = mec_model.init_state(BATCH_SIZE, DEVICE)

    g_history = []
    p_history = []
    x_pred_history = []

    for t in range(WALK_LENGTH_GEN):
        # MEC: g_{t-1}, a_t → g_t (path integration)
        a_t = gen_actions[t] if t > 0 else None
        state_mec: MECState = mec_model(None, locations, a_t, mec_state)

        # MEC: g → g_ (projection)
        g_ = mec_model.projection

        # HPC: g_ → p_g (memory retrieval)
        p_g = hpc_model.retrieve(g_, for_inference=False, state=hpc_state)

        # LEC: p_g → x_hat (decode sensory prediction)
        x_pred = lec_model.decoder(p_g)

        # Update state
        mec_state = MECState(g=

        # Store
        g_history.append([g_f.detach().cpu() for g_f in g])
        p_history.append([p_f.detach().cpu() for p_f in p_g])
        x_pred_history.append([x_f.detach().cpu() for x_f in x_pred])

        if (t + 1) % 20 == 0:
            print(f"  Processed {t + 1}/{WALK_LENGTH_GEN}")

    print(f"  ✓ Complete")
    print()

    # =========================================================================
    # PHASE 5: Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")

    fig1 = figures.data.plot_environment(env)
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")

    fig2 = figures.data.plot_walk(env, gen_locs.squeeze().cpu().numpy())
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")

    fig3 = figures.patterns.plot_abstract_location(g_history, F_INITIAL)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_grid_evolution.png", dpi=150, bbox_inches="tight")

    fig4 = figures.sensory.plot_sensory_predictions(x_pred_history, F_INITIAL)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_sensory_predictions.png", dpi=150, bbox_inches="tight")

    fig5 = figures.memory.plot_memory_matrix(M_gen.squeeze().cpu().numpy())
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_memory_structure.png", dpi=150, bbox_inches="tight")

    print(f"  ✓ Saved to: {config.output_dir}")

    if config.show_plots:
        plt.show()
    else:
        plt.close("all")

    print()
    print("=" * 80)
    print("Complete")
    print("=" * 80)
