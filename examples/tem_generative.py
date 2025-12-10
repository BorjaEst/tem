#!/usr/bin/env python3
""" """

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import core, data, figures, lec, mec, memory, utils
from torch_tem.config import EnvironmentConfig, ModelConfig
from torch_tem.mec.projection import Projection
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="tem_generative")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    f_initial: List[float] = Field(default_factory=lambda: [0.9, 0.5, 0.2], description="Initial frequencies for each module")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Output
    output_dir: Path = Field(default=Path("outputs/inference"), description="Directory for saving plots")
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
    """Run the complete TEM generative pipeline with visualizations."""
    config = ExampleConfig()

    # Create config objects with proper field mapping
    environment_config = EnvironmentConfig(width=config.grid_size, height=config.grid_size, observation_mode=config.observation_mode)
    model_config = ModelConfig(
        n_x=environment_config.n_locations, n_x_c=config.n_x_c, n_g_subsampled=config.n_g_subsampled, f_initial=config.f_initial, eta=config.eta, kappa=config.kappa
    )

    # Compute connectivity matrices from model config
    two_hot_table = utils.create_two_hot_table(model_config.n_x, model_config.n_x_c)
    g_downsample = utils.create_g_downsample(model_config.n_g, model_config.n_g_subsampled_combined)
    p_update_mask = utils.create_p_update_mask(model_config.n_p, model_config.n_f, model_config.n_f, 0, model_config.f_extended)
    mask_inf = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf)
    mask_gen = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_gen)
    W_repeat = utils.create_W_repeat(model_config.n_g_subsampled_combined, model_config.n_x_f)
    W_tile = utils.create_W_tile(model_config.n_g_subsampled_combined, model_config.n_x_f)

    print("=" * 80)
    print("Complete TEM Generative Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Frequencies: {model_config.n_f} ({model_config.f_initial[0]:.2f} to {model_config.f_initial[-1]:.2f})")
    print(f"  Architecture: n_g={model_config.n_g}, n_p={model_config.n_p}, n_x_c={model_config.n_x_c}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk Generation
    # =========================================================================
    print("Phase 1: Generating walk trajectory...")
    env = data.Environment(environment_config)
    env.validate()

    policy_gen = data.PolicyGenerator(env)
    policy = policy_gen.random_policy()

    walk_gen = data.WalkGenerator(env)
    walks = walk_gen.generate_walks(n_walks=1, walk_length=config.walk_length, policy=policy)
    walk = walks[0]

    observations = [obs.clone().detach() for obs in walk.observations]  # List[T] of [n_x]
    locations = torch.as_tensor(walk.locations, dtype=torch.long)  # [T]
    print(f"  ✓ Generated walk: {len(walk)} timesteps")
    print()

    # =========================================================================
    # PHASE 2: Initialize All Components
    # =========================================================================
    print("Phase 2: Initializing generative components...")

    # Abstract location transition model
    transition = mec.transition.TransitionModel(model_config)
    abstract = mec.abstract.AbstractLocInference(model_config)
    print(f"  ✓ TransitionModel: Action-based dynamics with hierarchical g_connections")
    print(f"  ✓ AbstractLocGenerative: precision-weighted fusion")
    print()

    # Projection to p-space
    mec_projection = core.Projection(model_config)
    print(f"  ✓ Projection: x_f → x_ (W_tile expansion + w_p gating)")

    # Memory system
    storage = MemoryStorage(model_config, p_update_mask)
    attractor = AttractorDynamics(model_config)
    print(f"  ✓ MemoryStorage: {sum(model_config.n_p)}×{sum(model_config.n_p)} Hebbian matrix")
    print(f"  ✓ AttractorDynamics: {model_config.i_attractor} iterations with hierarchical masking")

    # =========================================================================
    # PHASE 3: Initialize Grid Cell State
    # =========================================================================
    print("Phase 3: Initializing abstract location state...")
    # Generate a single synthetic transition for initial uncertainty estimate
    grid_generator = data.OscillatoryGridGenerator(model_config, 1, batch_size=1)
    initial_transition = grid_generator.generate()[0]  # Single Transition (g_init, sigma_init)
    print(f"  ✓ Initial abstract location and uncertainty sampled")
    print()

    # =========================================================================
    # PHASE 4: Run Complete Generative Pipeline
    # =========================================================================
    print("Phase 4: Running complete generative pipeline...")

    g_history = []  # g: inferred entorhinal (abstract location)
    x_history = []  # ~x: sensory input to hippocampus

    x_prev = [torch.zeros(1, model_config.n_x_c) for _ in range(model_config.n_f)]
    g_prev, sigma_prev = initial_transition  # Unpack initial Transition (g, sigma)
    for t in range(config.walk_length):
        # Step 1 (Manuscript): State transition g_t = N(·| g_{t-1}, a_t), σ_{t-1})
        g_gen = transition(a, g_prev)  # [B, n_g]
        g = abstract(g_gen, None, locations)  # List[n_f] of [B, n_g_sub[f]]
        g_history.append([g[0] for g in g])

        # Step 2 (Manuscript): Entorhinal input to hippocampus ~g = W_repeat·f_down(g_t)
        g_ = mec_projection(g_gen.mean)  # [B, sum(n_p)]

        # Step 3 (Manuscript): Retrieve memory p_x = attractor(~g ⊙ ~x, M_{t-1})
        M_gen = storage.get_memory(for_inference=False)
        p_g = attractor(g_, M_gen, for_inference=False)  # [B, sum(n_p)]

        # Step 4 (Manuscript): Sensory prediction x_hat = decoder(p_g)
        x_hat = core.Decoder(model_config)(p_g)  # [B, n_x]
        x_history.append(x_hat[0])

        # Step 5 (Manuscript): Repeat process for next timestep
        g_prev = g  # Update previous abstract location

    print(f"  ✓ Processed {config.walk_length} timesteps through complete pipeline")
    print()

    # =========================================================================
    # PHASE 5: Generate Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")
