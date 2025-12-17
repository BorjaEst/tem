#!/usr/bin/env python3
"""Complete TEM generative pipeline example.

This example demonstrates the generative pathway of TEM, showing how the model
generates sensory predictions from abstract location and memory.

Pipeline Stages (TEM Manuscript - Generative):
----------------------------------------------
Following the generative steps from the TEM manuscript:

1. Abstract location transition: g_t ~ transition(g_{t-1}, a_t)
2. Project to hippocampus: g_ = W_repeat·f_down(g_t)
3. Retrieve from memory: p_g = attractor(g_, M_gen)
4. Generate sensory prediction: x_hat = decoder(p_g)

Note: The manuscript uses a pre-learned memory M_gen without online updates
during generation. This example pre-generates the memory from walk data.

Data Flow (Manuscript Notation):
--------------------------------
    a_t (action)
    → g_t (abstract location from transition)
    → g_ (projected to hippocampus)
    → p_g (pattern completion via memory)
    → x_hat (sensory prediction via decoder)

Memory Initialization:
----------------------
    M_gen is pre-generated from a training walk to establish hippocampal
    associations before generation begins. No online updates during generation.

Usage Examples:
---------------
    # Default: 100 timesteps, save plots
    python examples/tem_generative.py

    # Longer walk with different architecture
    python examples/tem_generative.py --walk_length 200 --n_x_c 12

    # Different grid size and observation mode
    python examples/tem_generative.py --grid_size 7 --observation_mode tiled

    # Show plots interactively
    python examples/tem_generative.py --show_plots true --save_plots false

    # Full help
    python examples/tem_generative.py --help

Outputs:
--------
When save_plots=true, generates 5 visualizations in outputs/generation_location/:
    1. 01_environment.png - Grid layout
    2. 02_walk_trajectory.png - Agent trajectory
    3. 03_grid_evolution.png - Abstract location over time
    4. 04_sensory_predictions.png - True vs predicted observations
    5. 05_memory_structure.png - Pre-learned memory matrix
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from torch_tem import core, data, figures, hpc, lec, mec, utils
from torch_tem.config import EnvironmentConfig, ModelConfig


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for complete TEM generative pipeline example.

    This config implements all generative-related protocols:
    - TransitionParams
    - ProjectionParams
    - DecoderParams
    - MemoryParams (generative pathway)
    """

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
    output_dir: Path = Field(default=Path("outputs/generation_location"), description="Directory for saving plots")
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
    # Calculate total actions (directional + static if enabled)
    total_actions = environment_config.n_actions + (1 if environment_config.has_static_action else 0)
    model_config = ModelConfig(
        n_x=environment_config.n_locations,
        n_x_c=config.n_x_c,
        n_g_subsampled=config.n_g_subsampled,
        f_initial=config.f_initial,
        n_actions=total_actions,
        eta=config.eta,
        kappa=config.kappa,
        batch_size=1,  # Single walk trajectory
    )

    print("=" * 80)
    print("Complete TEM Generative Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Actions: {total_actions} (4 directional + {1 if environment_config.has_static_action else 0} static)")
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

    # MEC: Abstract location processing
    transition = mec.transition.TransitionModel(model_config)
    mec_projection = mec.projection.Projection(model_config)
    print(f"  ✓ TransitionModel: g_{{t-1}}, a_t → g_t")
    print(f"  ✓ Projection: g → g_ (downsample + repeat)")
    print()

    # Decoder: Hippocampus to sensory space
    decoder = lec.decoder.Decoder(model_config)
    # Create W_tile for decoder (only needs first frequency matrix)
    W_tile_0 = torch.randn(model_config.n_x_c, model_config.n_p[0]) / np.sqrt(model_config.n_p[0])
    print(f"  ✓ Decoder: p → x̂ (place cells to sensory prediction)")
    print()

    # =========================================================================
    # PHASE 3: Pre-generate Memory from Training Walk
    # =========================================================================
    print("Phase 3: Pre-generating memory from training walk...")

    # Generate training walk for memory learning
    training_walk_length = config.walk_length  # Use same length for simplicity
    training_walks = walk_gen.generate_walks(n_walks=1, walk_length=training_walk_length, policy=policy)
    training_walk = training_walks[0]

    # Simulate training: collect grid cell patterns
    g_training = []
    g_state = [torch.randn(1, model_config.n_g[f]) * 0.1 for f in range(model_config.n_f)]

    for t in range(training_walk_length):
        action = training_walk.actions[t].unsqueeze(0)
        g_trans = transition(g_state, action)
        g_state = g_trans.mean
        g_training.append([g_f.detach() for g_f in g_state])

    # Generate memory from training patterns
    memory_gen = data.MemoryMatrixGenerator(model_config)
    M_gen_init, M_inf_init = memory_gen.from_walk(g_training, lambda g: mec_projection.repeat(mec_projection.downsample(g)))

    # Initialize memory matrices externally (functional interface)
    n_p_total = sum(model_config.n_p)
    M_gen = M_gen_init.unsqueeze(0)  # Add batch dimension: [B, N, N]
    M_inf = M_inf_init.unsqueeze(0) if not model_config.common_memory else None

    # Initialize attractor for retrieval
    mask_inf = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_inf)
    mask_gen = utils.create_p_retrieve_mask(model_config.n_p, model_config.i_attractor, model_config.max_freq_gen)
    mem_attractor = hpc.attractor.AttractorDynamics(model_config, mask_inf, mask_gen)

    print(f"  ✓ Generated memory from {training_walk_length} training steps")
    print(f"  ✓ Memory: {sum(model_config.n_p)}×{sum(model_config.n_p)} Hebbian matrix")
    print(f"  ✓   - {model_config.i_attractor} attractor iterations")
    print()

    # =========================================================================
    # PHASE 4: Initialize Component States
    # =========================================================================
    print("Phase 4: Initializing component states...")
    # Initialize grid cell state (random start)
    g_prev = [torch.randn(1, model_config.n_g[f]) * 0.1 for f in range(model_config.n_f)]
    # Initialize sensory processor state (zeros for first timestep)
    x_prev = [torch.zeros(1, model_config.n_x_f[f]) for f in range(model_config.n_f)]
    print(f"  ✓ Grid cell state initialized: {model_config.n_g}")
    print(f"  ✓ Sensory processor state initialized: {model_config.n_x_f}")
    print()

    # =========================================================================
    # PHASE 5: Run Complete Generative Pipeline
    # =========================================================================
    print("Phase 5: Running complete generative pipeline...")

    # History tracking
    g_history = []  # g: abstract location evolution
    g__history = []  # g_: projected abstract location
    p_g_history = []  # p_g: retrieved place cells
    x_pred_history = []  # x̂: sensory predictions

    for t in range(config.walk_length):
        action_t = walk.actions[t].unsqueeze(0)  # [B] = [1]

        # ============================================================
        # Step 1 (Manuscript): Abstract location transition
        # g_t ~ N(μ_g(g_{t-1}, a_t), Σ_g)
        # ============================================================
        g_gen = transition(g_prev, action_t)  # Returns Transition(mean, uncertainty)
        g = g_gen.mean  # List[n_f] of [B, n_g[f]]
        g_history.append([g_f[0].detach() for g_f in g])

        # ============================================================
        # Step 2 (Manuscript): Project to hippocampus
        # g_ = W_repeat·f_down(g)
        # ============================================================
        g_downsampled = mec_projection.downsample(g)  # List[n_f] of [B, n_g_sub[f]]
        g_ = mec_projection.repeat(g_downsampled)  # List[n_f] of [B, n_p[f]]
        g__history.append([g_f[0].detach() for g_f in g_])

        # ============================================================
        # Step 3 (Manuscript): Retrieve from memory
        # p_g = attractor(g_, M_gen)
        # ============================================================
        p_g = mem_attractor(g_, M_gen, for_inference=False)  # List[n_f] of [B, n_p[f]]
        p_g_flat = torch.cat(p_g, dim=1)  # [B, sum(n_p)]
        p_g_history.append(p_g_flat[0].detach())

        # ============================================================
        # Step 4 (Manuscript): Generate sensory prediction
        # x̂ = decoder(p_g, W_tile_0)
        # ============================================================
        x_pred_result = decoder(p_g, W_tile_0)  # Returns SensoryPrediction
        x_pred = x_pred_result.values[0]  # First frequency: [B, n_x]
        x_pred_history.append(x_pred[0].detach())

        # Update state for next iteration
        g_prev = g

    print(f"  ✓ Processed {config.walk_length} timesteps")
    print(f"  ✓ Followed 4-step generative process:")
    print(f"      1. Abstract location transition (g)")
    print(f"      2. Project to hippocampus (g_)")
    print(f"      3. Retrieve from memory (p_g)")
    print(f"      4. Generate sensory prediction (x̂)")
    print(f"  ✓ Used pre-learned memory (no online updates)")
    print()

    # =========================================================================
    # PHASE 6: Generate Visualizations
    # =========================================================================
    print("Phase 6: Generating visualizations...")

    # Compute prediction accuracy
    x_pred_tensor = torch.stack(x_pred_history)  # [T, n_x]
    x_true_tensor = torch.stack(observations)  # [T, n_x]
    predictions = x_pred_tensor.argmax(dim=-1)
    truth = x_true_tensor.argmax(dim=-1)
    prediction_acc = (predictions == truth).float().mean().item()

    # Plot 1: Environment layout
    fig1 = figures.plot_environment_layout(env, title=f"Environment: {config.grid_size}×{config.grid_size} Grid")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: 01_environment.png")

    # Plot 2: Walk trajectory
    fig2 = figures.plot_walks(env, [walk], title=f"Walk Trajectory ({config.walk_length} steps)")
    if config.save_plots:
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: 02_walk_trajectory.png")

    # Plot 3: Abstract location evolution (Step 1)
    # Show evolution of abstract location g over time
    g_array = torch.stack([torch.cat(g_t) for g_t in g_history]).cpu().numpy()  # [T, sum(n_g)]
    fig3, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(g_array.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Abstract Location Dimension")
    ax.set_title(f"Step 1: Abstract Location Evolution (g) - {sum(model_config.n_g)} dims")
    plt.colorbar(im, ax=ax, label="Activation")
    plt.tight_layout()
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_grid_evolution.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: 03_grid_evolution.png")

    # Plot 4: Sensory predictions (Step 4)
    # Compare true observations vs generated predictions
    fig4, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # True observations
    x_true_np = x_true_tensor.cpu().numpy()  # [T, n_x]
    im1 = axes[0].imshow(x_true_np.T, aspect="auto", cmap="Blues", interpolation="nearest")
    axes[0].set_ylabel("Location ID")
    axes[0].set_title("True Observations (Ground Truth)")
    plt.colorbar(im1, ax=axes[0])

    # Predicted observations
    x_pred_np = x_pred_tensor.cpu().numpy()  # [T, n_x]
    im2 = axes[1].imshow(x_pred_np.T, aspect="auto", cmap="Oranges", interpolation="nearest")
    axes[1].set_ylabel("Location ID")
    axes[1].set_xlabel("Timestep")
    axes[1].set_title(f"Step 4: Generated Predictions (Accuracy: {prediction_acc:.1%})")
    plt.colorbar(im2, ax=axes[1])

    fig4.suptitle("Generative Model: x̂ = decoder(p_g)", fontsize=14, y=0.995)
    plt.tight_layout()
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_sensory_predictions.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: 04_sensory_predictions.png")

    # Plot 5: Memory structure (pre-learned)
    # Show structure of pre-learned memory matrix
    M_gen_final = M_gen[0].detach().cpu().numpy()  # [sum(n_p), sum(n_p)]
    fig5, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Memory matrix structure
    im1 = axes[0].imshow(M_gen_final, aspect="auto", cmap="RdBu_r", interpolation="nearest", vmin=-np.abs(M_gen_final).max(), vmax=np.abs(M_gen_final).max())
    axes[0].set_xlabel("Post-synaptic (j)")
    axes[0].set_ylabel("Pre-synaptic (i)")
    axes[0].set_title(f"Pre-learned Memory Matrix M_gen [{sum(model_config.n_p)}×{sum(model_config.n_p)}]")
    plt.colorbar(im1, ax=axes[0], label="Weight")

    # Memory matrix histogram
    axes[1].hist(M_gen_final.flatten(), bins=50, alpha=0.7, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Weight Value")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Memory Weight Distribution")
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=1, label="Zero")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig5.suptitle(f"Hebbian Memory (trained on {training_walk_length} steps)", fontsize=14, y=0.995)
    plt.tight_layout()
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_memory_structure.png", dpi=150, bbox_inches="tight")
        print(f"  ✓ Saved: 05_memory_structure.png")

    print()
    print("=" * 80)
    print("GENERATIVE PIPELINE SUMMARY")
    print("=" * 80)
    print(f"Environment:        {config.grid_size}×{config.grid_size} grid")
    print(f"Training Steps:     {training_walk_length} (for memory pre-learning)")
    print(f"Generation Steps:   {config.walk_length}")
    print(f"Frequencies:        {model_config.n_f} scales")
    print(f"Grid Cells:         {sum(model_config.n_g)} total")
    print(f"Conjunctive Codes:  {sum(model_config.n_p)} total")
    print(f"Prediction Acc:     {prediction_acc:.1%}")
    print(f"Memory Size:        {sum(model_config.n_p)}×{sum(model_config.n_p)}")
    print("=" * 80)
    print()
    print("Pipeline Flow (Manuscript Steps):")
    print("=" * 80)
    print(f"Pre-training: M_gen learned from {training_walk_length} training steps")
    print(f"Input:  a_t - Actions from policy")
    print(f"  ↓ Step 1: transition(g_{{t-1}}, a_t) - Evolve abstract location")
    print(f"Stage 1: g - Abstract location - {model_config.n_g}")
    print(f"  ↓ Step 2: W_repeat·f_down(g) - Project to hippocampus")
    print(f"Stage 2: g_ - Projected grid representation - {model_config.n_p}")
    print(f"  ↓ Step 3: attractor(g_, M_gen) - Retrieve from memory")
    print(f"Stage 3: p_g - Retrieved hippocampal patterns - {model_config.n_p}")
    print(f"  ↓ Step 4: decoder(p_g) - Generate sensory prediction")
    print(f"Output: x̂ - Predicted observations - {model_config.n_x}")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")

    print("\n✓ Generative pipeline demonstration complete!")
