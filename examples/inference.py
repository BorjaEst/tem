#!/usr/bin/env python3
"""Complete TEM inference pipeline example combining all inference components.

This example demonstrates the full torch_tem inference pipeline, integrating components
from inference_sensory.py, inference_grounded.py, inference_abstract.py, memory_storage.py,
and memory_attractor.py into a single comprehensive demonstration.

Pipeline Stages (Theory-Faithful):
-----------------------------------
1. Sensory Processing: Observation encoding and temporal filtering
   - SensoryEncoder: Two-hot encoding (n_x → n_x_c)
   - SensoryProcessor: Multi-frequency temporal filtering

2. Sensory to Hippocampus: Project sensory input to place cell space
   - SensoryProjection: ~x_t = W_tile @ w_p @ f_n(x_f)
   - AttractorDynamics: p_x = M^T @ ~x_t (retrieve from sensory input)

3. Abstract Location Inference: Combine memory and generative paths
   - Memory path: p_x → g_mem via learned projection
   - Generative path: previous g + action → g_gen via transition
   - AbstractLocationInference: Precision-weighted fusion g_inf

4. Grounded Location Inference: Final place cells from abstract location
   - ProjectionHead: Transform and downsample g_inf
   - GroundedLocationInference: p = g_inf ⊗ x_f (outer product)

5. Memory Update: Hebbian learning for next timestep
   - MemoryStorage: M ← λM + η·(p ⊗ p)

Data Flow (Theory):
-------------------
    x (observation)
    → x_c (compressed/two-hot encoding)
    → x_f (temporal filtering per frequency)
    → ~x_t (sensory projected to p-space via W_tile)
    → p_x (hippocampal retrieval from sensory: M^T @ ~x_t)
    → g_inf (abstract location from p_x and g_gen fusion)
    → p (final grounded location: g_inf ⊗ x_f)
    → M (Hebbian memory update)

Usage Examples:
---------------
    # Default: 50 timesteps, memory enabled, save plots
    python examples/inference.py

    # Longer walk with different architecture
    python examples/inference.py --walk_length 100 --n_frequencies 4

    # Different grid size and observation mode
    python examples/inference.py --grid_size 7 --observation_mode tiled

    # Show plots interactively
    python examples/inference.py --show_plots true --save_plots false

    # Full help
    python examples/inference.py --help

Outputs:
--------
When save_plots=true, generates 7 visualizations in outputs/inference/:
    1. 01_environment.png - Grid layout
    2. 02_walk_trajectory.png - Agent trajectory
    3. 03_sensory_processing.png - Temporal filtering heatmaps
    4. 04_place_cell_activity.png - Place field evolution
    5. 05_outer_product_structure.png - Decomposition at mid-point
    6. 06_abstract_location.png - Abstract location over time
    7. 07_memory_matrices.png - Hebbian associations
"""

from pathlib import Path
from typing import List, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from torch import Tensor

from torch_tem import data, figures, utils
from torch_tem.core.encoder import SensoryEncoder
from torch_tem.core.projection import ProjectionHead
from torch_tem.core.tiling import SensoryProjection
from torch_tem.inference.abstract import AbstractLocationInference
from torch_tem.inference.grounded import GroundedLocationInference
from torch_tem.inference.sensory import SensoryProcessor
from torch_tem.memory.attractor import AttractorDynamics
from torch_tem.memory.storage import MemoryStorage


# ==============================================================================
# Configuration
# ==============================================================================
class ExampleConfig(BaseSettings):
    """Configuration for complete TEM inference pipeline example.

    This config implements all inference-related protocols:
    - EncoderParams, SensoryProcessorParams
    - GroundedInferenceParams, ProjectionParams
    - AbstractInferenceParams
    - MemoryStorageParams, AttractorParams
    """

    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True, cli_prog_name="inference")

    # Environment configuration
    grid_size: int = Field(default=5, ge=3, le=10, description="Grid size for synthetic environment")
    observation_mode: Literal["unique", "tiled", "random"] = Field(default="unique", description="Observation generation mode")

    # Walk generation
    walk_length: int = Field(default=100, ge=20, le=500, description="Steps in the walk sequence")

    # Architecture configuration
    n_frequencies: int = Field(default=3, ge=2, le=5, description="Number of frequency modules")
    n_g_per_module: List[int] = Field(default_factory=lambda: [12, 10, 8], description="Grid cell dimensions per frequency")
    n_x_c: int = Field(default=8, ge=2, le=20, description="Compressed sensory dimension (two-hot)")

    # Frequency configuration
    f_min: float = Field(default=0.1, ge=0.01, le=0.5, description="Minimum frequency (longest memory)")
    f_max: float = Field(default=0.9, ge=0.5, le=1.0, description="Maximum frequency (shortest memory)")

    # Memory configuration
    eta: float = Field(default=0.3, ge=0.0, le=1.0, description="Hebbian learning rate")
    lambda_: float = Field(default=0.95, ge=0.0, le=1.0, description="Memory decay rate")
    kappa: float = Field(default=0.8, ge=0.0, le=1.0, description="Attractor stability parameter")

    # Abstract inference configuration
    g_mem_std: float = Field(default=0.01, gt=0, description="Std for memory MLP initialization")
    g_init_std: float = Field(default=0.1, gt=0, description="Std for learnable g_init")
    p2g_scale_offset: float = Field(default=0.5, ge=0.0, le=5.0, description="Memory influence schedule offset")

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
    # Computed Properties
    # ==============================================================================

    @computed_field(description="Number of unique observations in the environment")
    @property
    def n_x(self) -> int:
        n_locations = self.grid_size * self.grid_size
        if self.observation_mode == "unique":
            return n_locations
        elif self.observation_mode == "tiled":
            return 4
        elif self.observation_mode == "random":
            return max(4, n_locations // 4)
        raise ValueError(f"Invalid observation_mode: {self.observation_mode}")

    @computed_field(description="Total number of frequency modules")
    @property
    def n_f_calculated(self) -> int:
        return self.n_frequencies

    @computed_field(description="Grid cell dimensions per frequency")
    @property
    def n_g_calculated(self) -> List[int]:
        if len(self.n_g_per_module) < self.n_frequencies:
            return self.n_g_per_module + [self.n_g_per_module[-1]] * (self.n_frequencies - len(self.n_g_per_module))
        return self.n_g_per_module[: self.n_frequencies]

    @computed_field(description="Downsampled grid cell dimensions (1/3 of full)")
    @property
    def n_g_subsampled_combined(self) -> List[int]:
        return [max(1, n_g // 3) for n_g in self.n_g_calculated]

    @computed_field(description="Place cell dimensions per frequency")
    @property
    def n_p_calculated(self) -> List[int]:
        return [n_g * self.n_x_c for n_g in self.n_g_subsampled_combined]

    @computed_field(description="Sensory dimensions per frequency")
    @property
    def n_x_f_calculated(self) -> List[int]:
        return [self.n_x_c for _ in range(self.n_frequencies)]

    @computed_field(description="Frequency values (logarithmic spacing)")
    @property
    def f_initial_extended(self) -> List[float]:
        if self.n_frequencies == 1:
            return [self.f_max]
        return np.logspace(np.log10(self.f_min), np.log10(self.f_max), self.n_frequencies).tolist()

    @computed_field(description="Two-hot encoding table")
    @property
    def two_hot_table_calculated(self) -> List[Tensor]:
        return [torch.tensor([1.0 if j in {(i % self.n_x_c), ((i + 1) % self.n_x_c)} else 0.0 for j in range(self.n_x_c)]) for i in range(self.n_x)]

    @computed_field(description="Kronecker repeat matrices for outer product")
    @property
    def W_repeat_calculated(self) -> List[Tensor]:
        return utils.create_W_repeat(self.n_g_subsampled_combined, self.n_x_f_calculated)

    @computed_field(description="Kronecker tile matrices for outer product")
    @property
    def W_tile_calculated(self) -> List[Tensor]:
        return utils.create_W_tile(self.n_g_subsampled_combined, self.n_x_f_calculated)

    @computed_field(description="Downsampling matrices for grid cells")
    @property
    def g_downsample_calculated(self) -> List[Tensor]:
        return [torch.randn(self.n_g_calculated[f], self.n_g_subsampled_combined[f]) / np.sqrt(self.n_g_calculated[f]) for f in range(self.n_frequencies)]

    @computed_field(description="Memory update mask (hierarchical)")
    @property
    def p_update_mask_calculated(self) -> Tensor:
        return utils.create_p_update_mask(self.n_p_calculated, self.n_frequencies, self.n_frequencies, 0, self.f_initial_extended)

    @computed_field(description="Attractor retrieval iterations")
    @property
    def i_attractor_calculated(self) -> int:
        return 2 * self.n_frequencies

    @computed_field(description="Max iterations per frequency for inference retrieval")
    @property
    def i_attractor_max_freq_inf(self) -> List[int]:
        # Early stopping: low frequencies stop early, high frequencies continue longer
        return [2 * (f + 1) for f in range(self.n_frequencies)]

    @computed_field(description="Max iterations per frequency for generative retrieval")
    @property
    def i_attractor_max_freq_gen(self) -> List[int]:
        return self.i_attractor_max_freq_inf  # Use same schedule

    @computed_field(description="Hierarchical masks for inference and generative retrieval")
    @property
    def p_retrieve_masks_calculated(self) -> tuple[List[Tensor], List[Tensor]]:
        """Returns (masks_inf, masks_gen) from single function call."""
        return utils.create_p_retrieve_masks(self.n_p_calculated, self.i_attractor_calculated, self.i_attractor_max_freq_inf, self.i_attractor_max_freq_gen)

    @computed_field(description="Inference retrieval masks")
    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        return self.p_retrieve_masks_calculated[0]

    @computed_field(description="Generative retrieval masks")
    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        return self.p_retrieve_masks_calculated[1]

    @computed_field(description="Use inference-based grounded locations")
    @property
    def use_p_inf(self) -> bool:
        return True  # This example always uses memory

    @computed_field(description="Use common memory for inference and generation")
    @property
    def common_memory(self) -> bool:
        return False  # Use dual memory


# ==============================================================================
# Main Experiment
# ==============================================================================
if __name__ == "__main__":
    """Run the complete TEM inference pipeline with visualizations."""
    config = ExampleConfig()

    print("=" * 80)
    print("Complete TEM Inference Pipeline")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Environment: {config.grid_size}×{config.grid_size} grid ({config.observation_mode} observations)")
    print(f"  Walk length: {config.walk_length} timesteps")
    print(f"  Frequencies: {config.n_frequencies} ({config.f_min:.2f} to {config.f_max:.2f})")
    print(f"  Architecture: n_g={config.n_g_calculated}, n_p={config.n_p_calculated}, n_x_c={config.n_x_c}")
    print()

    # =========================================================================
    # PHASE 1: Environment and Walk Generation
    # =========================================================================
    print("Phase 1: Generating walk trajectory...")
    env = data.Environment.from_grid(config.grid_size, config.grid_size, config.observation_mode)
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
    print("Phase 2: Initializing inference components...")

    # Sensory processing
    encoder = SensoryEncoder(config)
    processor = SensoryProcessor(config)
    print(f"  ✓ SensoryEncoder: {config.n_x} → {config.n_x_c} (two-hot)")
    print(f"  ✓ SensoryProcessor: {config.n_frequencies} frequency channels")

    # Sensory projection to p-space
    sensory_projection = SensoryProjection(config)
    print(f"  ✓ SensoryProjection: x_f → ~x_t (W_tile transformation to p-space)")

    # Grounded location inference
    projection = ProjectionHead(config)
    grounded = GroundedLocationInference(config)
    print(f"  ✓ ProjectionHead: Laplacian transform + downsampling")
    print(f"  ✓ GroundedLocationInference: g ⊗ x → p (includes W_tile internally)")

    # Memory system
    storage = MemoryStorage(config)
    attractor = AttractorDynamics(config)
    print(f"  ✓ MemoryStorage: {sum(config.n_p_calculated)}×{sum(config.n_p_calculated)} Hebbian matrix")
    print(f"  ✓ AttractorDynamics: {config.i_attractor_calculated} iterations with hierarchical masking")

    # Abstract location inference
    abstract = AbstractLocationInference(config)
    print(f"  ✓ AbstractLocationInference: precision-weighted fusion")
    print()

    # =========================================================================
    # PHASE 3: Generate Synthetic Grid Cell Patterns
    # =========================================================================
    print("Phase 3: Generating synthetic grid cell patterns...")
    grid_generator = data.SyntheticGridGenerator(config, config.walk_length, batch_size=1)
    g_history = grid_generator.generate()  # List[T] of List[n_f] of [1, n_g[f]]
    print(f"  ✓ Generated {config.walk_length} timesteps of grid cell activity")
    print()

    # =========================================================================
    # PHASE 4: Run Complete Inference Pipeline
    # =========================================================================
    print("Phase 4: Running complete inference pipeline...")

    x_c_history = []
    x_f_history = []
    x_projected_history = []  # ~x_t: sensory input projected to p-space
    p_x_history = []  # p_x: hippocampal patterns retrieved from sensory
    p_history = []
    g_inf_history = []

    x_prev = [torch.zeros(1, config.n_x_c) for _ in range(config.n_frequencies)]

    for t in range(config.walk_length):
        # Step 1: Encode observation → compressed sensory
        x_t = observations[t].unsqueeze(0)  # [n_x] → [1, n_x]
        x_c = encoder(x_t)  # [1, n_x_c]

        # Step 2: Temporal filtering → multi-frequency representation
        x_f = processor(x_c, x_prev)  # List[n_f] of [1, n_x_c]

        # Step 3: Project sensory to p-space (theory: ~x_t = W_tile * w_p * f_n(x_f))
        x_projected = sensory_projection(x_f)  # List[n_f] of [1, n_p[f]]
        x_proj_concat = torch.cat(x_projected, dim=1)  # [1, sum(n_p)]

        # Step 4: Retrieve hippocampal patterns from sensory input via attractor
        M_inf = storage.get_memory(for_inference=True)
        p_x_concat = attractor.retrieve(x_proj_concat, M_inf, for_inference=True)  # [1, sum(n_p)]

        # Split retrieved patterns back to per-frequency lists
        p_x = []
        start_idx = 0
        for f in range(config.n_frequencies):
            end_idx = start_idx + config.n_p_calculated[f]
            p_x.append(p_x_concat[:, start_idx:end_idx])  # [1, n_p[f]]
            start_idx = end_idx

        # Step 5: Get synthetic grid cells at time t (for generative path)
        g_t = g_history[t]  # List[n_f] of [1, n_g[f]]

        # Step 6: Infer abstract location g from memory-retrieved p_x and generative g_t
        # Theory: g is inferred by precision-weighted fusion of:
        #   - g_gen (from transition/generative model)
        #   - g_mem (from p_x via learned projection p→g)

        # For g_mem path: project p_x to abstract location space
        # Theory: g_downsampled = p_x @ W_repeat^T (sum over sensory preferences)
        g_mem_downsampled = [torch.matmul(p_x[f], config.W_repeat_calculated[f].t()) for f in range(config.n_frequencies)]  # List[n_f] of [1, n_g_sub[f]]

        # Use g_t as generative prediction with moderate uncertainty
        sigma_gen = [torch.ones(1, n_g) * 0.5 for n_g in config.n_g_calculated]

        # Fuse generative and memory paths to infer abstract location
        g_inf = abstract(g_t, sigma_gen, g_mem_downsampled, shiny_signals=None, p2g_scale_offset=config.p2g_scale_offset)  # generative path  # memory path from p_x

        # Step 7: Transform and downsample inferred g for final grounded inference
        g_transformed = projection.transform(g_inf)
        g_downsampled = projection.downsample(g_transformed)

        # Step 8: Compute final grounded location via outer product g ⊗ x
        p_t = grounded(g_downsampled, x_f)  # List[n_f] of [1, n_p[f]]
        p_concat = torch.cat(p_t, dim=1)  # [1, sum(n_p)]

        # Step 9: Update memory with Hebbian learning
        storage.update(p_concat, p_concat, eta=config.eta, lamb=config.lambda_)

        # Store history (extract batch dimension for single-trajectory storage)
        x_c_history.append(x_c[0])
        x_f_history.append([x[0] for x in x_f])
        x_projected_history.append([x[0] for x in x_projected])
        p_x_history.append([p[0] for p in p_x])
        p_history.append([p[0] for p in p_t])
        g_inf_history.append(g_inf)

        x_prev = x_f

    print(f"  ✓ Processed {config.walk_length} timesteps through complete pipeline")
    print()

    # =========================================================================
    # PHASE 5: Generate Visualizations
    # =========================================================================
    print("Phase 5: Generating visualizations...")

    # Plot 1 & 2: Environment and walk trajectory
    fig1 = figures.plot_environment_layout(env, title=f"Environment: {config.grid_size}×{config.grid_size} Grid")
    fig2 = figures.plot_walks(env, [walk], title=f"Walk Trajectory ({config.walk_length} steps)")
    if config.save_plots:
        fig1.savefig(config.output_dir / "01_environment.png", dpi=150, bbox_inches="tight")
        fig2.savefig(config.output_dir / "02_walk_trajectory.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 01_environment.png, 02_walk_trajectory.png")

    # Plot 3: Sensory processing (temporal filtering)
    fig3 = figures.plot_temporal_filtering(x_c_history, x_f_history, config.f_initial_extended)
    if config.save_plots:
        fig3.savefig(config.output_dir / "03_sensory_processing.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 03_sensory_processing.png")

    # Plot 4: Grounded location activity (place cells)
    fig4 = figures.plot_grounded_location_activity(p_history, observations, locations, config.f_initial_extended, config.n_p_calculated)
    if config.save_plots:
        fig4.savefig(config.output_dir / "04_place_cell_activity.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 04_place_cell_activity.png")

    # Plot 5: Outer product structure (mid-point)
    mid_point = config.walk_length // 2
    g_mid = g_history[mid_point]  # List[n_f] of [1, n_g[f]]
    g_mid_downsampled = projection.downsample(projection.transform(g_mid))
    g_sample = [g_mid_downsampled[f][0] for f in range(config.n_frequencies)]
    fig5 = figures.plot_outer_product_structure(g_sample, x_f_history[mid_point], p_history[mid_point], config.f_initial_extended)
    if config.save_plots:
        fig5.savefig(config.output_dir / "05_outer_product_structure.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 05_outer_product_structure.png")

    # Plot 6: Abstract location evolution
    fig6 = figures.plot_g_inf_evolution(g_inf_history, config.n_frequencies, config.walk_length)
    if config.save_plots:
        fig6.savefig(config.output_dir / "06_abstract_location.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 06_abstract_location.png")

    # Plot 7: Memory matrices
    fig7 = figures.plot_memory_matrices(storage.M_gen, storage.get_memory(for_inference=True), config.n_p_calculated, config.walk_length)
    if config.save_plots:
        fig7.savefig(config.output_dir / "07_memory_matrices.png", dpi=150, bbox_inches="tight")
        print(f"  Saved: 07_memory_matrices.png")

    print()
    print("=" * 80)
    print("Pipeline Summary:")
    print("=" * 80)
    print(f"Input:  {config.n_x}-dim observations ({config.observation_mode} mode)")
    print(f"  ↓ SensoryEncoder (two-hot)")
    print(f"Stage 1: {config.n_x_c}-dim compressed sensory (x_c)")
    print(f"  ↓ SensoryProcessor ({config.n_frequencies} frequencies)")
    print(f"Stage 2: Multi-frequency filtered sensory (x_f)")
    print(f"  ↓ SensoryProjection (~x_t = W_tile @ x_f)")
    print(f"Stage 3: Sensory input to hippocampus (~x_t) - {config.n_p_calculated}")
    print(f"  ↓ AttractorDynamics (M^T @ ~x_t)")
    print(f"Stage 4: Hippocampal patterns from sensory (p_x) - {config.n_p_calculated}")
    print(f"  ↓ Projection to abstract (p_x @ W_repeat^T + g_gen fusion)")
    print(f"Stage 5: Abstract location (g_inf) - {config.n_g_calculated}")
    print(f"  ↓ ProjectionHead (transform + downsample)")
    print(f"Stage 6: Downsampled abstract (g_sub) - {config.n_g_subsampled_combined}")
    print(f"  ↓ GroundedLocationInference (g ⊗ x_f)")
    print(f"Stage 7: Final grounded location (p) - {config.n_p_calculated}")
    print(f"  ↓ MemoryStorage (Hebbian update)")
    print(f"Output: Memory-stored patterns (M) for next timestep")
    print("=" * 80)
    print()
    print(f"All outputs saved to: {config.output_dir}")

    # Show or close plots
    if config.show_plots:
        plt.show()
    else:
        plt.close("all")
