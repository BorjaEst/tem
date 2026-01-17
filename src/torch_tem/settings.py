"""Low-level settings for TEM training and data generation.

This module contains Pydantic BaseModel settings classes that define configuration
for individual components of TEM training, data generation, and scheduling.

Architecture Note
-----------------
Settings hierarchy in torch_tem:

    1. **types.py**: Pure types with no dependencies from torch_tem
    2. **settings.py** (this module): Low-level '*Settings' Pydantic models
       - Depend only on types.py (for type hints like Reduction, Scalar)
       - Define leaf-level configuration for individual components
       - Shared across multiple higher-level modules
    3. **Module configs**: '*Config' classes in datamodule.py, training.py, etc.
       - Compose multiple '*Settings' from this module
       - Prevent parameter duplication across components
    4. **Entry points**: run.py, etc.
       - Use '*Settings' as sub-arguments to construct module '*Config' objects
       - Ensures single source of truth for shared parameters

Settings Classes
----------------
Environment & Data:
    - EnvironmentSettings: Environment JSON files and observation randomization
    - RolloutSettings: Batch size and rollout chunking
    - EvalSettings: Validation and test dataset configuration
    - ExplorationSettings: World exploration behavior
    - ShinySettings: Shiny environment generation
    - WalkCurriculumSettings: Walk length curriculum bounds

Training Schedules:
    - LRScheduleSettings: Learning rate schedule
    - HebbianScheduleSettings: Hebbian memory plasticity schedule
    - P2GOffsetScheduleSettings: Place-to-grid variance offset schedule

Loss Configuration:
    - SensoryReconstructionSettings: Sensory loss (L_x) settings
    - AbstractLocationSettings: Abstract location loss (L_g) settings
    - GroundedLocationSettings: Grounded location loss (L_p) settings
    - RegularizationSettings: Regularization penalties
    - LossSettings: Composes all loss component settings

Infrastructure:
    - LoggerSettings: TensorBoard logging configuration
    - CheckpointSettings: Model checkpointing configuration
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field

from torch_tem.types import Reduction, Scalar

Activation = Literal["sigmoid", "none"]
ProjectionMode = Literal["identity", "tiling", "low_rank", "random"]
InitStrategy = Literal["identity", "random"]


class EnvironmentSettings(BaseModel):
    """Environment generation settings."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    envs: list[Path] = Field(
        default_factory=lambda: [Path("./envs/10x10.json")],
        description="Environment JSON files for training.",
    )
    randomise_observations: bool = Field(
        default=True,
        description="Randomise observations in environments.",
    )


class RolloutSettings(BaseModel):
    """Batch and rollout chunking settings."""

    model_config = ConfigDict(extra="forbid")

    batch_size: int = Field(
        default=16,
        description="Number of parallel walks.",
    )
    n_rollout: int = Field(
        default=20,
        description="Steps per truncated BPTT chunk.",
    )


class EvalSettings(BaseModel):
    """Validation and test dataset settings."""

    model_config = ConfigDict(extra="forbid")

    enable_validation: bool = Field(
        default=False,
        description="Enable validation loop during training.",
    )
    enable_test: bool = Field(
        default=False,
        description="Enable test loop after training.",
    )
    val_batches: int = Field(
        default=10,
        description="Number of batches per validation epoch (finite).",
    )
    test_batches: int = Field(
        default=10,
        description="Number of batches per test epoch (finite).",
    )
    val_seed: int = Field(
        default=42,
        description="Random seed for validation dataset (deterministic).",
    )
    test_seed: int = Field(
        default=43,
        description="Random seed for test dataset (deterministic).",
    )


class ExplorationSettings(BaseModel):
    """World exploration behavior settings."""

    model_config = ConfigDict(extra="forbid")

    explore_bias: int = Field(
        default=2,
        description="Bias for explorative behaviour to pick the same action again, to encourage straight walks",
    )


class ShinySettings(BaseModel):
    """Shiny environment generation settings."""

    model_config = ConfigDict(extra="forbid")

    shiny_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability at which shiny environments occur during training (0.0 to 1.0).",
    )
    shiny_gamma: float = Field(
        default=0.7,
        description="Shiny policy gamma.",
    )
    shiny_beta: float = Field(
        default=1.5,
        description="Shiny policy beta.",
    )
    shiny_n: int = Field(
        default=2,
        description="Number of shiny objects.",
    )
    shiny_returns: int = Field(
        default=15,
        description="Returns to shiny object after finding.",
    )

    @computed_field
    @property
    def shiny_dict(self) -> dict:
        """Dict consumed by torch_tem.data.World(shiny=...)."""
        return {
            "gamma": self.shiny_gamma,
            "beta": self.shiny_beta,
            "n": self.shiny_n,
            "returns": self.shiny_returns,
        }


class WalkCurriculumSettings(BaseModel):
    """Walk length curriculum settings (data-owned, training-controlled)."""

    model_config = ConfigDict(extra="forbid")

    walk_it_min: int = Field(
        default=25,
        description="Minimum walk length multiplier (curriculum endpoint)",
    )
    walk_it_max: int = Field(
        default=300,
        description="Maximum walk length multiplier (curriculum start)",
    )

    @computed_field
    @property
    def walk_it_window(self) -> float:
        """Width of window from which walk lengths are sampled."""
        return 0.2 * (self.walk_it_max - self.walk_it_min)


class LRScheduleSettings(BaseModel):
    """Learning rate schedule settings."""

    model_config = ConfigDict(extra="forbid")

    lr_max: float = Field(
        default=9.4e-4,
        description="Maximum learning rate.",
    )
    lr_min: float = Field(
        default=8e-5,
        description="Minimum learning rate.",
    )
    lr_decay_rate: float = Field(
        default=0.5,
        description="Exponential decay rate.",
    )
    lr_decay_steps: int = Field(
        default=4000,
        description="Decay period in steps.",
    )


class HebbianScheduleSettings(BaseModel):
    """Hebbian memory plasticity schedule settings."""

    model_config = ConfigDict(extra="forbid")

    eta: float = Field(
        default=0.5,
        description="Base Hebbian rate of remembering",
    )
    eta_it: int = Field(
        default=16000,
        description="Eta schedule iteration count.",
    )
    hebbian_decay: float = Field(
        default=0.9999,
        description="Base Hebbian decay factor (rate of forgetting)",
    )
    lambda_it: int = Field(
        default=200,
        description="Lambda schedule iteration count.",
    )


class P2GOffsetScheduleSettings(BaseModel):
    """Place-to-grid variance offset schedule settings."""

    model_config = ConfigDict(extra="forbid")

    p2g_sig_half_it: int = Field(
        default=400,
        description="Half-life for p2g variance offset schedule.",
    )
    p2g_sig_scale_it: int = Field(
        default=200,
        description="Scale factor for p2g variance offset schedule.",
    )


class LoggerSettings(BaseModel):
    """Logging settings for TensorBoard logger."""

    model_config = ConfigDict(extra="forbid")

    save_dir: Path = Field(
        default=Path("./logs"),
        description="Directory to save logs (default: ./lightning_logs).",
    )
    name: Optional[str] = Field(
        default=None,
        description="Experiment name for logger.",
    )
    version: Optional[str] = Field(
        default=None,
        description="Version/run identifier (auto-increments if None).",
    )
    log_graph: bool = Field(
        default=False,
        description="Log model graph to TensorBoard.",
    )
    prefix: str = Field(
        default="",
        description="Prefix for all logged metrics.",
    )


class CheckpointSettings(BaseModel):
    """Settings for model checkpointing."""

    model_config = ConfigDict(extra="forbid")

    every_n_train_steps: int = Field(
        default=1000,
        description="Save a checkpoint every N training steps.",
    )
    save_last: bool = Field(
        default=True,
        description="Whether to always save the last checkpoint.",
    )


class SensoryReconstructionSettings(BaseModel):
    """Settings for sensory reconstruction loss ($L_x$)."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    reduction: Reduction = Field(
        default="none",
        description="Reduction for sensory reconstruction loss.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight multiplier for all L_x components.",
    )


class AbstractLocationSettings(BaseModel):
    """Settings for abstract location transition loss ($L_g$)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    mode: Literal["mse", "nll"] = Field(
        default="mse",
        description="Loss mode: 'mse' (legacy surrogate), 'nll' (with uncertainty).",
    )
    reduction: Reduction = Field(
        default="none",
        description="Reduction for abstract location loss.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight multiplier for L_g.transition.",
    )


class GroundedLocationSettings(BaseModel):
    """Settings for grounded location consistency loss ($L_p$)."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether to use inferred grounded location loss.",
    )

    reduction: Reduction = Field(
        default="none",
        description="Reduction for grounded location loss.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight multiplier for all L_p components.",
    )


class RegularizationSettings(BaseModel):
    """Settings for regularization penalties."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    reduction: Reduction = Field(
        default="none",
        description="Reduction for regularization losses.",
    )
    weight_g_l2: float = Field(
        default=0.01,
        ge=0,
        description="Weight for abstract location L2 penalty.",
    )
    weight_p_l1: float = Field(
        default=0.02,
        ge=0,
        description="Weight for grounded location L1 penalty.",
    )


class LossSettings(BaseModel):
    """Complete settings tree for TEM loss computation.

    Attributes:
        x: Settings for sensory reconstruction losses.
        g: Settings for abstract location transition losses.
        p: Settings for grounded location consistency losses.
        reg: Settings for regularization penalties.
    """

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    x: SensoryReconstructionSettings = Field(
        default_factory=SensoryReconstructionSettings,
        description="Sensory reconstruction loss settings.",
    )
    g: AbstractLocationSettings = Field(
        default_factory=AbstractLocationSettings,
        description="Abstract location loss settings.",
    )
    p: GroundedLocationSettings = Field(
        default_factory=GroundedLocationSettings,
        description="Grounded location loss settings.",
    )
    reg: RegularizationSettings = Field(
        default_factory=RegularizationSettings,
        description="Regularization loss settings.",
    )


class AutoencoderSettings(BaseModel):
    """Settings for autoencoder modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    encode_mode: Literal["two_hot"] = Field(
        "two_hot",
        description="Compression mode for observations (e.g., 'two_hot')",
    )
    decode_mode: Literal["mlp"] = Field(
        "mlp",
        description="Decompression mode for observations (e.g., 'nnet')",
    )


class ProjectionSettings(BaseModel):
    """Settings for projection modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    mode: ProjectionMode = Field(
        ...,
        description="Projection mode strategy",
    )
    init: InitStrategy = Field(
        default="identity",
        description="Initialization strategy: 'identity' (structured) or 'random'",
    )
    learnable: bool = Field(
        default=False,
        description="If True, projection matrices are learnable",
    )
    rank: Optional[int] = Field(
        default=None,
        description="Low_rank rank for low_rank mode (auto-derived via GCD if None)",
    )


class LECProjectionSettings(ProjectionSettings):
    """Settings for LEC projection modules."""

    mode: ProjectionMode = Field(
        default="tiling",
        description="LEC default: tiling",
    )
    activation: Activation = Field(
        default="sigmoid",
        description="LEC default: sigmoid",
    )


class FreqFilterSettings(BaseModel):
    """Settings for LEC frequency filtering modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class FeatureNormSettings(BaseModel):
    """Settings for LEC normalization modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


class LECSettings(BaseModel):
    """Settings for LEC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    filter: FreqFilterSettings = Field(
        default_factory=FreqFilterSettings,
        description="Feature Frequency filtering module settings.",
    )
    norm: FeatureNormSettings = Field(
        default_factory=FeatureNormSettings,
        description="Feature normalization module settings.",
    )


class MECProjectionSettings(ProjectionSettings):
    """Settings for MEC projection modules."""

    mode: ProjectionMode = Field(
        default="low_rank",
        description="MEC default: low_rank",
    )
    init: InitStrategy = Field(
        default="identity",
        description="MEC default: structured (downsample+repeat)",
    )
    rank: Optional[List[int]] = Field(
        # default=None,  we will replace by None in future by default to simplify setup coherence
        default=[10, 10, 8, 6, 6],  # same as n_g_subsampled_base in legacy for now
        description="Base neurons for subsampled entorhinal abstract location f_g(g) for each frequency module",
    )


class PathSettings(BaseModel):
    """Settings for path integration modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    hidden_dim: int = Field(
        default=20,
        frozen=True,
        description="Hidden dimension for transition MLP.",
    )


class P2GMemSettings(BaseModel):
    """Settings for MEC memory inference modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    curriculum_sigma: float = Field(
        default=10000.0,
        description="Additional value to offset standard deviation of inferred grounded location",
    )
    sigma_init: float = Field(
        default=0.1,
        frozen=True,
        description="Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location",
    )


class OVCSettings(BaseModel):
    """Settings for OVC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    n_freq: Optional[int] = Field(
        default=None,
        frozen=True,
        description="Number of OVC modules receiving shiny correction. None: all modules, 0: disable OVC, k>0: last k modules.",
    )
    hidden_dim: int = Field(
        default=20,
        frozen=True,
        description="Hidden dimension for shiny landmark cue processing (OVC correction).",
    )


class MECSettings(BaseModel):
    """Settings for MEC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    do_sample: bool = Field(
        default=False,
        description="Whether to sample from distributions (stochastic) or use means (deterministic). Sampling policy is centralized in MECModel.",
    )
    sigma_init: float = Field(
        default=0.5,
        frozen=True,
        description="Standard deviation for initializing grid cell activations.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum activation clamp for OVC cells.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum activation clamp for OVC cells.",
    )
    path: PathSettings = Field(
        default_factory=PathSettings,
        description="Path integration module settings.",
    )
    p2g: P2GMemSettings = Field(
        default_factory=P2GMemSettings,
        description="Place-to-grid memory inference module settings.",
    )
    ovc: OVCSettings = Field(
        default_factory=OVCSettings,
        description="OVC module settings.",
    )


class HPCSettings(BaseModel):
    """Settings for HPC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether to use inferred ground location while inferring new abstract location",
    )

    common_memory: bool = Field(
        default=False,
        description="Use common memory for generative and inference network",
    )
    kappa: float = Field(
        default=0.8,
        description="Hebbian retrieval decay term",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to sample, or assume no noise and simply take mean of all distributions",
    )
