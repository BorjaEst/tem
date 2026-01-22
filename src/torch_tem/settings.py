"""Pydantic settings models for TEM configuration.

This module defines leaf-level `*Settings` classes used to configure training,
data generation, schedules, and core component hyperparameters.

Notes:
    - Keep in-code docstrings focused on what each setting controls.
    - Longer architecture notes live in `docs/foundations.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field

from torch_tem.types import Reduction, Scalar

Activation = Literal["leaky_relu", "sigmoid", "none"]
ProjectionMode = Literal["identity", "tiling", "low_rank", "random"]
InitStrategy = Literal["identity", "random"]


class SpaceContractSettings(BaseModel):
    """Space contract settings: observation and action space dimensions.

    This defines the interface contract between the TEM model and environment files.
    All environment JSON files must match these dimensions for training to proceed.

    Action Encoding Semantics:
        When action0_is_noop=True (default):
            - Action 0 (and None) encode as all-zeros (no-op/static action)
            - Actions 1..n_actions_move encode as one-hot of length n_actions_move
            - Total actions in env files must be n_actions_move + 1
    """

    model_config = ConfigDict(extra="forbid")

    n_observations: int = Field(
        default=45,
        ge=1,
        description="Number of discrete observation states (model input dimensionality).",
    )
    n_actions_move: int = Field(
        default=4,
        ge=1,
        description="Number of movement actions (excluding no-op if action0_is_noop=True).",
    )
    action0_is_noop: bool = Field(
        default=True,
        description="Whether action 0 represents a no-op/self-loop (encoded as all-zeros).",
    )

    @computed_field
    @property
    def n_actions_total(self) -> int:
        """Total number of actions expected in environment files.

        Returns:
            n_actions_move + 1 if action0_is_noop, else n_actions_move
        """
        return self.n_actions_move + 1 if self.action0_is_noop else self.n_actions_move


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
    offset_min: float = Field(
        default=0.0,
        description="Minimum additive uncertainty offset applied to p->g inference.",
    )
    offset_max: float = Field(
        default=10000.0,
        description="Maximum additive uncertainty offset applied to p->g inference (schedule start).",
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

    use_x_cued_recall: bool = Field(  # TODO: We should remove this an allow p_xi to be None
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


class ReconstructionSettings(BaseModel):
    """Settings for LEC reconstruction modules."""

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
    reconstruction: ReconstructionSettings = Field(
        default_factory=ReconstructionSettings,
        description="Feature reconstruction module settings.",
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

    sigma_init: float = Field(
        default=0.1,
        frozen=True,
        description="Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location",
    )


class OVCSettings(BaseModel):
    """Settings for OVC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

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


class AttractorSettings(BaseModel):
    """Settings for attractor dynamics modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    kappa: float = Field(
        default=0.8,
        description="Hebbian retrieval decay term",
    )
    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for attractor dynamics.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for attractor dynamics.",
    )


class GroundLocSettings(BaseModel):
    """Settings for location distribution modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    activation: Activation = Field(
        default="leaky_relu",
        frozen=True,
        description="Activation function for attractor dynamics.",
    )
    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


class HebbianUpdateSettings(BaseModel):
    """Settings for Hebbian update modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    clamp_min: float = Field(
        default=-1.0,
        description="Minimum clamp value for Hebbian memory weights.",
    )
    clamp_max: float = Field(
        default=1.0,
        description="Maximum clamp value for Hebbian memory weights.",
    )


class HPCSettings(BaseModel):
    """Settings for HPC modules."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    common_memory: bool = Field(  # Probably to move to hebbian which will rename memory
        default=False,
        description="Use common memory for generative and inference network",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to sample from location distributions or use means.",
    )
    attractor: AttractorSettings = Field(
        default_factory=AttractorSettings,
        description="Attractor dynamics module settings.",
    )
    location: GroundLocSettings = Field(
        default_factory=GroundLocSettings,
        description="Location distribution module settings.",
    )
    memory: HebbianUpdateSettings = Field(
        default_factory=HebbianUpdateSettings,
        description="Hebbian update module settings.",
    )


class TEMSettings(BaseModel):
    """Complete settings tree for TEM model configuration."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    autoencoder: AutoencoderSettings = Field(
        default_factory=AutoencoderSettings,
        description="Autoencoder module settings.",
    )
    f_initial: List[float] = Field(
        default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01],
        frozen=True,
        description="Initial spatial frequencies for multi-scale modules.",
    )
    n_features: int = Field(
        default=10,
        frozen=True,
        description="Number of LEC context features.",
    )
    lec_settings: LECSettings = Field(
        default_factory=LECSettings,
        description="LEC module settings.",
    )
    lec_projection: LECProjectionSettings = Field(
        default_factory=LECProjectionSettings,
        description="LEC projection module settings.",
    )
    n_grids: List[int] = Field(
        default_factory=lambda: [30, 30, 24, 18, 18],
        frozen=True,
        description="Number of MEC neurons per frequency module.",
    )
    n_ovc: Union[Literal["off", "merged"], List[int]] = Field(
        default="merged",
        frozen=True,
        description="Number of OVC neurons per frequency module. 'merged' to merge with n_grids.",
    )
    mec_settings: MECSettings = Field(
        default_factory=MECSettings,
        description="MEC module settings.",
    )
    mec_projection: MECProjectionSettings = Field(
        default_factory=MECProjectionSettings,
        description="MEC projection module settings.",
    )
    n_hippocampal: List[int] = Field(
        default_factory=lambda: [100, 100, 80, 60, 60],
        frozen=True,
        description="Number of HPC neurons per frequency module.",
    )
    hpc_settings: HPCSettings = Field(
        default_factory=HPCSettings,
        description="HPC module settings.",
    )
    use_x_cued_recall: bool = Field(
        default=True,
        description="Whether to use inferred ground location while inferring new abstract location",
    )
