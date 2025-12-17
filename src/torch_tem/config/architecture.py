"""Architecture configuration for the Temporal Experience Model (TEM)."""

import math
from typing import List

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from torch import Tensor

from torch_tem import utils


class ModelConfig(BaseModel):
    """Neural architecture configuration: dimensions, modules, connectivity, and memory structure.

    This defines the static structure of the model that would be serialized with trained weights.
    All connectivity masks and static matrices are computed from the base architectural parameters.
    """

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True)

    # ===================================================================================
    # BASE DIMENSIONS
    # ===================================================================================

    batch_size: int = Field(default=4, ge=1, description="Batch size for training and inference")
    n_x: int = Field(default=45, ge=1, description="Number of sensory observation neurons x")
    n_x_c: int = Field(default=10, ge=1, description="Number of compressed sensory neurons x_c")
    n_g_subsampled: List[int] = Field(default_factory=lambda: [10, 10, 8, 6, 6], description="Subsampled grid cells per frequency module")
    n_ovc: List[int] = Field(default_factory=list, description="Object-vector cells per module; merged into grid modules when separate_ovc=False")
    f_initial: List[float] = Field(default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01], description="Base module frequencies (higher = higher spatial frequency)")
    separate_ovc: bool = Field(default=False, description="If True, allocate separate modules for OVC instead of merging into grid modules")

    # ===================================================================================
    # NETWORK INITIALISATION
    # ===================================================================================

    g_init_std: float = Field(default=0.5, gt=0, description="Std of initial abstract location g (before learning)")
    g_mem_std: float = Field(default=0.1, gt=0, description="Std for MLP hidden→output weights in g transition network")
    d_hidden_dim: int = Field(default=20, ge=1, description="Hidden layer width of the abstract-location transition MLP")
    n_actions: int = Field(default=5, ge=1, description="Total number of actions (can include 'stay' action)")

    # ===================================================================================
    # MEMORY STRUCTURE
    # ===================================================================================

    common_memory: bool = Field(default=False, description="Share a single memory between generative and inference networks")

    # ===================================================================================
    # DERIVED DIMENSIONS (computed properties)
    # ===================================================================================
    # These are computed from base dimensions and define the full architecture.
    # Order: OVC structure → module counts → neuron counts per module → attractor dynamics

    @computed_field(description="Grid + OVC subsampled cell counts per module")
    @property
    def n_g_subsampled_combined(self) -> List[int]:
        if not self.n_ovc:
            return self.n_g_subsampled
        if self.separate_ovc:
            return self.n_g_subsampled + self.n_ovc
        return [grid + ovc for grid, ovc in zip(self.n_g_subsampled, self.n_ovc)]

    @computed_field(description="Number of hierarchical frequency modules that are OVC-only")
    @property
    def n_f_ovc(self) -> int:
        if not self.n_ovc:
            return 0
        return len(self.n_ovc) if self.separate_ovc else 0

    @computed_field(description="Number of hierarchical frequency modules for standard grid cells")
    @property
    def n_f_g(self) -> int:
        return len(self.n_g_subsampled_combined) - self.n_f_ovc

    @computed_field(description="Total number of frequency modules (grid + optional OVC)")
    @property
    def n_f(self) -> int:
        return len(self.n_g_subsampled_combined)

    @computed_field(description="Extended frequency list including OVC modules when they are separate")
    @property
    def f_extended(self) -> List[float]:
        if self.separate_ovc and self.n_ovc:
            return self.f_initial + self.f_initial[0 : self.n_f_ovc]
        return self.f_initial

    @computed_field(description="Entorhinal abstract location neurons per frequency (3 × n_g_subsampled_combined)")
    @property
    def n_g(self) -> List[int]:
        return [3 * g for g in self.n_g_subsampled_combined]

    @computed_field(description="Temporally filtered sensory neurons x_f per frequency (shares n_x_c)")
    @property
    def n_x_f(self) -> List[int]:
        return [self.n_x_c for _ in range(self.n_f)]

    @computed_field(description="Hippocampal grounded location neurons p per frequency (outer product g × x_f)")
    @property
    def n_p(self) -> List[int]:
        return [g * x for g, x in zip(self.n_g_subsampled_combined, self.n_x_f)]

    @computed_field(description="Two-hot encoding lookup table for sensory compression")
    @property
    def two_hot_table(self) -> List[Tensor]:
        from torch_tem import utils

        return utils.create_two_hot_table(self.n_x, self.n_x_c)

    @computed_field(description="Number of attractor iterations for memory retrieval (equals n_f_g)")
    @property
    def i_attractor(self) -> int:
        return self.n_f_g

    @computed_field(description="Per-frequency attractor iteration cap for the inference model")
    @property
    def max_freq_inf(self) -> List[int]:
        attractor = self.i_attractor
        return [attractor for _ in range(self.n_f)]

    @computed_field(description="Per-frequency attractor iteration cap for the generative model (OVC not early-stopped)")
    @property
    def max_freq_gen(self) -> List[int]:
        attractor = self.i_attractor
        n_f_g = self.n_f_g
        n_f_ovc = self.n_f_ovc
        return [attractor - freq_nr for freq_nr in range(n_f_g)] + [attractor for _ in range(n_f_ovc)]

    # ===================================================================================
    # VALIDATION
    # ===================================================================================

    @model_validator(mode="after")
    def validate_frequency_configuration(self) -> "ModelConfig":
        """Validate that base frequency configuration is self-consistent.

        Ensures that:
        - ``f_initial`` has the same length as ``n_g_subsampled`` (the grid
          frequency modules), since these jointly define the base modules.
        - When OVCs are separate, the extended frequency list
          (``f_extended``) matches the total number of modules
          (``n_f``).
        - ``n_x_c`` is large enough to support two-hot encoding of all observations.
        """

        if len(self.f_initial) != len(self.n_g_subsampled):
            raise ValueError("Length of f_initial must match length of n_g_subsampled; " f"got {len(self.f_initial)} frequencies and " f"{len(self.n_g_subsampled)} grid modules.")

        if self.separate_ovc and self.n_ovc and len(self.f_extended) != self.n_f:
            raise ValueError("When separate_ovc is True and n_ovc is non-empty, the " "extended frequency list (f_extended) must have " "one entry per module (n_f).")

        # Validate two-hot encoding capacity
        max_two_hot_codes = int(math.comb(self.n_x_c, 2))
        if self.n_x > max_two_hot_codes:
            min_n_x_c = math.ceil((1 + math.sqrt(1 + 8 * self.n_x)) / 2)
            raise ValueError(
                f"Insufficient two-hot encoding capacity: n_x={self.n_x} observations "
                f"requires at least C(n_x_c, 2)={self.n_x} unique codes, "
                f"but n_x_c={self.n_x_c} only provides C({self.n_x_c}, 2)={max_two_hot_codes} codes. "
                f"Minimum required: n_x_c={min_n_x_c}. "
                f"Solutions: (1) Increase n_x_c to {min_n_x_c}+, "
                f"(2) Use tiled/random observation_mode to reduce unique observations, "
                f"(3) Reduce grid size."
            )

        return self

    # ===================================================================================
    # INFERENCE BEHAVIOUR
    # ===================================================================================

    do_sample: bool = Field(default=False, description="If False, use distribution means instead of sampling (no observation noise)")
    p2g_scale_offset: float = Field(default=0.0, ge=0, description="Variance offset scaling for memory path during abstract inference (controls memory influence)")
    p2g_sig_val: float = Field(default=10000.0, ge=0, description="Base variance magnitude for memory-derived abstract location uncertainty")

    # ===================================================================================
    # MEMORY DYNAMICS (can be tuned at inference time)
    # ===================================================================================

    eta: float = Field(default=0.5, ge=0, le=1, description="Hebbian rate of remembering (η in memory update)")
    lambda_: float = Field(default=0.9, ge=0, le=1, description="Hebbian memory retention factor (λ in memory decay)")
    kappa: float = Field(default=0.8, ge=0, le=1, description="Hebbian retrieval decay term κ")
