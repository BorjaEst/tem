"""Protocol-based interface facets for torch_tem components.

Each Protocol defines a minimal interface that components require, allowing them to
depend on narrow contracts instead of the full Parameters model. This enables:
- Better testability (use lightweight fakes instead of full Parameters)
- Clear documentation of component dependencies
- Loose coupling between components and configuration
- Type safety through structural typing

The Parameters model satisfies all these Protocols through duck typing.
"""

from typing import List, Protocol

from torch import Tensor

# =======================================================================================
# CORE COMPONENTS
# =======================================================================================


class EncoderParams(Protocol):
    """Minimal interface for SensoryEncoder.

    Dependencies: n_x, n_x_c, two_hot_table_calculated
    Complexity: Low (3 parameters)
    """

    n_x: int
    n_x_c: int

    @property
    def two_hot_table_calculated(self) -> List[Tensor]:
        """Table for converting one-hot to two-hot compressed representation."""
        ...


class DecoderParams(Protocol):
    """Minimal interface for ObservationDecoder.

    Dependencies: n_x, n_x_c, n_x_f_calculated
    Complexity: Low (3 parameters)
    """

    n_x: int
    n_x_c: int

    @property
    def n_x_f_calculated(self) -> List[int]:
        """Neurons for temporally filtered sensory experience x per frequency."""
        ...


class TransitionParams(Protocol):
    """Minimal interface for TransitionModel.

    Dependencies: n_f_calculated, n_g_calculated, n_actions, g_connections_calculated,
                  do_sample, g_init_std, g_mem_std, d_hidden_dim
    Complexity: Medium (8 parameters)
    """

    n_actions: int
    do_sample: bool
    g_init_std: float
    g_mem_std: float
    d_hidden_dim: int

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_g_calculated(self) -> List[int]:
        """Neurons of entorhinal abstract location g per frequency."""
        ...

    @property
    def g_connections_calculated(self) -> List[List[bool]]:
        """Hierarchical connections for abstract location module transitions."""
        ...


class ProjectionParams(Protocol):
    """Minimal interface for ProjectionHead.

    Dependencies: n_f_calculated, n_g_calculated, g_downsample_calculated, f_initial_extended
    Complexity: Low (4 parameters)
    """

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_g_calculated(self) -> List[int]:
        """Neurons of entorhinal abstract location g per frequency."""
        ...

    @property
    def g_downsample_calculated(self) -> List[Tensor]:
        """Downsampling matrix from grid cells to compressed grid cells."""
        ...

    @property
    def f_initial_extended(self) -> List[float]:
        """Extended f_initial with OVC module frequencies (if separate_ovc)."""
        ...


class SensoryProjectionParams(Protocol):
    """Minimal interface for SensoryProjection.

    Dependencies: n_f_calculated, n_x_f_calculated, W_tile_calculated
    Complexity: Low (3 parameters)
    """

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_x_f_calculated(self) -> List[int]:
        """Neurons for temporally filtered sensory experience x per frequency."""
        ...

    @property
    def W_tile_calculated(self) -> List[Tensor]:
        """Matrix for tiling sensory observation x for outer product with g."""
        ...


# =======================================================================================
# MEMORY SYSTEM
# =======================================================================================


class MemoryStorageParams(Protocol):
    """Minimal interface for MemoryStorage.

    Dependencies: n_p_calculated, p_update_mask_calculated, use_p_inf, common_memory
    Complexity: Low (4 parameters)
    """

    use_p_inf: bool
    common_memory: bool

    @property
    def n_p_calculated(self) -> List[int]:
        """Neurons for hippocampal grounded location p per frequency."""
        ...

    @property
    def p_update_mask_calculated(self) -> Tensor:
        """Hebbian memory connections mask (low to high frequency)."""
        ...


class AttractorParams(Protocol):
    """Minimal interface for AttractorDynamics.

    Dependencies: kappa, i_attractor_calculated, p_retrieve_mask_inf_calculated,
                  p_retrieve_mask_gen_calculated
    Complexity: Low (4 parameters)
    """

    kappa: float

    @property
    def i_attractor_calculated(self) -> int:
        """Number of attractor dynamics iterations."""
        ...

    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        """Mask for hierarchical memory retrieval in inference model."""
        ...

    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        """Mask for hierarchical memory retrieval in generative model."""
        ...


# =======================================================================================
# INFERENCE SYSTEM
# =======================================================================================


class AbstractInferenceParams(Protocol):
    """Minimal interface for AbstractLocationInference.

    Dependencies: n_f_calculated, n_g_calculated, n_g_subsampled_combined,
                  use_p_inf, g_init_std, g_mem_std
    Complexity: Medium (6 parameters)
    """

    use_p_inf: bool
    g_init_std: float
    g_mem_std: float

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_g_calculated(self) -> List[int]:
        """Neurons of entorhinal abstract location g per frequency."""
        ...

    @property
    def n_g_subsampled_combined(self) -> List[int]:
        """Combined n_g_subsampled with OVC neurons."""
        ...


class GroundedInferenceParams(Protocol):
    """Minimal interface for GroundedLocationInference.

    Dependencies: n_f_calculated, n_p_calculated, n_x_c, W_repeat_calculated, W_tile_calculated
    Complexity: Low (5 parameters)
    """

    n_x_c: int

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_p_calculated(self) -> List[int]:
        """Neurons for hippocampal grounded location p per frequency."""
        ...

    @property
    def W_repeat_calculated(self) -> List[Tensor]:
        """Matrix for repeating abstract location g for outer product with x."""
        ...

    @property
    def W_tile_calculated(self) -> List[Tensor]:
        """Matrix for tiling sensory observation x for outer product with g."""
        ...


class SensoryProcessorParams(Protocol):
    """Minimal interface for SensoryProcessor.

    Dependencies: n_f_calculated, n_x_c, n_x_f_calculated, f_initial_extended
    Complexity: Low (4 parameters)
    """

    n_x_c: int

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_x_f_calculated(self) -> List[int]:
        """Neurons for temporally filtered sensory experience x per frequency."""
        ...

    @property
    def f_initial_extended(self) -> List[float]:
        """Extended f_initial with OVC module frequencies (if separate_ovc)."""
        ...


# =======================================================================================
# GENERATION SYSTEM
# =======================================================================================


class LocationGeneratorParams(Protocol):
    """Minimal interface for LocationGenerator.

    Dependencies: n_f_calculated, n_p_calculated, do_sample
    Complexity: Low (3 parameters)
    """

    do_sample: bool

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_p_calculated(self) -> List[int]:
        """Neurons for hippocampal grounded location p per frequency."""
        ...


# Note: ObservationGenerator only depends on ObservationDecoder, not Parameters


# =======================================================================================
# SHINY OBJECT MODULE
# =======================================================================================


class ShinyProcessorParams(Protocol):
    """Minimal interface for ShinyProcessor.

    Dependencies: n_f_calculated, n_f_g_calculated, n_f_ovc_calculated,
                  n_g_calculated, separate_ovc, d_hidden_dim
    Complexity: Medium (6 parameters)
    """

    separate_ovc: bool
    d_hidden_dim: int

    @property
    def n_f_calculated(self) -> int:
        """Total number of frequency modules."""
        ...

    @property
    def n_f_g_calculated(self) -> int:
        """Number of hierarchical frequency modules for grid cells."""
        ...

    @property
    def n_f_ovc_calculated(self) -> int:
        """Number of hierarchical frequency modules for object vector cells."""
        ...

    @property
    def n_g_calculated(self) -> List[int]:
        """Neurons of entorhinal abstract location g per frequency."""
        ...


# =======================================================================================
# LOSS COMPUTATION
# =======================================================================================


class LossComputerParams(Protocol):
    """Minimal interface for LossComputer.

    Dependencies: loss_weights
    Complexity: Low (1 parameter)
    """

    loss_weights: Tensor


# =======================================================================================
# TRAINING SYSTEM
# =======================================================================================


class ScheduleParams(Protocol):
    """Minimal interface for schedule computation.

    Dependencies: Training configuration parameters for curriculum learning.
    Complexity: High (15+ parameters)
    """

    # Basic training
    train_it: int
    batch_size: int

    # Walk configuration
    walk_it_min: int
    walk_it_max: int

    @property
    def walk_it_window_calculated(self) -> float:
        """Calculated width of window for sampling walk lengths."""
        ...

    # Learning rate schedule
    lr_max: float
    lr_min: float
    lr_decay_rate: float
    lr_decay_steps: int

    # Loss weight configuration
    loss_weights_x: float
    loss_weights_p: float
    loss_weights_g: float
    loss_weights_reg_g: float
    loss_weights_reg_p: float

    # Curriculum schedules
    loss_weights_p_g_it: int
    loss_weights_reg_p_it: int
    loss_weights_reg_g_it: int
    eta_it: int
    lambda_it: int

    # Precision-weighted mean scheduling
    p2g_scale_offset: float
    p2g_sig_val: float
    p2g_sig_half_it: int
    p2g_sig_scale_it: int


# =======================================================================================
# COMPOSITE PROTOCOLS (for high-level orchestrators)
# =======================================================================================


class TEMParams(Protocol):
    """Composite interface for TEM orchestrator.

    This is essentially the full Parameters interface, but defined as a Protocol
    for consistency. In practice, you'll pass the full Parameters object to TEM.

    Components inside TEM use their specific narrow Protocols.
    """

    # World configuration
    n_actions: int
    has_static_action: bool

    # Architecture flags
    do_sample: bool
    use_p_inf: bool
    separate_ovc: bool

    # Dimensions
    n_x: int
    n_x_c: int
    n_ovc: List[int]

    # Computed dimensions
    @property
    def n_f_calculated(self) -> int: ...

    @property
    def n_f_g_calculated(self) -> int: ...

    @property
    def n_f_ovc_calculated(self) -> int: ...

    @property
    def n_g_calculated(self) -> List[int]: ...

    @property
    def n_g_subsampled_combined(self) -> List[int]: ...

    @property
    def n_p_calculated(self) -> List[int]: ...

    @property
    def n_x_f_calculated(self) -> List[int]: ...

    # Initialization
    g_init_std: float
    g_mem_std: float
    d_hidden_dim: int

    # Memory
    lambda_: float
    eta: float
    kappa: float
    common_memory: bool

    @property
    def i_attractor_calculated(self) -> int: ...

    # Computed matrices and masks
    @property
    def two_hot_table_calculated(self) -> List[Tensor]: ...

    @property
    def g_downsample_calculated(self) -> List[Tensor]: ...

    @property
    def g_connections_calculated(self) -> List[List[bool]]: ...

    @property
    def W_repeat_calculated(self) -> List[Tensor]: ...

    @property
    def W_tile_calculated(self) -> List[Tensor]: ...

    @property
    def p_update_mask_calculated(self) -> Tensor: ...

    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]: ...

    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]: ...

    @property
    def f_initial_extended(self) -> List[float]: ...

    # Loss weights
    loss_weights: Tensor
