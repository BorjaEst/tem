from typing import Any, List

from pydantic import BaseModel, ConfigDict, Field, computed_field
from torch import Tensor, tensor, zeros

from torch_tem import utils


# -----------------------------------------------------------------------------------
class Parameters(BaseModel):
    """Parameters for the Temporal Experience Model (TEM)."""

    model_config = ConfigDict(extra="forbid", strict=True, arbitrary_types_allowed=True, populate_by_name=True)

    # ===================================================================================
    # WORLD CONFIGURATION
    # ===================================================================================
    # Parameters defining the environment structure and agent behavior settings.
    # These control action space, exploration strategies, and reward-driven behaviors.

    # Action space configuration
    has_static_action: bool = Field(default=True, description="Whether this world includes the standing still action")
    n_actions: int = Field(default=4, ge=1, description="Number of available actions (excluding stand still)")
    explore_bias: float = Field(default=2.0, ge=0, description="Bias to encourage repeating same action for straight walks")

    # Shiny object parameters (reward-driven behavior)
    shiny_rate: float = Field(default=0.0, ge=0, description="Rate of environments with shiny objects (0 for none)")
    shiny_gamma: float = Field(default=0.7, ge=0, le=1, description="Discount factor for Q-values in shiny object behaviour")
    shiny_beta: float = Field(default=1.5, ge=0, description="Inverse temperature for action selection with shiny objects")
    shiny_n: int = Field(default=2, ge=0, description="Number of shiny objects in the arena")
    shiny_returns: int = Field(default=15, ge=0, description="Number of times to return to a shiny object after finding it")
    shiny: dict[str, Any] = Field(default_factory=dict, description="Grouped shiny parameters for world object")

    @computed_field(description="Grouped shiny parameters dictionary for world object")
    @property
    def shiny_dict(self) -> dict[str, Any]:
        return {"gamma": self.shiny_gamma, "beta": self.shiny_beta, "n": self.shiny_n, "returns": self.shiny_returns}

    # ===================================================================================
    # NEURAL ARCHITECTURE
    # ===================================================================================
    # Network structure parameters defining neuron counts and module organization.
    # These specify the dimensionality of representations across different neural populations.

    # Sensory processing dimensions
    n_x: int = Field(default=45, ge=1, description="Neurons for sensory observation x")
    n_x_c: int = Field(default=10, ge=1, description="Neurons for compressed sensory experience x_c")
    n_x_f: List[int] = Field(default_factory=list, description="Neurons for temporally filtered sensory experience x per frequency")

    # Abstract location (grid cell) dimensions
    n_g_subsampled: List[int] = Field(default_factory=lambda: [10, 10, 8, 6, 6], description="Neurons for subsampled entorhinal abstract location f_g(g) per frequency module")
    n_g: List[int] = Field(default_factory=list, description="Neurons of entorhinal abstract location g per frequency")

    # Grounded location (place cell) dimensions
    n_p: List[int] = Field(default_factory=list, description="Neurons for hippocampal grounded location p per frequency")

    # Object vector cell dimensions
    n_ovc: List[int] = Field(default_factory=list, description="Neurons for object vector cells (added to new or existing modules)")

    # Module organization
    n_f_g: int = Field(default=5, ge=1, description="Number of hierarchical frequency modules for grid cells")
    n_f_ovc: int = Field(default=0, ge=0, description="Number of hierarchical frequency modules for object vector cells")
    n_f: int = Field(default=5, ge=1, description="Total number of modules")
    f_initial: List[float] = Field(default_factory=lambda: [0.99, 0.3, 0.09, 0.03, 0.01], description="Initial frequencies per module (higher number = higher frequency)")

    @computed_field(description="Combined n_g_subsampled with OVC neurons (concatenate if separate_ovc, else add)")
    @property
    def n_g_subsampled_combined(self) -> List[int]:
        if not self.n_ovc:
            return self.n_g_subsampled
        if self.separate_ovc:
            return self.n_g_subsampled + self.n_ovc
        return [grid + ovc for grid, ovc in zip(self.n_g_subsampled, self.n_ovc)]

    @computed_field(description="Number of hierarchical frequency modules for object vector cells")
    @property
    def n_f_ovc_calculated(self) -> int:
        if not self.n_ovc:
            return 0
        return len(self.n_ovc) if self.separate_ovc else 0

    @computed_field(description="Number of hierarchical frequency modules for grid cells")
    @property
    def n_f_g_calculated(self) -> int:
        return len(self.n_g_subsampled_combined) - self.n_f_ovc_calculated

    @computed_field(description="Total number of frequency modules")
    @property
    def n_f_calculated(self) -> int:
        return len(self.n_g_subsampled_combined)

    @computed_field(description="Neurons of entorhinal abstract location g per frequency (3x n_g_subsampled)")
    @property
    def n_g_calculated(self) -> List[int]:
        return [3 * g for g in self.n_g_subsampled_combined]

    @computed_field(description="Neurons for temporally filtered sensory experience x per frequency")
    @property
    def n_x_f_calculated(self) -> List[int]:
        return [self.n_x_c for _ in range(self.n_f_calculated)]

    @computed_field(description="Neurons for hippocampal grounded location p per frequency")
    @property
    def n_p_calculated(self) -> List[int]:
        return [g * x for g, x in zip(self.n_g_subsampled_combined, self.n_x_f_calculated)]

    @computed_field(description="Extended f_initial with OVC module frequencies (if separate_ovc)")
    @property
    def f_initial_extended(self) -> List[float]:
        if self.separate_ovc and self.n_ovc:
            return self.f_initial + self.f_initial[0 : self.n_f_ovc_calculated]
        return self.f_initial

    # ===================================================================================
    # MODEL BEHAVIOR
    # ===================================================================================
    # High-level configuration controlling model inference and generation modes.
    # These flags determine whether the model uses stochastic sampling or deterministic means.

    do_sample: bool = Field(default=False, description="Whether to sample or use mean of all distributions (no noise)")
    use_p_inf: bool = Field(default=True, description="Whether to use inferred ground location for new abstract location inference")
    separate_ovc: bool = Field(default=False, description="Whether to use separate grid modules for object vector cells (OVC)")

    # ===================================================================================
    # NETWORK INITIALIZATION
    # ===================================================================================
    # Standard deviation values and hidden layer sizes for weight initialization.
    # These control the initial random state of neural network parameters.

    g_init_std: float = Field(default=0.5, gt=0, description="Standard deviation for initial abstract location g")
    g_mem_std: float = Field(default=0.1, gt=0, description="Standard deviation for MLP hidden to output layer initialization")
    d_hidden_dim: int = Field(default=20, ge=1, description="Hidden layer size of MLP for abstract location transitions")

    # ===================================================================================
    # MEMORY SYSTEM (HEBBIAN LEARNING)
    # ===================================================================================
    # Parameters controlling associative memory dynamics using Hebbian plasticity rules.
    # Memory updates follow M_new = λ*M_old + η*outer(p, p), with retrieval via attractor dynamics.

    # Hebbian plasticity rates
    lambda_: float = Field(default=0.9999, ge=0, le=1, alias="lambda", description="Hebbian rate of forgetting")
    eta: float = Field(default=0.5, ge=0, le=1, description="Hebbian rate of remembering")
    kappa: float = Field(default=0.8, ge=0, le=1, description="Hebbian retrieval decay term")

    # Attractor dynamics configuration
    i_attractor: int = Field(default=5, ge=1, description="Number of iterations of attractor dynamics for memory retrieval")
    i_attractor_max_freq_inf: List[int] = Field(default_factory=list, description="Max attractor iterations per frequency in inference model (for early stopping)")
    i_attractor_max_freq_gen: List[int] = Field(default_factory=list, description="Max attractor iterations per frequency in generative model (for early stopping)")

    # Memory network configuration
    common_memory: bool = Field(default=False, description="Use common memory for generative and inference network")

    @computed_field(description="Number of attractor dynamics iterations (equals n_f_g)")
    @property
    def i_attractor_calculated(self) -> int:
        return self.n_f_g_calculated

    @computed_field(description="Max attractor iterations per frequency in inference model")
    @property
    def i_attractor_max_freq_inf_calculated(self) -> List[int]:
        if self.i_attractor_max_freq_inf:
            return self.i_attractor_max_freq_inf
        attractor = self.i_attractor_calculated
        return [attractor for _ in range(self.n_f_calculated)]

    @computed_field(description="Max attractor iterations per frequency in generative model")
    @property
    def i_attractor_max_freq_gen_calculated(self) -> List[int]:
        if self.i_attractor_max_freq_gen:
            return self.i_attractor_max_freq_gen
        attractor = self.i_attractor_calculated
        n_f_g = self.n_f_g_calculated
        n_f_ovc = self.n_f_ovc_calculated
        return [attractor - freq_nr for freq_nr in range(n_f_g)] + [attractor for _ in range(n_f_ovc)]

    # ===================================================================================
    # TRAINING CONFIGURATION
    # ===================================================================================
    # Parameters controlling the training process including batch sizes, walk generation,
    # learning rates, and curriculum scheduling (gradual enabling of loss terms).

    # Basic training hyperparameters
    train_it: int = Field(default=20000, ge=1, description="Number of walks to generate during training")
    n_rollout: int = Field(default=20, ge=1, description="Number of steps to roll out before backpropagation through time")
    batch_size: int = Field(default=16, ge=1, description="Number of walks for training simultaneously")

    # Walk length configuration
    walk_it_min: int = Field(default=25, ge=1, description="Minimum walk length (lower limit at end of training)")
    walk_it_max: int = Field(default=300, ge=1, description="Maximum walk length (upper limit at start of training)")
    walk_it_window: float = Field(default=55.0, ge=0, description="Width of window for sampling walk lengths")

    @computed_field(description="Calculated width of window for sampling walk lengths")
    @property
    def walk_it_window_calculated(self) -> float:
        return 0.2 * (self.walk_it_max - self.walk_it_min)

    # Learning rate schedule
    lr_max: float = Field(default=9.4e-4, gt=0, description="Maximum learning rate")
    lr_min: float = Field(default=8e-5, gt=0, description="Minimum learning rate")
    lr_decay_rate: float = Field(default=0.5, gt=0, le=1, description="Rate of learning rate decay")
    lr_decay_steps: int = Field(default=4000, ge=1, description="Steps of learning rate decay")

    # Loss weight configuration
    loss_weights_x: float = Field(default=1.0, ge=0, description="Weights of prediction losses")
    loss_weights_p: float = Field(default=1.0, ge=0, description="Weights of grounded location losses")
    loss_weights_g: float = Field(default=1.0, ge=0, description="Weights of abstract location losses")
    loss_weights_reg_g: float = Field(default=0.01, ge=0, description="Weights of regularisation losses for abstract location")
    loss_weights_reg_p: float = Field(default=0.02, ge=0, description="Weights of regularisation losses for grounded location")
    loss_weights: Tensor = Field(default_factory=lambda: zeros(8), description="Combined loss weights [L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p]")

    # Curriculum learning schedules (gradual loss term activation)
    loss_weights_p_g_it: int = Field(default=2000, ge=1, description="Backprop iters until latent losses (L_p_g, L_p_x, L_g) are fully weighted")
    loss_weights_reg_p_it: int = Field(default=4000, ge=1, description="Backprop iters until grounded location regularisation is fully weighted")
    loss_weights_reg_g_it: int = Field(default=40000000, ge=1, description="Backprop iters until abstract location regularisation is fully weighted")
    eta_it: int = Field(default=16000, ge=1, description="Backprop iters until eta (rate of remembering) is completely 'on'")
    lambda_it: int = Field(default=200, ge=1, description="Backprop iters until lambda (rate of forgetting) is completely 'on'")

    # Precision-weighted mean scheduling (p→g inference)
    p2g_scale_offset: float = Field(default=0.0, ge=0, description="Offset scaling for inferred grounded location standard deviation")
    p2g_sig_val: float = Field(default=10000.0, ge=0, description="Additional offset value to reduce influence in precision weighted mean")
    p2g_sig_half_it: int = Field(default=400, ge=1, description="Iterations where offset scaling should be 0.5")
    p2g_sig_scale_it: int = Field(default=200, ge=1, description="Rate of offset scaling decrease (down to ~0.25 after p2g_sig_half_it + this)")

    # ===================================================================================
    # CONNECTIVITY & STATIC MATRICES
    # ===================================================================================
    # Pre-computed matrices and masks defining network connectivity patterns.
    # These are typically calculated once and remain fixed during training.
    # Leave as empty lists/tensors to auto-calculate using composed parameters.

    # Hierarchical connectivity masks
    p_update_mask: Tensor = Field(default_factory=lambda: zeros(1, 1), description="Hebbian memory connections from low to high frequency (M_ij: FROM i TO j)")
    p_retrieve_mask_inf: List[Tensor] = Field(default_factory=list, description="Mask for hierarchical memory retrieval in inference model (early-stopping)")
    p_retrieve_mask_gen: List[Tensor] = Field(default_factory=list, description="Mask for hierarchical memory retrieval in generative model (early-stopping)")
    g_connections: List[List[bool]] = Field(default_factory=list, description="Hierarchical connections for abstract location module transitions (low to high)")

    @computed_field(description="Hebbian memory connections mask (low to high frequency)")
    @property
    def p_update_mask_calculated(self) -> Tensor:
        if self.p_update_mask.numel() > 1:
            return self.p_update_mask

        return utils.create_p_update_mask(
            n_p=self.n_p_calculated,
            n_f=self.n_f_calculated,
            n_f_g=self.n_f_g_calculated,
            n_f_ovc=self.n_f_ovc_calculated,
            f_initial=self.f_initial_extended,
        )

    @computed_field(description="Mask for hierarchical memory retrieval in inference model")
    @property
    def p_retrieve_mask_inf_calculated(self) -> List[Tensor]:
        if self.p_retrieve_mask_inf:
            return self.p_retrieve_mask_inf

        inf_masks, _ = utils.create_p_retrieve_masks(
            n_p=self.n_p_calculated,
            i_attractor=self.i_attractor_calculated,
            i_attractor_max_freq_inf=self.i_attractor_max_freq_inf_calculated,
            i_attractor_max_freq_gen=self.i_attractor_max_freq_gen_calculated,
        )
        return inf_masks

    @computed_field(description="Mask for hierarchical memory retrieval in generative model")
    @property
    def p_retrieve_mask_gen_calculated(self) -> List[Tensor]:
        if self.p_retrieve_mask_gen:
            return self.p_retrieve_mask_gen

        _, gen_masks = utils.create_p_retrieve_masks(
            n_p=self.n_p_calculated,
            i_attractor=self.i_attractor_calculated,
            i_attractor_max_freq_inf=self.i_attractor_max_freq_inf_calculated,
            i_attractor_max_freq_gen=self.i_attractor_max_freq_gen_calculated,
        )
        return gen_masks

    @computed_field(description="Hierarchical connections for abstract location module transitions")
    @property
    def g_connections_calculated(self) -> List[List[bool]]:
        if self.g_connections:
            return self.g_connections

        return utils.create_g_connections(
            n_f=self.n_f_calculated,
            n_f_g=self.n_f_g_calculated,
            n_f_ovc=self.n_f_ovc_calculated,
            f_initial=self.f_initial_extended,
        )

    # Outer product computation matrices
    W_repeat: List[Tensor] = Field(default_factory=list, description="Matrix for repeating abstract location g for outer product with x")
    W_tile: List[Tensor] = Field(default_factory=list, description="Matrix for tiling sensory observation x for outer product with g")

    @computed_field(description="Matrix for repeating abstract location g for outer product with x")
    @property
    def W_repeat_calculated(self) -> List[Tensor]:
        if self.W_repeat:
            return self.W_repeat

        n_g_sub = self.n_g_subsampled_combined
        n_x_f = self.n_x_f_calculated

        return utils.create_W_repeat(n_g_sub, n_x_f)

    @computed_field(description="Matrix for tiling sensory observation x for outer product with g")
    @property
    def W_tile_calculated(self) -> List[Tensor]:
        if self.W_tile:
            return self.W_tile

        n_g_sub = self.n_g_subsampled_combined
        n_x_f = self.n_x_f_calculated

        return utils.create_W_tile(n_g_sub, n_x_f)

    # Encoding/decoding matrices
    two_hot_table: List[Tensor] = Field(default_factory=list, description="Table for converting one-hot to two-hot compressed representation")
    g_downsample: List[Tensor] = Field(default_factory=list, description="Downsampling matrix from grid cells to compressed grid cells")

    @computed_field(description="Table for converting one-hot to two-hot compressed representation")
    @property
    def two_hot_table_calculated(self) -> List[Tensor]:
        if self.two_hot_table:
            return self.two_hot_table

        return utils.create_two_hot_table(n_x=self.n_x, n_x_c=self.n_x_c)

    @computed_field(description="Downsampling matrix from grid cells to compressed grid cells")
    @property
    def g_downsample_calculated(self) -> List[Tensor]:
        if self.g_downsample:
            return self.g_downsample

        n_g = self.n_g_calculated
        n_g_sub = self.n_g_subsampled_combined

        return utils.create_g_downsample(n_g, n_g_sub)
