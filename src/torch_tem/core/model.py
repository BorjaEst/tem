from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeAlias

# Standard modules
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator
from scipy.special import comb
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import settings, utils
from torch_tem.core.hpc import HPCModel, HPCState
from torch_tem.core.lec import LECModel, LECState
from torch_tem.core.mec import MECModel, MECState
from torch_tem.modules import MLP
from torch_tem.modules.autoencoder import AutoencoderModule
from torch_tem.modules.projection import ProjectionModule
from torch_tem.settings import TEMSettings
from torch_tem.types import Transition


class WorldParameters(BaseModel):
    """World + action-space parameters (input semantics)."""

    model_config = ConfigDict(extra="forbid")

    has_static_action: bool = Field(
        default=True,
        description="Does this world include the standing still action?",
    )
    n_actions: int = Field(
        default=4,
        description="Number of available actions, excluding the stand still action",
    )


class LECParameters(BaseModel):
    """LEC parameters: observation -> feature cell representation."""

    model_config = ConfigDict(extra="forbid")

    n_o: int = Field(default=45, description="Neurons for sensory observation o")
    n_c: int = Field(default=10, description="Neurons for compressed sensory experience c")


class MECParameters(BaseModel):
    """MEC parameters: abstract location (grid/ovc) structure and dynamics."""

    model_config = ConfigDict(extra="forbid")

    do_sample: bool = Field(
        default=False,
        description="Whether to sample, or assume no noise and simply take mean of all distributions",
    )
    separate_ovc: bool = Field(
        default=False,
        description="Whether to use separate grid modules that receive shiny information for object vector cells",
    )
    std_grid_init: float = Field(
        default=0.5,
        description="Standard deviation for initial g (which will then be learned)",
    )
    std_grid_mem: float = Field(
        default=0.1,
        description="Standard deviation to initialise hidden to output layer of MLP for inferring new abstract location",
    )
    d_hidden_dim: int = Field(
        default=20,
        description="Hidden layer size of MLP for abstract location transitions",
    )

    n_g_subsampled_base: list[int] = Field(
        default=[10, 10, 8, 6, 6],
        description="Base neurons for subsampled entorhinal abstract location f_g(g) for each frequency module",
    )
    n_ovc_base: list[int] | None = Field(
        default=None,
        description="Neurons for object vector cells",
    )

    f_initial_base: list[float] = Field(
        default=[0.99, 0.3, 0.09, 0.03, 0.01],
        description="Initial frequencies of each module",
    )


class HPCParameters(BaseModel):
    """HPC parameters: grounded location, inference toggles, and memory."""

    model_config = ConfigDict(extra="forbid")

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


class Parameters(BaseModel):
    """
    Pydantic model for Tolman-Eichenbaum Machine model parameters.

    This model defines model architecture and inference parameters only.
    Training schedules and data generation settings are in torch_tem.settings.
    """

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    world: WorldParameters = Field(
        default_factory=WorldParameters,
        description="World/action parameters.",
    )
    lec: LECParameters = Field(
        default_factory=LECParameters,
        description="LEC parameters (observation -> feature cells).",
    )
    mec: MECParameters = Field(
        default_factory=MECParameters,
        description="MEC parameters (grid/ovc abstract location).",
    )
    hpc: HPCParameters = Field(
        default_factory=HPCParameters,
        description="HPC parameters (memory + grounded location).",
    )

    # Module settings
    autoencoder: settings.AutoencoderSettings = Field(
        default_factory=settings.AutoencoderSettings,
        description="Autoencoder settings.",
    )
    lec_projection: settings.ProjectionSettings = Field(
        default_factory=settings.LECProjectionSettings,
        description="LEC projection settings.",
    )
    lec_settings: settings.LECSettings = Field(
        default_factory=settings.LECSettings,
        description="LEC module settings.",
    )
    mec_projection: settings.ProjectionSettings = Field(
        default_factory=settings.MECProjectionSettings,
        description="MEC projection settings.",
    )
    mec_settings: settings.MECSettings = Field(
        default_factory=settings.MECSettings,
        description="MEC module settings.",
    )
    hpc_settings: settings.HPCSettings = Field(
        default_factory=settings.HPCSettings,
        description="HPC module settings.",
    )

    @model_validator(mode="before")
    @classmethod
    def _upgrade_flat_to_nested(cls, data: Any) -> Any:
        """Accept both legacy flat keys and new nested groups."""
        if not isinstance(data, dict):
            return data

        data = dict(data)

        world = dict(data.get("world") or {})
        lec = dict(data.get("lec") or {})
        mec = dict(data.get("mec") or {})
        hpc = dict(data.get("hpc") or {})

        def pop_into(key: str, target: dict) -> None:
            if key in data:
                target.setdefault(key, data.pop(key))

        for key in ("has_static_action", "n_actions"):
            pop_into(key, world)

        for key in ("n_o", "n_c"):
            pop_into(key, lec)

        for key in (
            "do_sample",
            "separate_ovc",
            "std_grid_init",
            "std_grid_mem",
            "d_hidden_dim",
            "n_g_subsampled_base",
            "n_ovc_base",
            "f_initial_base",
        ):
            pop_into(key, mec)

        for key in ("use_x_cued_recall", "common_memory", "kappa"):
            pop_into(key, hpc)

        for key in ("p2g_sig_val",):
            pop_into(key, mec)

        if world:
            data["world"] = world
        if lec:
            data["lec"] = lec
        if mec:
            data["mec"] = mec
        if hpc:
            data["hpc"] = hpc

        return data

    # --- Backward-compatible flat attribute accessors (used by computed fields)
    @property
    def has_static_action(self) -> bool:
        return self.world.has_static_action

    @property
    def n_actions(self) -> int:
        return self.world.n_actions

    @property
    def n_o(self) -> int:
        return self.lec.n_o

    @property
    def n_c(self) -> int:
        return self.lec.n_c

    @property
    def do_sample(self) -> bool:
        return self.mec.do_sample

    @property
    def separate_ovc(self) -> bool:
        return self.mec.separate_ovc

    @property
    def std_grid_init(self) -> float:
        return self.mec.std_grid_init

    @property
    def std_grid_mem(self) -> float:
        return self.mec.std_grid_mem

    @property
    def d_hidden_dim(self) -> int:
        return self.mec.d_hidden_dim

    @property
    def n_g_subsampled_base(self) -> list[int]:
        return self.mec.n_g_subsampled_base

    @property
    def n_ovc_base(self) -> list[int] | None:
        return self.mec.n_ovc_base

    @property
    def f_initial_base(self) -> list[float]:
        return self.mec.f_initial_base

    @property
    def use_x_cued_recall(self) -> bool:
        return self.hpc.use_x_cued_recall

    @property
    def p2g_sig_val(self) -> float:
        return self.mec_settings.p2g.curriculum_sigma

    @property
    def common_memory(self) -> bool:
        return self.hpc.common_memory

    @property
    def kappa(self) -> float:
        return self.hpc.kappa

    @computed_field
    @property
    def n_ovc(self) -> list[int]:
        """Neurons for object vector cells."""
        if self.n_ovc_base is not None:
            return self.n_ovc_base
        return [0 for _ in range(len(self.n_g_subsampled_base))]

    @computed_field
    @property
    def n_g_subsampled(self) -> list[int]:
        """
        Add neurons for object vector cells. Add new modules if object vector cells get separate modules,
        or else add neurons to existing modules.
        """
        if self.separate_ovc:
            return self.n_g_subsampled_base + self.n_ovc
        else:
            return [grid + ovc for grid, ovc in zip(self.n_g_subsampled_base, self.n_ovc)]

    @computed_field
    @property
    def n_f_ovc(self) -> int:
        """Number of hierarchical frequency modules for object vector cells."""
        return len(self.n_ovc) if self.separate_ovc else 0

    @computed_field
    @property
    def n_f_g(self) -> int:
        """Number of hierarchical frequency modules for grid cells."""
        return len(self.n_g_subsampled) - self.n_f_ovc

    @computed_field
    @property
    def n_f(self) -> int:
        """Total number of modules."""
        return len(self.n_g_subsampled)

    @computed_field
    @property
    def n_g(self) -> list[int]:
        """Number of neurons of entorhinal abstract location g for each frequency."""
        return [3 * g for g in self.n_g_subsampled]

    @computed_field
    @property
    def n_x(self) -> list[int]:
        """Neurons for temporally filtered sensory experience x for each frequency."""
        return [self.n_c for _ in range(self.n_f)]

    @computed_field
    @property
    def n_p(self) -> list[int]:
        """Neurons for hippocampal grounded location p for each frequency."""
        return [g * x for g, x in zip(self.n_g_subsampled, self.n_x)]

    @computed_field
    @property
    def f_initial(self) -> list[float]:
        """Initial frequencies of each module, including object vector cell modules."""
        return self.f_initial_base + self.f_initial_base[0 : self.n_f_ovc]

    @computed_field
    @property
    def i_attractor(self) -> int:
        """Number of iterations of attractor dynamics for memory retrieval."""
        return self.n_f_g

    @computed_field
    @property
    def i_attractor_max_freq_inf(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in inference model."""
        return [self.i_attractor for _ in range(self.n_f)]

    @computed_field
    @property
    def i_attractor_max_freq_gen(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in generative model."""
        return [self.i_attractor - freq_nr for freq_nr in range(self.n_f_g)] + [self.i_attractor for _ in range(self.n_f_ovc)]

    # --- Connectivity matrices
    @computed_field
    @property
    def p_update_mask(self) -> torch.Tensor:
        """
        Set connections when forming Hebbian memory of grounded locations: from low frequency modules to high.
        High frequency modules come first.
        """
        mask = torch.zeros((np.sum(self.n_p), np.sum(self.n_p)), dtype=torch.float)
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # Entry M_ij (row i, col j) is the connection FROM cell i TO cell j
        for f_from in range(self.n_f):
            for f_to in range(self.n_f):
                # For connections that involve separate object vector modules
                if f_from >= self.n_f_g or f_to >= self.n_f_g:
                    # Connection between object vector modules: only allow from low to high frequency
                    if f_from >= self.n_f_g and f_to >= self.n_f_g:
                        if self.f_initial[f_from] <= self.f_initial[f_to]:
                            mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
                    # Connection between object vector and normal modules: allow any connections
                    else:
                        mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
                # Connection between abstract location frequency modules: only from low to high frequency
                else:
                    if self.f_initial[f_from] <= self.f_initial[f_to]:
                        mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0

        return mask

    @computed_field
    @property
    def p_retrieve_mask_inf(self) -> list[torch.Tensor]:
        """Hierarchical memory retrieval masks for inference model."""
        masks = [torch.zeros(sum(self.n_p)) for _ in range(self.i_attractor)]
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # For each frequency, insert ones in the mask for those iterations
        for f, max_i in enumerate(self.i_attractor_max_freq_inf):
            for i in range(max_i):
                masks[i][n_p[f] : n_p[f + 1]] = 1.0

        return masks

    @computed_field
    @property
    def p_retrieve_mask_gen(self) -> list[torch.Tensor]:
        """Hierarchical memory retrieval masks for generative model."""
        masks = [torch.zeros(sum(self.n_p)) for _ in range(self.i_attractor)]
        n_p = np.cumsum(np.concatenate(([0], self.n_p)))

        # For each frequency, insert ones in the mask for those iterations
        for f, max_i in enumerate(self.i_attractor_max_freq_gen):
            for i in range(max_i):
                masks[i][n_p[f] : n_p[f + 1]] = 1.0

        return masks

    @computed_field
    @property
    def g_connections(self) -> list[list[bool]]:
        """
        In path integration, abstract location frequency modules can influence the transition of other modules
        hierarchically (low to high).
        """
        # Connections for grid cell modules
        connections = [[self.f_initial[f_from] <= self.f_initial[f_to] for f_from in range(self.n_f_g)] + [False for _ in range(self.n_f_ovc)] for f_to in range(self.n_f_g)]

        # Add connections for separate object vector cell modules
        connections += [
            [False for _ in range(self.n_f_g)] + [self.f_initial[f_from] <= self.f_initial[f_to] for f_from in range(self.n_f_g, self.n_f)] for f_to in range(self.n_f_g, self.n_f)
        ]

        return connections

    # ---- Static matrices
    @computed_field
    @property
    def W_repeat(self) -> list[torch.Tensor]:
        """Matrix for repeating abstract location g to do outer product with sensory information x."""
        return [torch.tensor(np.kron(np.eye(self.n_g_subsampled[f]), np.ones((1, self.n_x[f]))), dtype=torch.float) for f in range(self.n_f)]

    @computed_field
    @property
    def W_tile(self) -> list[torch.Tensor]:
        """Matrix for tiling sensory observation x to do outer product with abstract location g."""
        return [torch.tensor(np.kron(np.ones((1, self.n_g_subsampled[f])), np.eye(self.n_x[f])), dtype=torch.float) for f in range(self.n_f)]

    @computed_field
    @property
    def two_hot_table(self) -> list[torch.Tensor]:
        """Table for converting one-hot to two-hot compressed representation."""
        table = [[0] * (self.n_c - 2) + [1] * 2]

        # Generate compressed codes for each possible observation
        for i in range(1, min(int(comb(self.n_c, 2)), self.n_o)):
            code = table[-1].copy()
            # Find latest occurrence of [0 1] in that code
            swap = [index for index in range(len(code) - 1, -1, -1) if code[index : index + 2] == [0, 1]][0]
            # Swap those to get new code
            code[swap : swap + 2] = [1, 0]
            # If the first one was swapped: value after swapped pair is 1
            if swap + 2 < len(code) and code[swap + 2] == 1:
                # Move the second 1 all the way back - reverse everything after the swapped pair
                code[swap + 2 :] = code[: swap + 1 : -1]
            table.append(code)

        # Convert each code to column vector pytorch tensor
        return [torch.tensor(code) for code in table]

    @computed_field
    @property
    def g_downsample(self) -> list[torch.Tensor]:
        """Downsampling matrix to go from grid cells to compressed grid cells."""
        return [
            torch.cat([torch.eye(dim_out, dtype=torch.float), torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)])
            for dim_in, dim_out in zip(self.n_g, self.n_g_subsampled)
        ]

    def to_legacy_dict(self) -> dict[str, Any]:
        """Flatten nested Parameters to the legacy dict consumed by TEMModel."""
        base: dict[str, Any] = {
            # World
            "has_static_action": self.has_static_action,
            "n_actions": self.n_actions,
            # LEC
            "n_o": self.n_o,
            "n_c": self.n_c,
            # MEC
            "do_sample": self.do_sample,
            "separate_ovc": self.separate_ovc,
            "std_grid_init": self.std_grid_init,
            "std_grid_mem": self.std_grid_mem,
            "d_hidden_dim": self.d_hidden_dim,
            "n_g_subsampled_base": self.n_g_subsampled_base,
            "n_ovc_base": self.n_ovc_base,
            "f_initial_base": self.f_initial_base,
            # HPC
            "use_x_cued_recall": self.use_x_cued_recall,
            "p2g_sig_val": self.p2g_sig_val,
            "common_memory": self.common_memory,
            "kappa": self.kappa,
        }

        derived: dict[str, Any] = {
            # sizes / derived scalars
            "n_ovc": self.n_ovc,
            "n_g_subsampled": self.n_g_subsampled,
            "n_f_ovc": self.n_f_ovc,
            "n_f_g": self.n_f_g,
            "n_f": self.n_f,
            "n_g": self.n_g,
            "n_x": self.n_x,
            "n_p": self.n_p,
            "f_initial": self.f_initial,
            "i_attractor": self.i_attractor,
            "i_attractor_max_freq_inf": self.i_attractor_max_freq_inf,
            "i_attractor_max_freq_gen": self.i_attractor_max_freq_gen,
            # masks / matrices
            "p_update_mask": self.p_update_mask,
            "p_retrieve_mask_inf": self.p_retrieve_mask_inf,
            "p_retrieve_mask_gen": self.p_retrieve_mask_gen,
            "g_connections": self.g_connections,
            "W_repeat": self.W_repeat,
            "W_tile": self.W_tile,
            "two_hot_table": self.two_hot_table,
            "g_downsample": self.g_downsample,
        }

        return {**base, **derived}


@dataclass
class RuntimeHyperparameters:
    """Runtime hyperparameters injected by training (not part of model architecture).

    These values are computed by the training schedule and updated each step.
    They control time-varying aspects of model behavior during training.
    """

    eta: float = 0.0  # Hebbian learning rate (rate of remembering)
    hebbian_decay: float = 0.9999  # Hebbian decay factor (rate of forgetting)
    p2g_scale_offset: float = 1.0  # Variance offset scaling for p->g inference


# This specifies how parameters are updated at every backpropagation iteration/gradient update
def parameter_iteration(iteration, params):
    # Calculate eta (rate of remembering) and hebian decay (rate of forgetting) for Hebbian memory updates
    hebbian_decay = params.get("hebbian_decay")
    eta = min((iteration + 1) / params["eta_it"], 1) * params["eta"]
    lamb = min((iteration + 1) / params["lambda_it"], 1) * hebbian_decay
    # Calculate current scaling of variance offset for ground location inference
    p2g_scale_offset = 1 / (1 + np.exp((iteration - params["p2g_sig_half_it"]) / params["p2g_sig_scale_it"]))
    # Calculate current learning rate
    lr = max(params["lr_min"] + (params["lr_max"] - params["lr_min"]) * (params["lr_decay_rate"] ** (iteration / params["lr_decay_steps"])), params["lr_min"])
    # Calculate center of walk length window, within which the walk lenghts of new walks are uniformly sampled
    # Use max_steps (Lightning's Trainer.max_steps) as the training horizon
    max_steps = max(int(params.get("max_steps", 1)), 1)
    walk_length_center = (
        params["walk_it_max"] - params["walk_it_window"] * 0.5 - min((iteration + 1) / max_steps, 1) * (params["walk_it_max"] - params["walk_it_min"] - params["walk_it_window"])
    )
    # Calculate current loss weights
    L_p_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_p"]
    L_p_x = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_p"] * (1 - p2g_scale_offset)
    L_x_gen = params["loss_weights_x"]
    L_x_g = params["loss_weights_x"]
    L_x_p = params["loss_weights_x"]
    L_g = min((iteration + 1) / params["loss_weights_p_g_it"], 1) * params["loss_weights_g"]
    L_reg_g = (1 - min((iteration + 1) / params["loss_weights_reg_g_it"], 1)) * params["loss_weights_reg_g"]
    L_reg_p = (1 - min((iteration + 1) / params["loss_weights_reg_p_it"], 1)) * params["loss_weights_reg_p"]
    # And concatenate them in the order expected by the model
    loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p])
    # Return all updated parameters
    return eta, lamb, p2g_scale_offset, lr, walk_length_center, loss_weights


@dataclass
class TEMLabel:
    o: Tensor  # True sensory observation (for loss computation)
    locations: List[Tensor]  # True locations (for loss computation)


@dataclass
class TEMState:
    lec_state: LECState
    mec_state: MECState
    hpc_state: HPCState

    def detach(self) -> "TEMState":
        return TEMState(
            lec_state=self.lec_state.detach(),
            mec_state=self.mec_state.detach(),
            hpc_state=self.hpc_state.detach(),
        )


@dataclass
class TEMInference:
    g_inf: List[Tensor]
    p_inf: List[Tensor]
    p_xi: List[Tensor]


@dataclass
class TEMGenerative:
    g_gen: List[Tensor]
    p_gen_gg: List[Tensor]
    p_gen_gi: List[Tensor]


@dataclass
class TEMReconstruction:
    o_hat: Sequence[Tensor]
    o_logits: Sequence[Tensor]


@dataclass
class TEMOutput:
    inference: TEMInference
    generative: TEMGenerative
    reconstruction: TEMReconstruction


class TEMModel(nn.Module):
    def __init__(self, params: Parameters):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(TEMModel, self).__init__()

        # Accept either Parameters object or legacy dict
        self._params = params  # TODO: replace by settings
        self.hyper = params.to_legacy_dict()

        # Initialize runtime hyperparameters with safe defaults
        # These will be updated by training before each forward pass
        self.runtime = RuntimeHyperparameters()

        # Extract commonly used parameters
        n_a = self.hyper["n_actions"]
        n_o = self.hyper["n_o"]
        n_c = self.hyper["n_c"]
        n_p = self.hyper["n_p"]
        n_g = self.hyper["n_g"]
        n_x = self.hyper["n_x"]
        f_init = self.hyper["f_initial"]

        # Initialize LEC (Lateral Entorhinal Cortex) component
        self.autoencoder = AutoencoderModule(n_o, n_c, params.autoencoder)
        self.lec = lec = LECModel(n_c, f_init, params.lec_settings)
        self.mec = mec = MECModel(n_a, n_p, n_g, f_init, params.mec_settings)
        self.hpc = hpc = HPCModel(params.i_attractor, n_p, f_init, params.hpc_settings)  # i_attactor must be equal to n of frequencies for grid cells
        self.lec_projection = ProjectionModule(lec, hpc, params.lec_projection)
        self.mec_projection = ProjectionModule(mec, hpc, params.mec_projection)

    def set_runtime_hyperparams(self, eta: float, hebbian_decay: float, p2g_scale_offset: float) -> None:
        """Set runtime hyperparameters (called by training loop each step).

        Args:
            eta: Hebbian learning rate (rate of remembering)
            hebbian_decay: Hebbian decay factor (rate of forgetting)
            p2g_scale_offset: Variance offset scaling for p->g inference
        """
        self.runtime.eta = eta
        self.runtime.hebbian_decay = hebbian_decay
        self.runtime.p2g_scale_offset = p2g_scale_offset
        self.mec.set_runtime(p2g_scale_offset=p2g_scale_offset)
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

    @property
    def settings(self) -> TEMSettings:
        """Return TEM settings object constructed from model parameters."""
        return self._params

    def _apply(self, fn):
        """Override _apply to move tensors in self.hyper when model is moved to GPU/CPU."""
        super()._apply(fn)
        self.hyper = self._apply_to_nested_tensors(self.hyper, fn)
        return self

    def _apply_to_nested_tensors(self, obj, fn):
        """Recursively apply function to all tensors in nested dict/list/tuple structure."""
        if torch.is_tensor(obj):
            return fn(obj)
        if isinstance(obj, dict):
            return {k: self._apply_to_nested_tensors(v, fn) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._apply_to_nested_tensors(v, fn) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._apply_to_nested_tensors(v, fn) for v in obj)
        return obj

    def forward(self, o, locations, a_prev, state: TEMState) -> tuple[TEMOutput, TEMState]:
        mec_state, lec_state, hpc_state = state.mec_state, state.lec_state, state.hpc_state
        c = self.autoencoder.encode(o)
        device = o.device
        memory = hpc_state.memory

        # Handle reset boundaries: where a_prev is None, reset state to priors before transition
        reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool, device=device)
        if torch.any(reset_mask):
            # Reset g to priors for envs with no previous action
            g_reset = [torch.where(reset_mask.unsqueeze(-1), self.mec.cells_init[f].unsqueeze(0), mec_state.cells[f]) for f in range(self.hyper["n_f"])]
            mec_state.cells = g_reset

        # Convert actions to one-hot format expected by MEC (use 0 for None, will be reset above)
        if self.hyper["has_static_action"]:
            a = utils.one_hot_with_zero(a_prev, self.hyper["n_actions"], device=device)
        else:
            a_idx = torch.tensor([int(a) if a is not None else 0 for a in a_prev], dtype=torch.long, device=device)
            a = torch.nn.functional.one_hot(a_idx, num_classes=self.hyper["n_actions"]).float()

        # Observe / infer: LEC filtering + HPC retrieval + MEC correction
        x_inf, lec_state = self.lec.inference(c, lec_state)
        x_ = self.lec_projection(x_inf)  # Project to memory format
        p_xi = self.hpc.attractor(x_, memory[1], retrieve_it_mask=self.hyper["p_retrieve_mask_inf"]) if self.hyper["use_x_cued_recall"] else None

        # Transition: MEC path integration (action-driven)
        g_gen, mec_state = self.mec.generative(a, locations, mec_state)  # Updates mec state with g_path
        g_ = self.mec_projection(g_gen)
        p_gg = self.hpc.attractor(g_, memory[0], retrieve_it_mask=self.hyper["p_retrieve_mask_gen"])

        # Infer abstract location by using state and sensory experience
        g_inf, mec_state = self.mec.inference(p_xi, locations=locations, state=mec_state)
        g_ = self.mec_projection(g_inf)
        p_gi = self.hpc.attractor(g_, memory[0], retrieve_it_mask=self.hyper["p_retrieve_mask_gen"])

        # Generate grounded location from inferred abstract location
        p_gen_gi, hpc_state = self.hpc.generative(p_gi, hpc_state)
        p_gen_gg, hpc_state = self.hpc.generative(p_gg, hpc_state)

        # Infer grounded location from abstract location and sensory experience
        p_inf, hpc_state = self.hpc.inference(x_, g_, hpc_state)

        # Update memory (Hebbian write)  (Idealy should be in hpc.generative and hpc.inference)
        M = self.update_memory(hpc_state.memory, p_inf, p_xi, p_gen_gi)
        hpc_state.memory = M
        state.hpc_state = hpc_state

        # Build tem state
        state = TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)

        # Build tem generative and inference interfaces
        generative = TEMGenerative(g_gen=g_gen, p_gen_gg=p_gen_gg, p_gen_gi=p_gen_gi)
        inference = TEMInference(g_inf=g_inf, p_inf=p_inf, p_xi=p_xi)

        # Generate observation prediction from inferred grounded location
        x = self.lec_projection.inverse(p_inf)
        c_p_inf = self.lec.generative(x)
        o_p_inf_logits = self.autoencoder.decode(c_p_inf)
        o_p_inf = utils.softmax(o_p_inf_logits)

        # Generate observation from inferred grounded location
        x = self.lec_projection.inverse(p_gen_gi)
        c_p_gen_gi = self.lec.generative(x)
        o_gen_gi_logits = self.autoencoder.decode(c_p_gen_gi)
        o_gen_gi = utils.softmax(o_gen_gi_logits)

        # Generate observation from generated grounded location
        x = self.lec_projection.inverse(p_gen_gg)
        c_p_gen_gg = self.lec.generative(x)
        x_gen_gg_logits = self.autoencoder.decode(c_p_gen_gg)
        o_gen_gg = utils.softmax(x_gen_gg_logits)

        # Return all generated observations and their corresponding logits
        o_hat, o_logits = (o_p_inf, o_gen_gi, o_gen_gg), (o_p_inf_logits, o_gen_gi_logits, x_gen_gg_logits)
        reconstructions = TEMReconstruction(o_hat=o_hat, o_logits=o_logits)

        # Build full output, state and return
        output = TEMOutput(inference=inference, generative=generative, reconstruction=reconstructions)
        state = TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)
        return output, state

    def update_memory(self, memory_prev: List[Tensor], p_inf, p_xi, p_gen_gi) -> List[Tensor]:
        """Update Hebbian memory matrices.

        Computes M_{t+1} after predictions are made (to avoid write-then-read shortcut).

        Args:
            memory_prev: Previous memory matrices (from state.hpc_state.memory)
            p_inf: Inferred place cells
            p_xi: Place cells from sensory retrieval
            p_gen_gi: Generated place cells (from g_inf via memory)

        Returns:
            Updated memory matrices [M_gen, M_inf] (M_inf only if use_x_cued_recall=True)
        """
        # Update generative memory with generated and inferred grounded location
        M = [self.hpc.hebbian_updater(memory_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen_gi, dim=1))]
        # If using memory for grounded location inference: append inference memory
        if self.hyper["use_x_cued_recall"]:
            # Inference memory is identical to generative memory if using common memory, and updated separately if not
            M.append(
                M[0]
                if self.hyper["common_memory"]
                else self.hpc.hebbian_updater(memory_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_xi, dim=1), do_hierarchical_connections=False)
            )
        return M

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> TEMState:
        # Create initial LEC state (x starts as x_filtered since no scaling/normalization yet)
        lec_state = self.lec.init_state(batch_size, device)
        # Initialise previous abstract location by stacking abstract location prior
        mec_state = self.mec.init_state(batch_size, device)
        # Create initial HPC state with initialized memory
        hpc_state = self.hpc.init_state(batch_size, device)
        # And construct new iteration for that g, o, a, and M
        return TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)


Walk = Iterable[Tuple[Any, Tensor, Any]]  # (locations, o, a)


class Rollout(Iterator[TEMState]):
    def __init__(self, model: TEMModel, walk: Walk, initial: Optional[TEMState] = None):
        self.model = model
        self.walk = list(walk)  # Materialize for predictable indexing

        if len(self.walk) == 0:
            raise ValueError("Rollout requires at least 1 timestep in walk")

        # Extract first step to determine batch size and initialize state
        locations_0, o_0, _ = self.walk[0]

        # Determine initial state
        state = initial or model.init_state(batch_size=o_0.shape[0], device=o_0.device)

        # Initialize prev-values for first forward pass
        self._a_prev = [None for _ in range(o_0.shape[0])]
        self._state = state

        # Track current position in walk
        self._idx = 0

    def __iter__(self) -> "Rollout":
        """Return self as iterator."""
        return self

    def __next__(self) -> TEMState:
        """Process next timestep and return state.

        Returns:
            TEMState for current timestep.

        Raises:
            StopIteration: When all timesteps have been processed.
        """
        if self._idx >= len(self.walk):
            raise StopIteration

        # Get current timestep
        locations, o, a = self.walk[self._idx]
        self._idx += 1

        # Run model forward
        output, state = self.model(o, locations, self._a_prev, self._state)

        # Build state with updated components
        state.g = locations
        state.a_prev = a

        # Update prev-values for next iteration
        self._a_prev = a
        self._state = state

        # Build labels for current timestep
        labels = TEMLabel(o=o, locations=locations)

        return output, labels, state
