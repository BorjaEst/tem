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
    g_init_std: float = Field(
        default=0.5,
        description="Standard deviation for initial g (which will then be learned)",
    )
    g_mem_std: float = Field(
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
    p2g_sig_val: float = Field(
        default=10000.0,
        description="Additional value to offset standard deviation of inferred grounded location",
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
            "g_init_std",
            "g_mem_std",
            "d_hidden_dim",
            "n_g_subsampled_base",
            "n_ovc_base",
            "f_initial_base",
        ):
            pop_into(key, mec)

        for key in ("use_x_cued_recall", "p2g_sig_val", "common_memory", "kappa"):
            pop_into(key, hpc)

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
    def g_init_std(self) -> float:
        return self.mec.g_init_std

    @property
    def g_mem_std(self) -> float:
        return self.mec.g_mem_std

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
        return self.hpc.p2g_sig_val

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
            "g_init_std": self.g_init_std,
            "g_mem_std": self.g_mem_std,
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
class TEMPrediction:
    """Prediction outputs from TEM state.

    Contains all predictive outputs used for loss computation:
    - o_hat: sensory predictions from the 3 pathways (path integration, generative, and grounded location cued)
    - o_logits: logits for loss computation from the 3 pathways
    """

    o_hat: Sequence[Tensor]  # (x_p, x_g, x_gt) - sensory predictions from 3 pathways
    o_logits: Sequence[Tensor]  # (logits_p, logits_g, logits_gt) - logits for loss


@dataclass
class TEMState:
    """State container for TEM dynamics.

    Represents the complete latent state at timestep t:
    - lec_state: LEC temporal filtering state
    - mec_state: MEC grid cell state (contains g, g_gen, g_path)
    - hpc_state: HPC place cell + memory state (HPCState.memory holds Hebbian matrices)
    - g_inf: Inferred abstract location (corrected grid cells)
    - p_inf: Inferred grounded location (corrected place cells)
    - p_inf_x: Place cells from sensory retrieval (for loss computation)
    """

    lec_state: Optional[LECState] = None
    mec_state: Optional[MECState] = None
    hpc_state: Optional[HPCState] = None

    g_inf: Optional[List[Tensor]] = None
    p_inf: Optional[List[Tensor]] = None
    p_inf_x: Optional[List[Tensor]] = None  # Grounded location from sensory input (for loss computation)

    # Legacy fields for backward compatibility (mirrors hpc_state.memory when present)
    g: Any = None
    o: Optional[Tensor] = None
    a_prev: Any = None
    p_gen: Optional[List[Tensor]] = None
    x_gen: Optional[Sequence[Tensor]] = None
    o_logits: Optional[Sequence[Tensor]] = None

    @property
    def g_gen(self) -> Optional[List[Tensor]]:
        """Return per-frequency generated abstract location g for the current timestep."""
        if self.mec_state is None:
            return None
        return self.mec_state.g_gen

    def correct(self) -> List[np.ndarray]:
        """Return per-prediction correctness arrays for the current timestep."""
        if self.o is None or self.x_gen is None:
            return []

        observation = self.o.detach().cpu().numpy()
        predictions = [tensor.detach().cpu().numpy() for tensor in self.x_gen]
        return [np.argmax(pred, axis=-1) == np.argmax(observation, axis=-1) for pred in predictions]

    def detach(self) -> "TEMState":
        """Return a detached copy suitable for storing as `prev_iter`."""

        def _detach(obj: Any) -> Any:
            if obj is None:
                return None
            if isinstance(obj, LECState):
                # Detach LECState components
                return LECState(x=_detach(obj.x), x_filtered=_detach(obj.x_filtered))
            if isinstance(obj, MECState):
                # Detach MECState components
                return MECState(
                    g_gen=_detach(obj.g_gen),
                    g_path=Transition(mean=_detach(obj.g_path.mean), uncertainty=_detach(obj.g_path.uncertainty)),
                    g=_detach(obj.g),
                )
            if isinstance(obj, HPCState):
                # Detach HPCState components (including memory matrices)
                return HPCState(p=_detach(obj.p), memory=_detach(obj.memory))
            if torch.is_tensor(obj):
                return obj.detach()
            if isinstance(obj, list):
                return [_detach(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_detach(v) for v in obj)
            if isinstance(obj, dict):
                return {k: _detach(v) for k, v in obj.items()}
            return obj

        return TEMState(
            g=self.g,
            o=_detach(self.o),
            a_prev=self.a_prev,
            lec_state=_detach(self.lec_state),
            mec_state=_detach(self.mec_state),
            hpc_state=_detach(self.hpc_state),
            p_gen=_detach(self.p_gen),
            x_gen=_detach(self.x_gen),
            o_logits=_detach(self.o_logits),
            g_inf=_detach(self.g_inf),
            p_inf=_detach(self.p_inf),
            p_inf_x=_detach(self.p_inf_x),
        )

    @property
    def p_gen_gi(self) -> Optional[List[Tensor]]:
        """Return per-frequency generated grounded location p for the current timestep."""
        return self.p_gen

    @property
    def p_xi(self) -> Optional[List[Tensor]]:
        """Return per-frequency inferred grounded location p from sensory input for the current timestep."""
        return self.p_inf_x


@dataclass
class TEMLabel:
    o: Tensor  # True sensory observation (for loss computation)
    locations: List[Tensor]  # True locations (for loss computation)


class TEMModel(nn.Module):
    def __init__(self, params: Parameters):
        # First call super class init function to set up torch.nn.Module style model and inherit it's functionality
        super(TEMModel, self).__init__()

        # Accept either Parameters object or legacy dict
        self._params = params
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
        self.lec = lec = LECModel(n_c, n_x, f_init, params.lec_settings)
        self.mec = mec = MECModel(n_a, n_g, f_init, params.mec_settings)
        self.hpc = hpc = HPCModel(params.i_attractor, n_p, f_init, params.hpc_settings)  # i_attactor must be equal to n of frequencies for grid cells
        self.lec_projection = ProjectionModule(lec, hpc, params.lec_projection)
        self.mec_projection = ProjectionModule(mec, hpc, params.mec_projection)

        # Create trainable parameters
        self.init_trainable()

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
        self.hpc.set_runtime(eta=eta, hebbian_decay=hebbian_decay)

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

    def _init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
        """Create initial Hebbian memory matrices in the legacy [M_gen, M_inf?] format."""
        m0 = torch.zeros(
            (batch_size, sum(self.hyper["n_p"]), sum(self.hyper["n_p"])),
            dtype=torch.float,
            device=device,
        )
        memory = [m0]
        if self.hyper["use_x_cued_recall"]:
            memory.append(m0 if self.hyper["common_memory"] else m0.clone())
        return memory

    def forward(self, o, locations, a_prev, state: TEMState):
        """Legacy forward pass for backward compatibility.

        Standard Markov chain flow:
        1. transition(state, action) -> state_prior
        2. observe(state_prior, observation) -> state_posterior
        3. predict(state_posterior) -> predictions
        4. update_memory(state_posterior) -> M_next
        """
        mec_state, lec_state, hpc_state = state.mec_state, state.lec_state, state.hpc_state

        # 1. Transition: MEC path integration (action-driven)
        mec_state = self.transition(mec_state, a_prev, locations, o.device)

        # 2. Observe / infer: LEC filtering + HPC retrieval + MEC correction
        lec_state, g_inf, p_inf_x, p_inf, hpc_state = self.inference(
            o,
            locations,
            TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state),
        )

        # Update mec_state.g to inferred g for next transition (legacy parity)
        mec_state.g = g_inf

        # 3. Predict: Generate predictions from corrected state
        predictions, p_gen = self.predict(
            p_inf,
            g_inf,
            mec_state.g_gen,
            TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state),
        )

        # 4. Update memory (Hebbian write)
        M = self.update_memory(hpc_state.memory, p_inf, p_inf_x, p_gen)
        hpc_state.memory = M
        state.hpc_state = hpc_state

        # Return all iteration values (loss now computed in Lightning module)
        state = TEMState(lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state, g_inf=g_inf, p_inf=p_inf, p_inf_x=p_inf_x, p_gen=p_gen)

        return predictions, state

    def transition(self, mec_state: MECState, a_prev, locations, device) -> MECState:
        """Transition: MEC path integration (action-driven, no observation).

        Computes S_{t|t-1} = f(S_{t-1|t-1}, a_{t-1})

        Args:
            mec_state: Previous MEC state (contains g posterior from t-1)
            a_prev: Actions from previous timestep
            locations: Location metadata (for shiny detection)
            device: Torch device

        Returns:
            Updated MEC state with g_path (prior), g_gen (ancestral prediction)
        """
        # Handle reset boundaries: where a_prev is None, reset state to priors before transition
        reset_mask = torch.tensor([a is None for a in a_prev], dtype=torch.bool, device=device)
        if torch.any(reset_mask):
            # Reset g to priors for envs with no previous action
            g_reset = [torch.where(reset_mask.unsqueeze(-1), self.mec.grid.g_init_mean[f].unsqueeze(0), mec_state.g[f]) for f in range(self.hyper["n_f"])]
            mec_state.g = g_reset

        # Convert actions to one-hot format expected by MEC (use 0 for None, will be reset above)
        if self.hyper["has_static_action"]:
            a = utils.one_hot_with_zero(a_prev, self.hyper["n_actions"], device=device)
        else:
            a_idx = torch.tensor([int(a) if a is not None else 0 for a in a_prev], dtype=torch.long, device=device)
            a = torch.nn.functional.one_hot(a_idx, num_classes=self.hyper["n_actions"]).float()

        g_gen, mec_state = self.mec.generative(a, locations, mec_state)
        return mec_state

    def predict(self, p_inf, g_inf, g_gen, state: TEMState) -> TEMPrediction:
        r"""Predict: Generate predictions from corrected state.

        Computes \hat{o}_t = g(S_{t|t})

        Generates sensory predictions from three pathways:
        1. From inferred place cells (p_inf -> o_hat)
        2. From inferred grid cells via memory (g_inf -> p -> o_hat)
        3. From generated grid cells via memory (g_gen -> p -> o_hat)

        Args:
            p_inf: Inferred place cells
            g_inf: Inferred grid cells (corrected)
            g_gen: Generated grid cells (from transition)
            state: TEMState containing hpc_state with memory

        Returns:
            TEMPrediction with o_hat, o_logits, p_gen
        """
        o_hat, o_logits, p_gen = self.generative(p_inf, g_inf, g_gen, state)
        return TEMPrediction(o_hat=o_hat, o_logits=o_logits), p_gen

    def update_memory(self, memory_prev: List[Tensor], p_inf, p_inf_x, p_gen) -> List[Tensor]:
        """Update Hebbian memory matrices.

        Computes M_{t+1} after predictions are made (to avoid write-then-read shortcut).

        Args:
            memory_prev: Previous memory matrices (from state.hpc_state.memory)
            p_inf: Inferred place cells
            p_inf_x: Place cells from sensory retrieval
            p_gen: Generated place cells (from g_inf via memory)

        Returns:
            Updated memory matrices [M_gen, M_inf] (M_inf only if use_x_cued_recall=True)
        """
        # Update generative memory with generated and inferred grounded location
        M = [self.hpc.hebbian(memory_prev[0], torch.cat(p_inf, dim=1), torch.cat(p_gen, dim=1))]
        # If using memory for grounded location inference: append inference memory
        if self.hyper["use_x_cued_recall"]:
            # Inference memory is identical to generative memory if using common memory, and updated separately if not
            M.append(
                M[0] if self.hyper["common_memory"] else self.hpc.hebbian(memory_prev[1], torch.cat(p_inf, dim=1), torch.cat(p_inf_x, dim=1), do_hierarchical_connections=False)
            )
        return M

    def inference(self, o, locations, state: TEMState):
        lec_state, mec_state, hpc_state = state.lec_state, state.mec_state, state.hpc_state
        memory = hpc_state.memory
        # Delegate sensory processing to modular components:
        # 1. Autoencoder: o -> c (compression)
        # 2. LEC: c, x_prev -> x (temporal filtering)
        # 3. Projection: x -> x_ (normalization + tiling for memory)
        c = self.autoencoder.encode(o)
        x, lec_state = self.lec(c, lec_state)
        x_ = self.lec_projection(x)  # Project to memory format
        # Retrieve grounded location from memory by doing pattern completion on current sensory experience
        p_x = self.hpc.attractor(x_, memory[1], retrieve_it_mask=self.hyper["p_retrieve_mask_inf"]) if self.hyper["use_x_cued_recall"] else None
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        g = self.mec_inference(p_x, o, locations, mec_state)
        # Prepare abstract location for input to memory by downsampling and weighting
        g_ = self.mec_projection(g)
        # Infer grounded location from sensory experience and inferred abstract location
        p, hpc_state = self.hpc.inference(x_, g_, hpc_state)
        # Return LECState (for next step) and inferred variables
        return lec_state, g, p_x, p, hpc_state

    def generative(self, p_inf, g_inf, g_gen, state: TEMState):
        memory = state.hpc_state.memory
        # Generate observation from inferred grounded location, using only the highest frequency. Also keep non-softmaxed logits which are used in the loss later
        x_p, x_p_logits = self.gen_o(p_inf[0])
        # Retrieve grounded location from memory by pattern completion on inferred abstract location
        g_ = self.mec_projection(g_inf)
        p_g = self.hpc.attractor(g_, memory[0], retrieve_it_mask=self.hyper["p_retrieve_mask_gen"])
        p_g_inf, _ = self.hpc.generative(p_g, state.hpc_state)  # was p_mem_inf
        # And generate observation from the grounded location retrieved from inferred abstract location
        x_g, x_g_logits = self.gen_o(p_g_inf[0])
        # Retreive grounded location from memory by pattern completion on abstract location by transitioning
        g_ = self.mec_projection(g_gen)
        p_g = self.hpc.attractor(g_, memory[0], retrieve_it_mask=self.hyper["p_retrieve_mask_gen"])
        p_g_gen, _ = self.hpc.generative(p_g, state.hpc_state)  # was p_mem_gen
        # Generate observation from sampled grounded location
        x_gt, x_gt_logits = self.gen_o(p_g_gen[0])
        # Return all generated observations and their corresponding logits
        return (x_p, x_g, x_gt), (x_p_logits, x_g_logits, x_gt_logits), p_g_inf

    def init_trainable(self):
        # Initialize LEC (Lateral Entorhinal Cortex) component parameters with proper initial values
        # Use LEC's init_alpha method to set temporal filtering factors
        # self.autoencoder.init_trainable() already inits in Autoencoder.__init__
        # self.lec.init_trainable(self.hyper["f_initial"]) already inits in LECModel __init__
        # self.lec_projection.init_trainable() already inits in ProjectionModule __init__

        # Initialize MEC (Medial Entorhinal Cortex) component parameters with proper initial values
        # Use MEC's init_f method to set frequencies
        # self.mec.init_trainable(self.hyper["f_initial"]) already inits in MECModel __init__
        # self.mec_projection.init_trainable() already inits in ProjectionModule __init__
        n_g = self.hyper["n_g"]
        n_g_subsampled = self.hyper["n_g_subsampled"]
        g_mem_std = self.hyper["g_mem_std"]
        n_f = self.hyper["n_f"]

        pass  # We can remove this method if there's nothing to init here
        self.MLP_mu_g_mem = MLP(n_g_subsampled, n_g, hidden_dim=[2 * g for g in n_g])
        self.MLP_mu_g_mem.set_weights(
            -1, [torch.tensor(truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=g_mem_std), dtype=torch.float32) for f in range(n_f)]
        )
        self.MLP_sigma_g_mem = MLP([2 for _ in n_g_subsampled], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def init_iteration(self, g, o, a, M):
        # On the very first iteration, update the batch size based on the data. This is useful when doing analysis on the network with different batch sizes compared to training
        self.hyper["batch_size"] = o.shape[0]
        # Create initial LEC state (x starts as x_filtered since no scaling/normalization yet)
        lec_state = self.lec.init_state(batch_size=self.hyper["batch_size"], device=o.device)
        # Initialise previous abstract location by stacking abstract location prior
        mec_state = self.mec.init_state(batch_size=self.hyper["batch_size"], device=o.device)
        # Create initial HPC state with initialized memory
        hpc_state = self.hpc.init_state(batch_size=self.hyper["batch_size"], device=o.device)
        hpc_state.memory = M if M is not None else self._init_memory(batch_size=int(self.hyper["batch_size"]), device=o.device)
        # And construct new iteration for that g, o, a, and M
        return TEMState(g=g, o=o, a_prev=a, lec_state=lec_state, mec_state=mec_state, hpc_state=hpc_state)

    def gen_o(self, p):
        # Get categorical distribution over observations from grounded location
        # If you actually want to sample observation, you need a reparaterisation trick for categorical distributions
        # Sampling would be the correct way to do this, since observations are discrete, and it's also what the TEM paper says
        # However, it looks like you could also get away with using categorical distribution directly as an approximation of the one-hot observations
        if self.hyper["do_sample"]:
            o, logits = self.f_o(
                p
            )  # This is a placeholder! Should be done using reparameterisation trick (like https://blog.evjang.com/2016/11/tutorial-categorical-variational.html)
        else:
            o, logits = self.f_o(p)
        # Return one-hot (or almost one-hot...) observation obtained from grounded location, and also the non-softmaxed logits
        return o, logits

    def f_o(self, p: Tensor):
        # Calculate categorical probability distribution over observations for a given ground location
        # Legacy behavior: p is only the highest-frequency module with shape (B, n_p[0])
        # p is the outer product of g and x for the highest frequency (p = g^T * x)
        # To get x from p, sum over abstract locations g (transpose of tiling matrix)

        # Project highest-frequency grounded location back to all frequency modules
        # using the inverse tiling operation for each frequency
        p_list = [p] if isinstance(p, Tensor) else p  # Handle both Tensor and List[Tensor]
        x = self.lec_projection.inverse(p_list)

        # Reconstruct compressed features from filtered features (affine transform)
        c = self.lec.reconstruct(x)

        # Decompress c to observation logits using decoder
        logits = self.autoencoder.decode(c)

        # Keep both logits and probabilities
        probability = utils.softmax(logits)
        return probability, logits

    def mec_inference(self, p_x, o, locations, mec_state: MECState):
        g_path = mec_state.g_path
        # Infer abstract location from the combination of [grounded location retrieved from memory by sensory experience] ...
        if self.hyper["use_x_cued_recall"]:
            # Not in paper, but makes sense from symmetry with f_o: first get g from p by "summing over sensory preferences" g = p * W_repeat^T
            g_downsampled = [torch.matmul(p_x[f], torch.t(self.hyper["W_repeat"][f])) for f in range(self.hyper["n_f"])]
            # Then use abstract location after summing over sensory preferences as input to MLP to obtain the inferred abstract location from memory
            mu_g_mem = self.f_mu_g_mem(g_downsampled)
            # Not in paper, but this greatly improves zero-shot inference: provide the uncertainty function of the inferred abstract location with measures of memory quality
            with torch.no_grad():
                # For the first measure, use the grounded location inferred from memory to generate an observation
                o_hat, o_hat_logits = self.gen_o(p_x[0])
                # Then calculate the error between the generated observation and the actual observation: if the memory is working well, this error should be small
                err = utils.squared_error(o, o_hat)
            # The second measure is the vector norm of the inferred abstract location; good memories should have similar vector norms. Concatenate the two measures as input for the abstract location uncertainty function
            sigma_g_input = [torch.cat((torch.sum(g**2, dim=1, keepdim=True), torch.unsqueeze(err, dim=1)), dim=1) for g in mu_g_mem]
            # Not in paper, but recommended by James for stability: get final mean of inferred abstract location by clamping activations between -1 and 1
            mu_g_mem = self.mec.grid.g_clamp(mu_g_mem)
            # And get standard deviation/uncertainty of inferred abstract location by providing uncertainty function with memory quality measures
            sigma_g_mem = self.f_sigma_g_mem(sigma_g_input)
        # ... and [previous abstract location and action (path integration)]
        mu_g_path = g_path.mean
        sigma_g_path = g_path.uncertainty
        # Infer abstract location by combining previous abstract location and grounded location retrieved from memory by current sensory experience
        mu_g, sigma_g = [], []
        for f in range(self.hyper["n_f"]):
            if self.hyper["use_x_cued_recall"]:
                # Then get full gaussian distribution of inferred abstract location by calculating precision weighted mean
                mu, sigma = utils.inv_var_weight([mu_g_path[f], mu_g_mem[f]], [sigma_g_path[f], sigma_g_mem[f]])
            else:
                # Or simply completely ignore the inference memory here, to test if things are working
                mu, sigma = mu_g_path[f], sigma_g_path[f]
            # Append mu and sigma to list for all frequency modules
            mu_g.append(mu)
            sigma_g.append(sigma)
        # Finally (though not in paper), also add object vector cell information to inferred abstract location for environments with shiny objects
        shiny_envs = [location["shiny"] is not None for location in locations]
        if any(shiny_envs):
            # Find for which environments the current location has a shiny object
            shiny_locations = torch.unsqueeze(torch.stack([torch.tensor(location["shiny"], dtype=torch.float) for location in locations if location["shiny"] is not None]), dim=-1)
            # Get abstract location for environments with shiny objects and feed to each of the object vector cell modules
            mu_g_shiny = self.f_mu_g_shiny([shiny_locations for _ in range(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else self.hyper["n_f"])])
            sigma_g_shiny = self.f_sigma_g_shiny([shiny_locations for _ in range(self.hyper["n_f_g"] if self.hyper["separate_ovc"] else self.hyper["n_f"])])
            # Update only object vector modules with shiny-inferred abstract location: start from offset if object vector modules are separate
            module_start = self.hyper["n_f_g"] if self.hyper["separate_ovc"] else 0
            # Inverse variance weighting is associative, so I can just do additional inverse variance weighting to the previously obtained mu and sigma - but only for object vector cell modules!
            for f in range(module_start, self.hyper["n_f"]):
                # Add inferred abstract location from shiny objects to previously obtained position, only for environments with shiny objects
                mu, sigma = utils.inv_var_weight([mu_g[f][shiny_envs, :], mu_g_shiny[f - module_start]], [sigma_g[f][shiny_envs, :], sigma_g_shiny[f - module_start]])
                # In order to update only the environments with shiny objects, without in-place value assignment, construct a mask of shiny environments
                mask = torch.zeros_like(mu_g[f], dtype=torch.bool)
                mask[shiny_envs, :] = True
                # Use mask to update the shiny environment entries in inferred abstract locations
                mu_g[f] = mu_g[f].masked_scatter(mask, mu)
                sigma_g[f] = sigma_g[f].masked_scatter(mask, sigma)
        # Either sample inferred abstract location from combined (precision weighted) distribution or just take mean
        g = [mu_g[f] + sigma_g[f] * np.random.randn() if self.hyper["do_sample"] else mu_g[f] for f in range(self.hyper["n_f"])]
        # Return abstract location inferred from grounded location from memory and previous abstract location
        return g

    def f_mu_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate mean of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        return self.MLP_mu_g_mem(g_downsampled)

    def f_sigma_g_mem(self, g_downsampled):
        # Multi layer perceptron to generate standard deviation of abstract location from down-sampled abstract location, obtained by summing over sensory dimension of grounded location
        sigma = self.MLP_sigma_g_mem(g_downsampled)
        # Not in paper, but also offset this sigma over training, so you can reduce influence of inferred p early on (from runtime, not hyper)
        return [sigma[f] + self.runtime.p2g_scale_offset * self.hyper["p2g_sig_val"] for f in range(self.hyper["n_f"])]

    def f_mu_g_shiny(self, shiny):
        # Multi layer perceptron to generate mean of abstract location from boolean location shiny-ness
        mu_g = self.mec.ovc.MLP_mu_g_shiny(shiny)
        # Take absolute because James wants object vector cells to be positive
        mu_g = [torch.abs(mu) for mu in mu_g]
        # Then apply clamp and leaky relu to get object vector module activations, like it's done for ground location activations
        g = self.f_p(mu_g)
        return g

    def f_sigma_g_shiny(self, shiny):
        # Multi layer perceptron to generate standard deviation of abstract location from boolean location shiny-ness
        return self.mec.ovc.MLP_sigma_g_shiny(shiny)


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
        if initial is not None:
            state = initial
        else:
            # Create fresh initial state: derive batch size from observation tensor
            batch_size = int(o_0.shape[0]) if o_0.ndim > 1 else 1
            # Initialize with reset action (episode boundary)
            state = model.init_iteration(locations_0, o_0, [None] * batch_size, None)

        # Initialize prev-values for first forward pass
        self._a_prev = state.a_prev
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
        prediction, state = self.model(o, locations, self._a_prev, self._state)

        # Build state with updated components
        state.g = locations
        state.a_prev = a

        # Update prev-values for next iteration
        self._a_prev = a
        self._state = state

        # Build labels for current timestep
        labels = TEMLabel(o=o, locations=locations)

        return prediction, state, labels
