from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import HPCSettings
from torch_tem.types import Matrix


@dataclass
class HPCState:
    """State container for HPC dynamics."""

    p: List[Tensor]  # Multi-frequency filtered features
    memory: List[Matrix]  # Memory matrices

    def detach(self) -> "HPCState":
        """Return a detached copy suitable for storing as `prev_iter`."""
        return HPCState(
            p=[v.detach() for v in self.p] if self.p is not None else None,
            memory=[m.detach() for m in self.memory] if self.memory is not None else None,
        )


@dataclass
class HPCRuntime:
    """Runtime values injected by training (not architectural parameters)."""

    eta: float = 0.0
    hebbian_decay: float = 0.9999


class HPCModel(nn.Module):

    def __init__(self, i_attractor: int, shape: List[int], f_init: List[float], settings: HPCSettings):
        super().__init__()
        self._settings = settings

        # Store hyperparameters
        self._n_p = shape
        self._i_attractor = i_attractor

        if self._i_attractor < 1 or self._i_attractor > len(self._n_p):
            raise ValueError(f"i_attractor must be in [1, n_freq]. Got i_attractor={self._i_attractor}, n_freq={len(self._n_p)}.")

        # Store runtime parameters
        self.runtime = HPCRuntime()

        # Store mask as a buffer since it is not learnable
        self.register_buffer("p_update_mask", p_update_mask(self, f_init))

        # Initialize MLPs for generating grounded location statistics
        self.MLP_sigma_p = MLP(shape, shape, activation=[torch.tanh, torch.exp])

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> HPCState:
        """Initialize HPC state with empty memory and zeroed features."""
        p_init = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        memory_init = [torch.zeros((sum(self.shape), sum(self.shape)), device=device)]  # Update to TEMModel._init_memory
        return HPCState(p=p_init, memory=memory_init)

    # def _init_memory(self, *, batch_size: int, device: torch.device) -> List[Tensor]:
    #     """Create initial Hebbian memory matrices in the legacy [M_gen, M_inf?] format."""
    #     m0 = torch.zeros((batch_size, sum(self.shape), sum(self.shape)), dtype=torch.float, device=device)
    #     memory = [m0]
    #     if self.hyper["use_x_cued_recall"]:
    #         memory.append(m0 if self.hyper["common_memory"] else m0.clone())
    #     return memory

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._n_p

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return len(self.shape)

    @property
    def i_attractor(self) -> int:
        """Number of attractor iterations."""
        return self._i_attractor

    @property
    def i_attractor_max_freq_inf(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in inference model."""
        return [self.i_attractor for _ in range(self.n_freq)]

    @property
    def i_attractor_max_freq_gen(self) -> list[int]:
        """Maximum iterations of attractor dynamics per frequency in generative model."""
        return [self.i_attractor - freq_nr for freq_nr in range(self.i_attractor)] + [self.i_attractor for _ in range(self.n_freq - self.i_attractor)]

    def forward(self, *, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        raise NotImplementedError("HPCModel forward pass not yet implemented.")

    def set_runtime(self, *, eta: float, hebbian_decay: float) -> None:
        """Update runtime hyperparameters for Hebbian updates."""
        self.runtime.eta = float(eta)
        self.runtime.hebbian_decay = float(hebbian_decay)

    def generative(self, p_g, state: HPCState) -> Tuple[List[Tensor], HPCState]:
        # Retreive memory: do pattern completion on abstract location to get grounded location
        if not self._settings.do_sample:
            return p_g, HPCState(p=p_g, memory=state.memory)
        sigma_p = self.MLP_sigma_p(p_g)
        p = [p_g[f] + sigma_p[f] * torch.randn_like(sigma_p[f]) for f in range(self.n_freq)]
        return p, HPCState(p=p, memory=state.memory)

    def inference(self, x_: List[Tensor], g_: List[Tensor], state: HPCState) -> Tuple[List[Tensor], HPCState]:
        mu_p = self.f_p([g_[f] * x_[f] for f in range(self.n_freq)])  # This is element-wise multiplication
        if not self._settings.do_sample:
            return mu_p, HPCState(p=mu_p, memory=state.memory)
        sigma_p = self.MLP_sigma_p(mu_p)
        p = [mu_p[f] + sigma_p[f] * torch.randn_like(sigma_p[f]) for f in range(self.n_freq)]
        return p, HPCState(p=p, memory=state.memory)

    def attractor(self, p_query, M, retrieve_it_mask=None):
        # Retreive grounded location from attractor network memory with weights M by pattern-completing query
        # For example, initial attractor input can come from abstract location (g_) or sensory experience (x_)
        # Start by flattening query grounded locations across frequency modules
        h_t = torch.cat(p_query, dim=1)
        # Apply activation function to initial memory index
        h_t = self.f_p(h_t)
        # Hierarchical retrieval (not in paper) is implemented by early stopping retrieval for low frequencies, using a mask. If not specified: initialise mask as all 1s
        if retrieve_it_mask is None:
            retrieve_it_mask = [torch.ones(sum(self.shape), device=h_t.device, dtype=h_t.dtype) for _ in range(self.i_attractor)]
        else:
            retrieve_it_mask = [m.to(device=h_t.device, dtype=h_t.dtype) for m in retrieve_it_mask]
        # Iterate attractor dynamics to do pattern completion
        for tau in range(self.i_attractor):
            # Apply one iteration of attractor dynamics, but only where there is a 1 in the mask. NB retrieve_it_mask entries have only one row, but are broadcasted to batch_size
            h_t = (1 - retrieve_it_mask[tau]) * h_t + retrieve_it_mask[tau] * (self.f_p(self._settings.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), M))))
        # Make helper list of cumulative neurons per frequency module for grounded locations
        n_p = np.cumsum(np.concatenate(([0], self.shape)))
        # Now re-cast the grounded location into different frequency modules, since memory retrieval turned it into one long vector
        p = [h_t[:, n_p[f] : n_p[f + 1]] for f in range(self.n_freq)]
        return p

    def hebbian(self, M_prev, p_inf, p_gen_gi, do_hierarchical_connections=True):
        # Create new ground memory for attractor network by setting weights to outer product of learned vectors
        # p_inf corresponds to p in the paper, and p_gen_gi corresponds to p^.
        # The order of p + p^ and p - p^ is reversed since these are row vectors, instead of column vectors in the paper.
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inf + p_gen_gi, 2), torch.unsqueeze(p_inf - p_gen_gi, 1)))
        # Multiply by connection vector, e.g. only keeping weights from low to high frequencies for hierarchical retrieval
        if do_hierarchical_connections:
            M_new = M_new * self.p_update_mask
        # Store grounded location in attractor network memory with weights M by Hebbian learning of pattern
        # Rate of remembering controlled by eta, rate of forgetting by hebbian_decay (from runtime, not hyper)
        M = torch.clamp(self.runtime.hebbian_decay * M_prev + self.runtime.eta * M_new, min=-1, max=1)
        return M

    def f_p(self, p):
        # Calculate activation for inferred grounded location, using a leaky relu for sparsity. Either apply to full multi-frequency grounded location or single frequency module
        return [utils.leaky_relu(torch.clamp(p_f, min=-1, max=1)) for p_f in p] if type(p) is list else utils.leaky_relu(torch.clamp(p, min=-1, max=1))


def p_update_mask(hpc: HPCModel, f_init: List[float]) -> Tensor:
    n_p, n_f, i_attractor = hpc.shape, hpc.n_freq, hpc.i_attractor
    mask = torch.zeros((np.sum(n_p), np.sum(n_p)), dtype=torch.float)
    n_p = np.cumsum(np.concatenate(([0], n_p)))

    # Entry M_ij (row i, col j) is the connection FROM cell i TO cell j
    for f_from in range(n_f):
        for f_to in range(n_f):
            # For connections that involve separate object vector modules
            if f_from >= i_attractor or f_to >= i_attractor:
                # Connection between object vector modules: only allow from low to high frequency
                if f_from >= i_attractor and f_to >= i_attractor:
                    if f_init[f_from] <= f_init[f_to]:
                        mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
                # Connection between object vector and normal modules: allow any connections
                else:
                    mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0
            # Connection between abstract location frequency modules: only from low to high frequency
            else:
                if f_init[f_from] <= f_init[f_to]:
                    mask[n_p[f_from] : n_p[f_from + 1], n_p[f_to] : n_p[f_to + 1]] = 1.0

    return mask


def p_retrieve_mask_inf(hpc: HPCModel) -> List[torch.Tensor]:
    n_p, n_f, i_attractor = hpc.shape, hpc.n_freq, hpc.i_attractor
    masks = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]
    n_p = np.cumsum(np.concatenate(([0], n_p)))

    # For each frequency, insert ones in the mask for those iterations
    for f, max_i in enumerate(hpc.i_attractor_max_freq_inf):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0

    return masks


def p_retrieve_mask_gen(hpc: HPCModel) -> List[torch.Tensor]:
    n_p, n_f, i_attractor = hpc.shape, hpc.n_freq, hpc.i_attractor
    masks = [torch.zeros(sum(n_p)) for _ in range(i_attractor)]
    n_p = np.cumsum(np.concatenate(([0], n_p)))

    # For each frequency, insert ones in the mask for those iterations
    for f, max_i in enumerate(hpc.i_attractor_max_freq_gen):
        for i in range(max_i):
            masks[i][n_p[f] : n_p[f + 1]] = 1.0

    return masks
