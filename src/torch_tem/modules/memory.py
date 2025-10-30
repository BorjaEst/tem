from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch_tem import utils
from torch_tem.config import ArchitectureConfig, MemoryConfig, StaticMatrices


class MemorySystem(nn.Module):
    """Hebbian associative memory with attractor dynamics."""

    def __init__(self, arch_config: ArchitectureConfig, mem_config: MemoryConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.arch = arch_config
        self.mem = mem_config
        self.matrices = static_matrices

    def retrieve(self, query: List[torch.Tensor], memory_matrix: torch.Tensor, mode: str = "inference") -> List[torch.Tensor]:
        """Pattern completion via attractor dynamics."""
        # Concatenate frequency modules
        h_t = torch.cat(query, dim=1)
        h_t = self._activation(h_t)

        # Get retrieval mask
        masks = self.matrices.p_retrieve_mask_inf if mode == "inference" else self.matrices.p_retrieve_mask_gen

        # Attractor iterations
        for tau in range(self.mem.i_attractor):
            h_t = (1 - masks[tau]) * h_t + masks[tau] * self._activation(self.mem.kappa * h_t + torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), memory_matrix)))

        # Split back into frequency modules
        n_p_cumsum = np.cumsum(np.concatenate(([0], self.arch.n_p)))
        return [h_t[:, n_p_cumsum[f] : n_p_cumsum[f + 1]] for f in range(self.arch.n_f)]

    def update(self, memory_prev: torch.Tensor, p_inferred: List[torch.Tensor], p_generated: List[torch.Tensor], hierarchical: bool = True) -> torch.Tensor:
        """Hebbian memory update."""
        # Concatenate frequencies
        p_inf_flat = torch.cat(p_inferred, dim=1)
        p_gen_flat = torch.cat(p_generated, dim=1)

        # Compute outer product update
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inf_flat + p_gen_flat, 2), torch.unsqueeze(p_inf_flat - p_gen_flat, 1)))

        # Apply connectivity mask if hierarchical
        if hierarchical:
            M_new = M_new * self.matrices.p_update_mask

        # Hebbian update with forgetting
        M = torch.clamp(self.mem.lambda_ * memory_prev + self.mem.eta * M_new, min=-1, max=1)
        return M

    def _activation(self, p: torch.Tensor) -> torch.Tensor:
        """Leaky ReLU activation for place cells."""
        return utils.leaky_relu(torch.clamp(p, min=-1, max=1))
