from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from torch_tem import utils
from torch_tem.config import ArchitectureConfig, StaticMatrices
from torch_tem.core.states import SensoryState


class SensoryProcessor(nn.Module):
    """Handles all sensory observation processing."""

    def __init__(self, arch_config: ArchitectureConfig, static_matrices: StaticMatrices):
        super().__init__()
        self.config = arch_config
        self.matrices = static_matrices

        # Temporal filtering parameters
        self.alpha = nn.ParameterList(
            [nn.Parameter(torch.tensor(np.log(arch_config.f_initial_extended[f] / (1 - arch_config.f_initial_extended[f])), dtype=torch.float)) for f in range(arch_config.n_f)]
        )

        # Memory preparation weights
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(arch_config.n_f)])

    def forward(self, x_raw: torch.Tensor, x_prev: List[torch.Tensor]) -> SensoryState:
        """Process sensory observation through compression, filtering, and normalization."""
        # Compress from one-hot to two-hot
        x_compressed = self.compress(x_raw)

        # Temporal filtering
        x_filtered = self.temporal_filter(x_compressed, x_prev)

        # Normalize
        x_normalized = self.normalize(x_filtered)

        # Prepare for memory
        x_memory = self.prepare_for_memory(x_normalized)

        return SensoryState(raw=x_raw, compressed=x_compressed, filtered=x_filtered, normalized=x_normalized, memory_ready=x_memory)

    def compress(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Compress one-hot to two-hot representation."""
        return torch.stack([self.matrices.two_hot_table[i] for i in torch.argmax(x_onehot, dim=1)], dim=0)

    def temporal_filter(self, x_compressed: torch.Tensor, x_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply exponential temporal filtering."""
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.config.n_f)]
        return [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_compressed for f in range(self.config.n_f)]

    def normalize(self, x_filtered: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize sensory input."""
        return [utils.normalise(utils.relu(x[f] - torch.mean(x[f]))) for f in range(self.config.n_f)]

    def prepare_for_memory(self, x_normalized: List[torch.Tensor]) -> List[torch.Tensor]:
        """Prepare sensory input for memory operations."""
        return [torch.sigmoid(self.w_p[f]) * torch.matmul(x_normalized[f], self.matrices.W_tile[f]) for f in range(self.config.n_f)]
