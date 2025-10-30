from typing import List

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from torch_tem import utils


class SensoryState(BaseModel):
    """Container for sensory processing outputs.

    This state object captures all intermediate and final representations
    produced by the sensory processing pipeline, enabling downstream modules
    to access whichever representation is most appropriate for their needs.

    Attributes:
        raw: Raw one-hot observation [batch, n_x]
        compressed: Two-hot encoded observation [batch, n_x_c]
        filtered: Temporally smoothed representations per frequency [n_freq × [batch, n_x_c]]
        normalized: Zero-mean, unit-norm representations per frequency [n_freq × [batch, n_x_c]]
        memory_ready: Projected to place cell dimensions [n_freq × [batch, n_p]]
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    raw: Tensor
    compressed: Tensor
    filtered: List[Tensor]
    normalized: List[Tensor]
    memory_ready: List[Tensor]


class SensoryProcessor(nn.Module):
    """Processes sensory observations through compression, filtering, and normalization.

    This module implements the sensory processing pipeline that transforms raw one-hot
    observations into memory-ready representations. The pipeline consists of:
    1. Compression: One-hot → Two-hot encoding (reduces dimensionality)
    2. Temporal filtering: Exponential smoothing across time steps
    3. Normalization: Zero-mean, unit-norm representations
    4. Memory preparation: Transform to place cell dimension via tile matrices

    Args:
        n_frequencies: Number of hierarchical frequency modules (typically 5)
        initial_frequencies: Initial filter coefficients per frequency [0.99, 0.3, ...]
            Higher values = faster adaptation to new observations
        two_hot_table: Lookup table [n_codes x n_bits] for one-hot to two-hot compression
        tile_matrices: List of tile matrices [n_freq x n_x_c x n_p] for memory projection

    Attributes:
        n_freq: Number of frequency modules being processed
        two_hot_table: Compression lookup table
        W_tile: Tile matrices for outer product computation
        alpha: Learned temporal filtering rates (logit-space, applied via sigmoid)
        w_p: Learned weighting for memory preparation (applied via sigmoid)
    """

    def __init__(self, n_frequencies: int, initial_frequencies: List[float], two_hot_table: torch.Tensor, tile_matrices: List[torch.Tensor]):
        super().__init__()
        self.n_freq = n_frequencies
        self.two_hot_table = two_hot_table
        self.W_tile = tile_matrices

        # Temporal filtering parameters (learned)
        # Stored in logit space: logit(α) = log(α/(1-α))
        # Applied as: α = sigmoid(logit(α))
        # This ensures α stays in (0,1) during optimization
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(np.log(f / (1 - f)), dtype=torch.float)) for f in initial_frequencies])

        # Memory preparation weights (learned)
        # Initialized to 1.0, applied via sigmoid to ensure (0,1) range
        # Controls the strength of sensory→memory projection
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(n_frequencies)])

    def forward(self, x_raw: torch.Tensor, x_prev: List[torch.Tensor]) -> SensoryState:
        """Process sensory observation through the complete pipeline.

        Args:
            x_raw: Raw one-hot observations [batch x n_x]
            x_prev: Previous filtered states per frequency [n_freq x [batch x n_x_c]]

        Returns:
            SensoryState containing all intermediate and final representations:
                - raw: Original one-hot input
                - compressed: Two-hot encoded
                - filtered: Temporally smoothed per frequency
                - normalized: Zero-mean, unit-norm per frequency
                - memory_ready: Projected to place cell dimensions
        """
        x_compressed = self.compress(x_raw)  # One-hot → two-hot (dimensionality reduction)
        x_filtered = self.temporal_filter(x_compressed, x_prev)  # Exponential smoothing
        x_normalized = self.normalize(x_filtered)  # Center and normalize
        x_memory = self.prepare_for_memory(x_normalized)  # Project to memory space

        return SensoryState(raw=x_raw, compressed=x_compressed, filtered=x_filtered, normalized=x_normalized, memory_ready=x_memory)

    def compress(self, x_onehot: torch.Tensor) -> torch.Tensor:
        """Compress one-hot to two-hot representation using lookup table.

        Two-hot encoding reduces dimensionality while preserving distinctiveness.
        Each observation is represented by exactly two active bits, providing
        ~C(n,2) possible codes (combinatorial capacity).

        Args:
            x_onehot: One-hot encoded observations [batch × n_x]

        Returns:
            Two-hot encoded observations [batch × n_x_c] where n_x_c < n_x
        """
        return torch.stack([self.two_hot_table[i] for i in torch.argmax(x_onehot, dim=1)], dim=0)

    def temporal_filter(self, x_compressed: torch.Tensor, x_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply exponential temporal filtering (moving average).

        Each frequency module maintains its own filter rate:
            x_filtered[t] = (1-α)·x[t-1] + α·x[t]

        Higher α → faster adaptation (higher frequency)
        Lower α → slower adaptation (lower frequency, more smoothing)

        Args:
            x_compressed: Current compressed observation [batch × n_x_c]
            x_prev: Previous filtered states [n_freq × [batch × n_x_c]]

        Returns:
            Filtered observations per frequency [n_freq × [batch × n_x_c]]
        """
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.n_freq)]
        return [(1 - alpha[f]) * x_prev[f] + alpha[f] * x_compressed for f in range(self.n_freq)]

    def normalize(self, x_filtered: List[torch.Tensor]) -> List[torch.Tensor]:
        """Normalize sensory input to zero-mean, unit-norm.

        Normalization ensures consistent scale across different observations
        and improves numerical stability in downstream memory operations.

        Steps:
            1. Center: subtract mean
            2. ReLU: remove negative values (sparsity)
            3. Normalize: scale to unit norm

        Args:
            x_filtered: Temporally filtered observations [n_freq × [batch × n_x_c]]

        Returns:
            Normalized observations [n_freq × [batch × n_x_c]]
        """
        return [utils.normalise(utils.relu(x_filtered[f] - torch.mean(x_filtered[f]))) for f in range(self.n_freq)]

    def prepare_for_memory(self, x_normalized: List[torch.Tensor]) -> List[torch.Tensor]:
        """Project sensory representations to place cell dimensions.

        Prepares sensory input for Hebbian memory operations by:
            1. Weighting by learned w_p (controls sensory influence)
            2. Tiling via W_tile matrices to match place cell dimensions

        The result can be used as a query to the associative memory to
        retrieve grounded location (place cell) representations.

        Args:
            x_normalized: Normalized sensory input [n_freq × [batch × n_x_c]]

        Returns:
            Memory-ready representations [n_freq × [batch × n_p]]
        """
        return [torch.sigmoid(self.w_p[f]) * torch.matmul(x_normalized[f], self.W_tile[f]) for f in range(self.n_freq)]
