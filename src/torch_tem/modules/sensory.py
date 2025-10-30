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
        filtered: Temporally smoothed representations per frequency [n_freq x [batch, n_x_c]]
        normalized: Zero-mean, unit-norm representations per frequency [n_freq x [batch, n_x_c]]
        memory_ready: Projected to place cell dimensions [n_freq x [batch, n_p]]
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

    def __init__(
        self,
        n_frequencies: int,
        initial_frequencies: List[float],
        two_hot_table: torch.Tensor,
        tile_matrices: List[torch.Tensor],
    ):
        super().__init__()
        self.n_freq = n_frequencies
        self.two_hot_table = two_hot_table
        self.W_tile = tile_matrices

        # Temporal filtering parameters (learned)
        # Stored in logit space: logit(α) = log(α/(1-α))
        # Applied as: α = sigmoid(logit(α))
        # This ensures α stays in (0,1) during optimization
        self.alpha = nn.ParameterList([
            nn.Parameter(torch.tensor(np.log(f / (1 - f)), dtype=torch.float)) 
            for f in initial_frequencies
        ])  # fmt: skip

        # Memory preparation weights (learned)
        # Initialized to 1.0, applied via sigmoid to ensure (0,1) range
        # Controls the strength of sensory→memory projection
        self.w_p = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) 
            for _ in range(n_frequencies)
        ])  # fmt: skip

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
            x_onehot: One-hot encoded observations [batch x n_x]

        Returns:
            Two-hot encoded observations [batch x n_x_c] where n_x_c < n_x
        """
        return torch.stack([self.two_hot_table[i] for i in torch.argmax(x_onehot, dim=1)], dim=0)

    def temporal_filter(self, x_compressed: torch.Tensor, x_prev: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply exponential temporal filtering (moving average).

        Each frequency module maintains its own filter rate:
            x_filtered[t] = (1-α)·x[t-1] + α·x[t]

        Higher α → faster adaptation (higher frequency)
        Lower α → slower adaptation (lower frequency, more smoothing)

        Args:
            x_compressed: Current compressed observation [batch x n_x_c]
            x_prev: Previous filtered states [n_freq x [batch x n_x_c]]

        Returns:
            Filtered observations per frequency [n_freq x [batch x n_x_c]]
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
            x_filtered: Temporally filtered observations [n_freq x [batch x n_x_c]]

        Returns:
            Normalized observations [n_freq x [batch x n_x_c]]
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
            x_normalized: Normalized sensory input [n_freq x [batch x n_x_c]]

        Returns:
            Memory-ready representations [n_freq x [batch x n_p]]
        """
        return [torch.sigmoid(self.w_p[f]) * torch.matmul(x_normalized[f], self.W_tile[f]) for f in range(self.n_freq)]


# ==============================================================================
# Usage Example
# ==============================================================================

if __name__ == "__main__":
    """
    Example usage of SensoryProcessor for processing observations.

    This demonstrates the complete sensory processing pipeline from raw
    one-hot observations to memory-ready representations.
    """
    import torch

    # Configuration
    batch_size = 4
    n_observations = 45  # Number of possible observations
    n_compressed = 10  # Compressed observation dimension
    n_frequencies = 5  # Number of frequency modules
    n_grid = 30  # Grid cell dimension for memory projection

    print("=" * 70)
    print("SensoryProcessor Example")
    print("=" * 70)
    print("Setting up processor with:")
    print(f"  - {n_observations} possible observations")
    print(f"  - Compression to {n_compressed} dimensions")
    print(f"  - {n_frequencies} frequency modules")
    print(f"  - Memory projection to {n_grid * n_compressed} dimensions")

    # Create two-hot encoding table (simplified for example)
    two_hot_table = torch.eye(n_observations)[:, :n_compressed]  # Simplified: just truncate

    # Create tile matrices for outer product
    tile_matrices = [torch.randn(n_compressed, n_grid * n_compressed) / 10 for _ in range(n_frequencies)]

    # Initialize frequencies (high to low frequency)
    initial_frequencies = [0.99, 0.3, 0.09, 0.03, 0.01]

    # Create processor
    processor = SensoryProcessor(
        n_frequencies=n_frequencies,
        initial_frequencies=initial_frequencies,
        two_hot_table=two_hot_table,
        tile_matrices=tile_matrices,
    )

    # Create sample one-hot observations
    x_raw = torch.zeros(batch_size, n_observations)
    x_raw[0, 5] = 1.0  # Observation 5 for sample 0
    x_raw[1, 10] = 1.0  # Observation 10 for sample 1
    x_raw[2, 20] = 1.0  # Observation 20 for sample 2
    x_raw[3, 30] = 1.0  # Observation 30 for sample 3

    # Initialize previous states (first time step)
    x_prev = [torch.zeros(batch_size, n_compressed) for _ in range(n_frequencies)]

    print(f"\nProcessing {batch_size} observations...")

    # Process observations
    with torch.no_grad():  # Example doesn't need gradients
        sensory_state = processor(x_raw, x_prev)

    # Display results
    print(f"\n✅ Pipeline complete!")
    print(f"\nPipeline stages:")
    print(f"  1. Raw (one-hot):      {sensory_state.raw.shape}")
    print(f"  2. Compressed:         {sensory_state.compressed.shape}")
    print(f"  3. Filtered (×{n_frequencies}):      each {sensory_state.filtered[0].shape}")
    print(f"  4. Normalized (×{n_frequencies}):    each {sensory_state.normalized[0].shape}")
    print(f"  5. Memory-ready (×{n_frequencies}):  each {sensory_state.memory_ready[0].shape}")

    print(f"\nData reduction: {n_observations} → {n_compressed} dimensions ({n_compressed/n_observations:.1%})")
    print(f"Memory expansion: {n_compressed} → {sensory_state.memory_ready[0].shape[1]} dimensions")

    # Demonstrate temporal filtering across multiple steps
    print(f"\nTemporal filtering demonstration:")
    print(f"Processing same observations 3 times to show frequency response...")

    current_filtered = x_prev
    for step in range(3):
        with torch.no_grad():
            state = processor(x_raw, current_filtered)
        current_filtered = state.filtered

        # Show convergence at different frequencies
        high_freq_val = state.filtered[0][0, 0].item()
        low_freq_val = state.filtered[-1][0, 0].item()

        print(f"  Step {step + 1}: High freq (α≈0.99) = {high_freq_val:6.3f}, " f"Low freq (α≈0.01) = {low_freq_val:6.3f}")

    print(f"\n💡 High frequency adapts quickly, low frequency smooths gradually")
    print(f"💡 This multi-scale representation helps with both rapid and stable learning")
    print("=" * 70)
