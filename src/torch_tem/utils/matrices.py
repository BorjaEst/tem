"""Matrix generation utilities for torch_tem package."""

from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.special import comb
from torch import Tensor

from torch_tem.types import BatchedMemory, GroundedLocation, Matrix, MultiScaleCode, Vector


def create_initial_memory(n_p_total: int, batch_size: int, dual_memory: bool, device: torch.device) -> List[BatchedMemory]:
    """Create initial zero memory matrices for HPC state initialization.

    Args:
        n_p_total: Total number of place cells (sum of n_p across frequencies)
        batch_size: Number of parallel memory instances
        dual_memory: If True, create separate M_gen and M_inf; if False, create single shared memory
        device: Device for tensor allocation

    Returns:
        List containing [M_gen, M_inf] if dual_memory=True, else [M_gen, None]

    Example:
        >>> memory = create_initial_memory(n_p_total=50, batch_size=8, dual_memory=True, device='cpu')
        >>> M_gen, M_inf = memory
        >>> M_gen.shape
        torch.Size([8, 50, 50])
    """
    M_gen = torch.zeros(batch_size, n_p_total, n_p_total, device=device)
    M_inf = torch.zeros(batch_size, n_p_total, n_p_total, device=device) if dual_memory else None
    return [M_gen, M_inf]


def squared_error_freq(value: Union[Vector, MultiScaleCode], target: Union[Vector, MultiScaleCode]) -> Union[Vector, MultiScaleCode]:
    """Compute squared error across frequencies.

    Helper function for computing ||value - target||² * 0.5 across frequency modules.
    Handles both single tensors and lists of tensors (for multi-frequency representations).

    Args:
        value: Predicted value(s)
        target: Target value(s)

    Returns:
        Squared error(s), scaled by 0.5
    """
    if isinstance(value, list) and isinstance(target, list):
        return [torch.sum((v - t) ** 2, dim=1) * 0.5 for v, t in zip(value, target)]
    return torch.sum((value - target) ** 2, dim=1) * 0.5


def create_W_repeat(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Matrix]:
    """Create repeat matrices for outer product computation.

    Matrix for repeating abstract location g to do outer product with sensory
    information x using elementwise product after matrix multiplication.

    Args:
        n_g_subsampled: Subsampled abstract location dimensions per frequency
        n_x_f: Sensory dimensions per frequency

    Returns:
        List of repeat matrices, one per frequency module
    """
    return [torch.tensor(np.kron(np.eye(g), np.ones((1, x))), dtype=torch.float) for g, x in zip(n_g_subsampled, n_x_f)]


def create_W_tile(n_g_subsampled: List[int], n_x_f: List[int]) -> List[Matrix]:
    """Create tile matrices for outer product computation.

    Matrix for tiling sensory observation x to do outer product with abstract
    location using elementwise product after matrix multiplication.

    Args:
        n_g_subsampled: Subsampled abstract location dimensions per frequency
        n_x_f: Sensory dimensions per frequency

    Returns:
        List of tile matrices, one per frequency module
    """
    return [torch.tensor(np.kron(np.ones((1, g)), np.eye(x)), dtype=torch.float) for g, x in zip(n_g_subsampled, n_x_f)]


def create_g_downsample(n_g: List[int], n_g_subsampled: List[int]) -> List[Matrix]:
    """Create downsampling matrices for abstract location.

    Downsampling matrix to go from grid cells to compressed grid cells for
    indexing memories by simply taking only the first n_g_subsampled grid cells.

    Args:
        n_g: Full abstract location dimensions per frequency
        n_g_subsampled: Subsampled abstract location dimensions per frequency

    Returns:
        List of downsampling matrices, one per frequency module
    """
    return [torch.cat([torch.eye(dim_out, dtype=torch.float), torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)]) for dim_in, dim_out in zip(n_g, n_g_subsampled)]


def create_W_random_projection(n_g: List[int], n_p: List[int], sparsity: float = 1.0, seed: Optional[int] = None) -> List[Matrix]:
    """Create random fixed projection matrices from entorhinal cortex to hippocampus.

    Biologically-inspired alternative to downsampling + W_repeat expansion.
    Models the random connectivity from EC (grid cells) to HPC (place cells)
    as observed in experimental data and used in CAN models (Chandra et al. 2025).

    This replaces the two-step structured transformation:
        g → downsample → g_ → W_repeat expansion → hippocampal space
    With a single direct random projection:
        g → W_random → hippocampal space

    The random projection is fixed (non-learnable) and can be sparse to match
    biological connectivity patterns (~10-20% in real circuits).

    Args:
        n_g: Full abstract location dimensions per frequency (EC grid cells)
        n_p: Grounded location dimensions per frequency (HPC place cells)
        sparsity: Connection probability (1.0 = fully connected, 0.1 = 10% connectivity)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of random projection matrices [n_g[f], n_p[f]], one per frequency module

    Example:
        >>> n_g = [36, 30, 24]  # Grid cell dimensions
        >>> n_p = [96, 80, 64]  # Place cell dimensions
        >>> W_random = create_W_random_projection(n_g, n_p, sparsity=0.15)
        >>> # Use in projection head:
        >>> g_ = [g[f] @ W_random[f] for f in range(n_f)]

    References:
        Chandra et al. (2025). "Episodic and associative memory from spatial
        scaffolds in the hippocampus." CAN model architecture.
    """
    if seed is not None:
        torch.manual_seed(seed)

    matrices = []
    for g_dim, p_dim in zip(n_g, n_p):
        # Random Gaussian initialization scaled by input dimension
        # This ensures variance is maintained across the projection
        W = torch.randn(g_dim, p_dim, dtype=torch.float) / np.sqrt(g_dim)

        # Apply sparsity mask if requested
        if sparsity < 1.0:
            mask = torch.rand(g_dim, p_dim) < sparsity
            W = W * mask.float()
            # Rescale to maintain expected magnitude after sparsification
            W = W / np.sqrt(sparsity)

        matrices.append(W)

    return matrices


def create_encoding_table(n_x: int, n_x_c: int, n_hot: int = 2) -> List[Vector]:
    """Create n-hot encoding lookup table.

    Generates a lookup table for converting one-hot observations to n-hot
    compressed representations. Each observation is encoded using exactly
    `n_hot` active units from `n_x_c` dimensions.

    Args:
        n_x: Number of possible observations (must be <= C(n_x_c, n_hot))
        n_x_c: Compressed sensory dimension
        n_hot: Number of active units per code (1, 2, 3, etc.)
            - n_hot=1: One-hot (identity, no compression unless n_x > n_x_c)
            - n_hot=2: Two-hot (default, typically 45 → 10)
            - n_hot=3: Three-hot (more distributed, e.g., 220 → 12)

    Returns:
        List of n-hot code tensors, one per possible observation [n_x, n_x_c]

    Raises:
        ValueError: If n_x > C(n_x_c, n_hot) (too many observations for compression)

    Example:
        >>> # Two-hot encoding: 45 observations → 10 dimensions
        >>> table = create_encoding_table(n_x=45, n_x_c=10, n_hot=2)
        >>> len(table)
        45
        >>> table[0].sum()
        2.0

        >>> # Three-hot encoding: 220 observations → 12 dimensions
        >>> table = create_encoding_table(n_x=220, n_x_c=12, n_hot=3)
        >>> len(table)
        220
        >>> table[0].sum()
        3.0
    """

    # Validate: number of observations must not exceed possible n-hot codes
    max_codes = int(comb(n_x_c, n_hot))
    if n_x > max_codes:
        raise ValueError(f"Cannot encode {n_x} observations with {n_hot}-hot codes in {n_x_c} dimensions. " f"Maximum possible codes: C({n_x_c}, {n_hot}) = {max_codes}")

    # Generate all possible n-hot codes using combinations
    # combinations(range(n_x_c), n_hot) gives all ways to choose n_hot positions
    encoding_table = []
    for active_positions in combinations(range(n_x_c), n_hot):
        # Create zero vector
        code = [0] * n_x_c
        # Activate n_hot positions
        for pos in active_positions:
            code[pos] = 1
        # Add to table
        encoding_table.append(torch.tensor(code, dtype=torch.float))

        # Stop when we have enough codes for all observations
        if len(encoding_table) >= n_x:
            break

    return encoding_table


def create_two_hot_table(n_x: int, n_x_c: int) -> List[Vector]:
    """Create two-hot encoding lookup table (backward compatibility wrapper).

    DEPRECATED: Use create_encoding_table(n_x, n_x_c, n_hot=2) instead.

    This function is maintained for backward compatibility. New code should use
    the more general create_encoding_table() function.

    Args:
        n_x: Number of possible observations
        n_x_c: Compressed sensory dimension

    Returns:
        List of two-hot code tensors, one per possible observation
    """
    return create_encoding_table(n_x, n_x_c, n_hot=2)


def split_to_frequencies(p_flat: Vector, n_p: List[int]) -> GroundedLocation:
    """Split concatenated place cell tensor into per-frequency list.

    This utility function converts between the two common formats for grounded
    location representations in TEM:

    - Concatenated format [B, sum(n_p)]: Used by memory operations (AttractorDynamics,
      MemoryStorage) for efficient matrix multiplication with Hebbian matrices
    - Per-frequency format List[n_f] of [B, n_p[f]]: Used by hierarchical operations
      (GroundedLocInference, AbstractLocInference) that process each
      frequency module independently

    This conversion is frequently needed after memory retrieval operations that
    return concatenated tensors, before passing to components that expect
    per-frequency lists.

    Args:
        p_flat: Concatenated grounded location [B, sum(n_p)] where B is batch size
                and sum(n_p) is total place cells across all frequency modules
        n_p: List of place cell dimensions per frequency module [n_p[0], n_p[1], ...]

    Returns:
        List of [n_f] tensors, each of shape [B, n_p[f]] representing place cell
        activity for each frequency module separately

    Example:
        >>> # After memory retrieval
        >>> p_concat = attractor.retrieve(query, M_inf, for_inference=True)  # [B, 96]
        >>> # Convert to per-frequency format for AbstractLocInference
        >>> n_p = [40, 32, 24]  # 3 frequency modules
        >>> p_list = split_to_frequencies(p_concat, n_p)  # List of [B,40], [B,32], [B,24]
        >>> # Now ready for hierarchical processing
        >>> g_inf = abstract_inference(g_gen, sigma_gen, p_list, ...)
    """
    p_list = []
    start_idx = 0
    for f in range(len(n_p)):
        end_idx = start_idx + n_p[f]
        p_list.append(p_flat[:, start_idx:end_idx])
        start_idx = end_idx
    return p_list


def compute_snr_db(signal: Tensor, noisy_signal: Tensor) -> float:
    """Compute signal-to-noise ratio in decibels.

    Calculates SNR as 10 * log10(signal_power / noise_power) where:
    - signal_power = mean(signal^2)
    - noise_power = mean((signal - noisy_signal)^2)

    Args:
        signal: Clean signal tensor of any shape
        noisy_signal: Noisy version of signal (same shape as signal)

    Returns:
        SNR in decibels (dB), or infinity if noise power is zero

    Example:
        >>> clean = torch.randn(10, 100)
        >>> noisy = clean + torch.randn(10, 100) * 0.1
        >>> snr = compute_snr_db(clean, noisy)
        >>> print(f"SNR: {snr:.2f} dB")
        SNR: 20.15 dB
    """
    signal_power = (signal**2).mean()
    noise_power = ((signal - noisy_signal) ** 2).mean()
    return 10 * torch.log10(signal_power / noise_power).item() if noise_power > 0 else float("inf")


def concatenate_frequencies(p_list: GroundedLocation) -> Vector:
    """Concatenate per-frequency place cell list into flat tensor.

    This utility function converts between the two common formats for grounded
    location representations in TEM:

    - Per-frequency format List[n_f] of [B, n_p[f]]: Used by hierarchical operations
      (GroundedLocInference, AbstractLocInference) that process each
      frequency module independently
    - Concatenated format [B, sum(n_p)]: Used by memory operations (AttractorDynamics,
      MemoryStorage) for efficient matrix multiplication with Hebbian matrices

    This conversion is needed when preparing per-frequency representations for
    memory storage or retrieval operations.

    Args:
        p_list: List of [n_f] tensors, each of shape [B, n_p[f]] representing
                place cell activity per frequency module

    Returns:
        Concatenated tensor of shape [B, sum(n_p)] where all frequency modules
        are stacked along dimension 1

    Example:
        >>> # After inference generates per-frequency place cells
        >>> p_list = grounded_inference(g_inf, x_f)  # List of [B,40], [B,32], [B,24]
        >>> # Convert to concatenated format for memory update
        >>> p_concat = concatenate_frequencies(p_list)  # [B, 96]
        >>> # Now ready for Hebbian update
        >>> storage.update(p_inferred=p_concat, p_generated=p_gen_concat, ...)
    """
    return torch.cat(p_list, dim=1)


def detect_grid_structure(adj: np.ndarray, n_locs: int) -> Optional[Tuple[int, int]]:
    """Detect if adjacency matrix represents a grid and return (width, height).

    Analyzes the graph structure to determine if it matches a rectangular grid
    topology where each node connects to its 4-neighbors (up, down, left, right).

    Args:
        adj: Adjacency matrix as numpy array
        n_locs: Number of locations (nodes) in the graph

    Returns:
        Tuple of (width, height) if grid detected, None otherwise

    Example:
        >>> adj = np.array([[0, 1, 1, 0],
        ...                 [1, 0, 0, 1],
        ...                 [1, 0, 0, 1],
        ...                 [0, 1, 1, 0]])
        >>> detect_grid_structure(adj, 4)
        (2, 2)
    """
    # Try common grid dimensions
    for width in range(2, int(np.sqrt(n_locs)) + 2):
        if n_locs % width == 0:
            height = n_locs // width

            # Check if adjacency matches grid pattern
            is_grid = True
            for loc_id in range(n_locs):
                i, j = loc_id // width, loc_id % width

                # Count expected neighbors
                expected_neighbors = []
                if i > 0:
                    expected_neighbors.append((i - 1) * width + j)  # up
                if i < height - 1:
                    expected_neighbors.append((i + 1) * width + j)  # down
                if j > 0:
                    expected_neighbors.append(i * width + (j - 1))  # left
                if j < width - 1:
                    expected_neighbors.append(i * width + (j + 1))  # right

                # Check actual neighbors match
                actual_neighbors = [k for k in range(n_locs) if adj[loc_id, k] > 0]
                if set(actual_neighbors) != set(expected_neighbors):
                    is_grid = False
                    break

            if is_grid:
                return (width, height)

    return None
