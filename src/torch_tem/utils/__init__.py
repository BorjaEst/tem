import datetime
import logging
import os
from itertools import combinations
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import comb
from torch import Tensor

from torch_tem.types import Matrix, Reduction, Transition, Vector


def sample_diag_gaussian(transition: Transition, *, scale: float = 1.0) -> List[Tensor]:
    """Sample a diagonal Gaussian distribution.
    If uncertainty is None, returns the mean directly.

    Args:
        transition: Transition object with mean and uncertainty.
        scale: Optional noise scale multiplier.

    Returns:
        Sampled activations if enabled, otherwise the means.
    """
    mu, sigma = transition.mean, transition.uncertainty
    return [gaussian(*args, scale=scale) for args in zip(mu, sigma)] if sigma else mu


def gaussian(mean: Tensor, std: Tensor, *, scale: float = 1.0) -> Tensor:
    # TODO: Probably there are built-in functions for this
    return mean + float(scale) * std * torch.randn_like(mean)


def inv_var_trans(base: Transition, corr: Transition, mask: Optional[Tensor] = None, freqs: Optional[range] = None) -> Transition:
    """Fuse a correction estimate into a base transition.

    Mean is always updated. Uncertainty is fused using inverse-variance
    weighting when available.

    Semantics for missing uncertainty:
    - If ``corr.uncertainty is None``: treat it as perfect precision (sigma = 0).
    - If ``base.uncertainty is None``: update mean only and keep uncertainty ``None``.

    Args:
        base: Base transition to correct.
        corr: Correction transition. If ``freqs`` is provided, the correction
            is assumed to be indexed relative to that frequency slice.
        mask: Optional boolean mask over batch items (shape ``(B,)``) selecting
            environments to update.
        freqs: Optional frequency indices in the base transition to update.

    Returns:
        Fused ``Transition``.
    """

    if mask is not None:
        if mask.ndim != 1:
            raise ValueError(f"mask must be 1D (B,), got shape={tuple(mask.shape)}")
        if mask.dtype is not torch.bool:
            raise TypeError(f"mask must be boolean, got dtype={mask.dtype}")

    freqs = range(len(base.mean)) if freqs is None else freqs
    if len(corr.mean) != len(freqs):
        raise ValueError(f"corr.mean length ({len(corr.mean)}) must match freqs length ({len(freqs)})")
    if corr.uncertainty is not None and len(corr.uncertainty) != len(freqs):
        raise ValueError(f"corr.uncertainty length ({len(corr.uncertainty)}) must match freqs length ({len(freqs)})")

    mu_out = list(base.mean)

    # Mean-only mode: update mean entries, keep uncertainty None (do_sample=False pathways)
    if base.uncertainty is None:
        for i, f in enumerate(freqs):
            base_mu = base.mean[f]
            corr_mu = corr.mean[i]

            if mask is None:
                mu_out[f] = corr_mu
                continue

            mu_f = base_mu.clone()
            mu_f[mask] = corr_mu
            mu_out[f] = mu_f

        return Transition(mean=mu_out, uncertainty=None)

    sigma_out = list(base.uncertainty)

    for i, f in enumerate(freqs):
        base_mu = base.mean[f]
        base_sigma = base.uncertainty[f]

        corr_mu = corr.mean[i]
        corr_sigma = corr.uncertainty[i] if corr.uncertainty is not None else None

        if mask is None:
            mu_new, sigma_new = inv_var_weight([base_mu, corr_mu], [base_sigma, corr_sigma])
            mu_out[f], sigma_out[f] = mu_new, sigma_new
            continue

        mu_sel, sigma_sel = inv_var_weight([base_mu[mask], corr_mu], [base_sigma[mask], corr_sigma])
        mu_f = base_mu.clone()
        sigma_f = base_sigma.clone()
        mu_f[mask] = mu_sel
        sigma_f[mask] = sigma_sel
        mu_out[f], sigma_out[f] = mu_f, sigma_f

    return Transition(mean=mu_out, uncertainty=sigma_out)


def inv_var_weight(mus: Sequence[Tensor], sigmas: Sequence[Tensor | None], *, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Inverse-variance (precision) weighted fusion of diagonal Gaussians.

    This fuses multiple independent Gaussian estimates with diagonal covariance.

    Missing sigma is treated as perfect precision (sigma == 0), meaning the
    corresponding mean dominates and the fused sigma becomes exactly zero for
    those elements.

    Args:
        mus: Sequence of mean tensors, all same shape.
        sigmas: Sequence of std-dev tensors (same shapes as mus) or ``None``.
        eps: Small value to avoid division-by-zero for non-perfect entries.

    Returns:
        (mu, sigma): precision-weighted mean and std-dev.
    """
    if len(mus) == 0:
        raise ValueError("mus must be non-empty")
    if len(mus) != len(sigmas):
        raise ValueError(f"mus and sigmas must have same length, got {len(mus)} and {len(sigmas)}")

    sigma_tensors = [torch.zeros_like(mu) if sigma is None else sigma for mu, sigma in zip(mus, sigmas)]

    mu_stack = torch.stack(list(mus), dim=0)
    sigma_stack = torch.stack(sigma_tensors, dim=0)

    # Compute precision = 1 / var (with safe epsilon for non-perfect values)
    var = torch.square(sigma_stack).clamp_min(eps)
    precision = torch.reciprocal(var)

    precision_sum = precision.sum(dim=0)
    mu_out = (mu_stack * precision).sum(dim=0) / precision_sum.clamp_min(eps)
    sigma_out = torch.sqrt(torch.reciprocal(precision_sum.clamp_min(eps)))

    # If any source has perfect precision (sigma == 0), the fused result should be exact.
    perfect = sigma_stack == 0
    perfect_any = perfect.any(dim=0)
    if perfect_any.any():
        weights = perfect.to(dtype=mu_out.dtype)
        denom = weights.sum(dim=0).clamp_min(1.0)
        mu_perfect = (mu_stack * weights).sum(dim=0) / denom
        mu_out = torch.where(perfect_any, mu_perfect, mu_out)
        sigma_out = torch.where(perfect_any, torch.zeros_like(sigma_out), sigma_out)

    return mu_out, sigma_out


def softmax(o):
    """
    Applies softmax to tensors of inputs, using torch softmax funcion
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Return torch softmax
    return torch.nn.Softmax(dim=-1)(o)


def normalise(o):
    """
    Normalises vector of input to unit norm, using torch normalise funcion
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Return torch normalise with p=2 for L2 norm
    return torch.nn.functional.normalize(o, p=2, dim=-1)


def relu(o):
    """
    Applies rectified linear activation unit to tensors of inputs, using torch relu funcion
    """
    # Return torch relu
    return torch.nn.functional.relu(o)


def leaky_relu(o):
    """
    Applies leaky (meaning small negative slope instead of zeros) rectified linear activation unit to tensors of inputs, using torch leaky relu funcion
    """
    # Return torch leaky relu [torch.nn.functional.leaky_relu(val) for val in o] if type(o) is list else
    return torch.nn.functional.leaky_relu(o)


def activation_from_str(name: str):
    """Return activation function from string name."""
    if name == "none":
        return lambda x: x
    if name == "relu":
        return torch.nn.functional.relu
    name = name.lower()
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    if name == "sigmoid":
        return torch.sigmoid
    if name == "softmax":
        return torch.nn.Softmax(dim=-1)
    if name == "tanh":
        return torch.tanh
    if name == "exp":
        return torch.exp
    raise ValueError(f"Unknown activation function: {name}")


def squared_error(value, target):
    """
    Calculates mean squared error (L2 norm) between (list of) tensors value and target by using torch MSE loss
    Include a factor 0.5 to squared error by convention
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    """
    # Return torch MSE loss
    if type(value) is list:
        loss = [0.5 * torch.sum(torch.nn.MSELoss(reduction="none")(value[i], target[i]), dim=-1) for i in range(len(value))]
    else:
        loss = 0.5 * torch.sum(torch.nn.MSELoss(reduction="none")(value, target), dim=-1)
    return loss


def cross_entropy(value, target):
    """
    Calculates binary cross entropy between tensors value and target by using torch cross entropy loss
    Set reduction to none, then get mean over last dimension to keep losses of different batches separate
    """
    # Return torch BCE loss
    if type(value) is list:
        loss = [torch.nn.CrossEntropyLoss(reduction="none")(val, targ) for val, targ in zip(value, target)]
    else:
        loss = torch.nn.CrossEntropyLoss(reduction="none")(value, target)
    return loss


def downsample(value, target_dim):
    """
    Does downsampling by taking the an input vector, then averaging chunks to make it of requested dimension
    Assumes o is a 1D vector, or batches of row vectors with the batches along dim 0
    """
    # Get input dimension
    value_dim = value.size()[-1]
    # Set places to break up input vector into chunks
    edges = np.append(np.round(np.arange(0, value_dim, float(value_dim) / target_dim)), value_dim).astype(int)
    # Create downsampling matrix
    downsample = torch.zeros((value_dim, target_dim), dtype=torch.float)
    # Fill downsampling matrix with chunks
    for curr_entry in range(target_dim):
        downsample[edges[curr_entry] : edges[curr_entry + 1], curr_entry] = torch.tensor(1.0 / (edges[curr_entry + 1] - edges[curr_entry]), dtype=torch.float)
    # Do downsampling by matrix multiplication
    return torch.matmul(value, downsample)


def make_directories():
    """
    Creates directories for storing data during a model training run
    """
    # Get current date for saving folder
    date = datetime.datetime.today().strftime("%Y-%m-%d")
    # Initialise the run and dir_check to create a new run folder within the current date
    run = 0
    dir_check = True
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    while dir_check:
        # Construct new paths
        run_path = "../Summaries/" + date + "/run" + str(run) + "/"
        train_path = run_path + "train"
        model_path = run_path + "model"
        save_path = run_path + "save"
        script_path = run_path + "script"
        envs_path = script_path + "/envs"
        run += 1
        # And once a path doesn't exist yet: create new folders
        if not os.path.exists(train_path) and not os.path.exists(model_path) and not os.path.exists(save_path):
            os.makedirs(train_path)
            os.makedirs(model_path)
            os.makedirs(save_path)
            os.makedirs(script_path)
            os.makedirs(envs_path)
            dir_check = False
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def set_directories(date, run):
    """
    Returns directories for storing data during a model training run from a given previous training run
    """
    # Initialise all pahts
    train_path, model_path, save_path, script_path, run_path = None, None, None, None, None
    # Find the current run: the first run that doesn't exist yet
    run_path = "../Summaries/" + date + "/run" + str(run) + "/"
    train_path = run_path + "train"
    model_path = run_path + "model"
    save_path = run_path + "save"
    script_path = run_path + "script"
    envs_path = script_path + "/envs"
    # Return folders to new path
    return run_path, train_path, model_path, save_path, script_path, envs_path


def make_logger(run_path):
    """
    Creates logger so output during training can be stored to file in a consistent way
    """
    # Create new logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Remove anly existing handlers so you don't output to old files, or to new files twice
    logger.handlers = []
    # Create a file handler, but only if the handler does
    handler = logging.FileHandler(run_path + "report.log")
    handler.setLevel(logging.INFO)
    # Create a logging format
    formatter = logging.Formatter("%(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(handler)
    # Return the logger object
    return logger


def as_dir_str(path: Path) -> str:
    """Return a directory path as a string with trailing separator for legacy utils."""
    s = str(path)
    return s if s.endswith(os.sep) else s + os.sep


def resolve_envs_path(run_path: Path) -> Path:
    """Find envs directory for a run (prefer script/envs, fall back to envs)."""
    candidate = run_path / "script" / "envs"
    if candidate.exists():
        return candidate
    candidate = run_path / "envs"
    return candidate


def parse_iter_from_stem(stem: str) -> Optional[int]:
    """Parse iteration number from checkpoint stem like 'tem_4000' or 'params_4000'"""
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def apply_overrides(params: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return params with override applied (shallow merge)."""
    if not override:
        return params
    for k, v in override.items():
        params[k] = v
    return params


def require_exists(path: Path, what: str) -> None:
    """Raise friendly error if path doesn't exist."""
    if not path.exists():
        raise FileNotFoundError(f"{what} not found: {path}")


def reduce_per_env(loss_per_env: Tensor, reduction: Reduction) -> Tensor:
    """Reduce a per-environment loss vector according to `reduction`."""
    if reduction == "sum":
        return loss_per_env.sum()
    if reduction == "mean":
        return loss_per_env.mean()
    if reduction == "none":
        return loss_per_env
    raise ValueError(f"Unknown reduction: {reduction}")


def create_downsample_matrix(n: List[int], n_subsampled: List[int]) -> List[Matrix]:
    """Create downsampling matrices.

    Downsampling matrix to go from cells to compressed cells for
    indexing memories by simply taking only the first n_subsampled cells.

    Args:
        n: Full input dimensions per frequency
        n_subsampled: Subsampled output dimensions per frequency

    Returns:
        List of downsampling matrices, one per frequency module
    """
    # Matrix shape: [n_in, n_out] where we select first n_out columns
    # For input o: [B, n_in], result is o @ W_down = [B, n_out]
    return [torch.cat([torch.eye(dim_out, dtype=torch.float), torch.zeros((dim_in - dim_out, dim_out), dtype=torch.float)]) for dim_in, dim_out in zip(n, n_subsampled)]


def create_repeat_matrices(n_subsampled: List[int], n: List[int]) -> List[Matrix]:
    """Create repeat matrices.

    Matrix for repeating cells information using elementwise product
    after matrix multiplication.

    Args:
        n_subsampled: Subsampled input dimensions per frequency
        n: Full output dimensions per frequency

    Returns:
        List of repeat matrices, one per frequency module
    """
    # Matrix shape: [n_subsampled, n_p] where each row is repeated
    # For input g: [B, n_subsampled], result is g @ W_repeat = [B, n_p]
    # Uses Kronecker product: eye(n_subsampled) ⊗ ones(1, n_p/n_subsampled)
    return [torch.tensor(np.kron(np.eye(dim_in), np.ones((1, dim_out // dim_in))), dtype=torch.float) for dim_in, dim_out in zip(n_subsampled, n)]


def create_tiling_matrices(n_in: List[int], n_out: List[int]) -> List[Matrix]:
    """Create tile matrices.

    Tiling matrix to project from one cortical region to another by repeating
    the input representation multiple times.

    Args:
        n_in: Input dimensions per frequency module
        n_out: Output dimensions per frequency module

    Returns:
        List of tile matrices, one per frequency module.
    """
    # Matrix shape: [n_in, n_out] where each input is tiled
    # For input o_c: [B, n_in], result is o_c @ W_tile = [B, n_out]
    # Uses Kronecker product: ones(1, n_tiles) ⊗ eye(n_in)
    # where n_tiles = n_out / n_in

    # Validate divisibility
    if any(out % inp != 0 for out, inp in zip(n_out, n_in)):
        raise ValueError(f"n_out must be divisible by n_in. Got n_out={n_out}, n_in={n_in}")

    return [torch.tensor(np.kron(np.ones((1, out // inp)), np.eye(inp)), dtype=torch.float) for inp, out in zip(n_in, n_out)]


def create_random_projection(n_in: List[int], n_out: List[int], sparsity: float = 1.0, seed: Optional[int] = None) -> List[Matrix]:
    """Create random fixed projection matrices.

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
        n_in: Input dimensions per frequency module
        n_out: Output dimensions per frequency module
        sparsity: Connection probability (1.0 = fully connected, 0.1 = 10% connectivity)
        seed: Random seed for reproducibility (optional)

    Returns:
        List of random projection matrices [n_g[f], n_p[f]], one per frequency module

    Example:
        >>> n_g = [36, 30, 24]  # Grid cell dimensions
        >>> n_p = [96, 80, 64]  # Place cell dimensions
        >>> W_random = create_random_projection(n_g, n_p, sparsity=0.15)
        >>> # Use in projection head:
        >>> g_ = [g[f] @ W_random[f] for f in range(n_f)]

    References:
        Chandra et al. (2025). "Episodic and associative memory from spatial
        scaffolds in the hippocampus." CAN model architecture.
    """
    if seed is not None:
        torch.manual_seed(seed)

    matrices = []
    for g_dim, p_dim in zip(n_in, n_out):
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


def create_encoding_table(n_in: int, n_out: int, n_hot: int = 2) -> List[Vector]:
    """Create n-hot encoding lookup table.

    Generates a lookup table for converting one-hot observations to n-hot
    compressed representations. Each observation is encoded using exactly
    `n_hot` active units from `n_out` dimensions.

    Args:
        n_in: Number of possible observations (must be <= C(n_out, n_hot))
        n_out: Compressed sensory dimension
        n_hot: Number of active units per code (1, 2, 3, etc.)
            - n_hot=1: One-hot (identity, no compression unless n_in > n_out)
            - n_hot=2: Two-hot (default, typically 45 → 10)
            - n_hot=3: Three-hot (more distributed, e.g., 220 → 12)

    Returns:
        List of n-hot code tensors, one per possible observation [n_in, n_out]

    Raises:
        ValueError: If n_in > C(n_out, n_hot) (too many observations for compression)

    Example:
        >>> # Two-hot encoding: 45 observations → 10 dimensions
        >>> table = create_encoding_table(n_in=45, n_out=10, n_hot=2)
        >>> len(table)
        45
        >>> table[0].sum()
        2.0

        >>> # Three-hot encoding: 220 observations → 12 dimensions
        >>> table = create_encoding_table(n_in=220, n_out=12, n_hot=3)
        >>> len(table)
        220
        >>> table[0].sum()
        3.0
    """

    # Validate: number of observations must not exceed possible n-hot codes
    max_codes = int(comb(n_out, n_hot))
    if n_in > max_codes:
        raise ValueError(f"Cannot encode {n_in} observations with {n_hot}-hot codes in {n_out} dimensions. " f"Maximum possible codes: C({n_out}, {n_hot}) = {max_codes}")

    # Generate all possible n-hot codes using combinations
    # combinations(range(n_out), n_hot) gives all ways to choose n_hot positions
    encoding_table = []
    for active_positions in combinations(range(n_out), n_hot):
        # Create zero vector
        code = [0] * n_out
        # Activate n_hot positions
        for pos in active_positions:
            code[pos] = 1
        # Add to table
        encoding_table.append(torch.tensor(code, dtype=torch.float))

        # Stop when we have enough codes for all observations
        if len(encoding_table) >= n_in:
            break

    return torch.stack(encoding_table, dim=0)


def uncat_to_list(x: Tensor, dims: List[int]) -> List[Tensor]:
    """Split a concatenated tensor into a list of tensors with given last-dim sizes.

    Args:
        x: Tensor shaped (B, sum(dims)).
        dims: List of segment sizes.

    Returns:
        List of tensors [x0, x1, ...] where xi.shape == (B, dims[i]).
    """
    return list(torch.split(x, dims, dim=1))


def one_hot_with_zero(action: list[int | None], num_actions: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Convert actions to one-hot encoding where action 0/None = all-zeros (static action).

    action: list of int or None
        0|None = static / no-op -> all zeros
        1..num_actions = discrete actions -> one-hot at index (action - 1)
    returns: FloatTensor, shape (len(action), num_actions)
    """
    action_t = torch.tensor([a if a is not None else 0 for a in action], dtype=torch.long, device=device)
    mask = action_t > 0
    out = torch.zeros((len(action), num_actions), dtype=torch.float32, device=device)
    if mask.any():
        out[mask] = F.one_hot(action_t[mask] - 1, num_classes=num_actions).float()
    return out


def connections(f_grid: list[float]) -> list[list[bool]]:
    """Compute hierarchical connection matrix from frequency list.

    Entry [f_to][f_from] is True if f_from connects to f_to.
    Connection rule: lower-frequency modules connect to higher-frequency.

    Args:
        f_grid: Frequency values per module (higher value = higher frequency)

    Returns:
        Connection matrix where [f_to][f_from] indicates connectivity
    """
    n = len(f_grid)
    return [[f_grid[f1] <= f_grid[f2] for f1 in range(n)] for f2 in range(n)]


def resolve_ovc_slice(n_freq_total: int, n_freq_ovc: Optional[int]) -> Tuple[int, int]:
    """Determine which MEC modules are OVC (receive shiny landmark correction).

    Args:
        n_freq_total: Total number of MEC modules
        n_freq_ovc: User setting for OVC module count:
            - None: all modules are OVC (legacy separate_ovc=False)
            - 0: no OVC modules (disable shiny correction)
            - k>0: last k modules are OVC (legacy separate_ovc=True, n_f_ovc=k)

    Returns:
        (ovc_start, ovc_count): slice range [ovc_start:ovc_start+ovc_count] for OVC modules

    Examples:
        >>> resolve_ovc_slice(5, None)  # All modules are OVC
        (0, 5)
        >>> resolve_ovc_slice(5, 0)     # No OVC modules
        (5, 0)
        >>> resolve_ovc_slice(5, 2)     # Last 2 modules are OVC
        (3, 2)
    """
    if n_freq_ovc is None:
        # Apply to all modules (legacy separate_ovc=False)
        return 0, n_freq_total

    n_freq_ovc = int(n_freq_ovc)
    if n_freq_ovc < 0 or n_freq_ovc > n_freq_total:
        raise ValueError(f"MECSettings.n_freq_ovc must be in [0, {n_freq_total}] or None; got {n_freq_ovc}")

    if n_freq_ovc == 0:
        # No OVC correction
        return n_freq_total, 0

    # Apply to last k modules (legacy separate_ovc=True)
    return n_freq_total - n_freq_ovc, n_freq_ovc


def update_to_masks(shape: List[int], *, update: torch.Tensor) -> torch.Tensor:
    """Expand a stage×freq update matrix to a stage×sum(shape) mask tensor.

    Args:
        shape: Feature dims per frequency module.
        update: Bool/0-1 tensor of shape (n_stages, n_freq).

    Returns:
        masks: Float tensor of shape (n_stages, sum(hpc_shape)).
    """
    n_freq = len(shape)
    if update.ndim != 2 or update.shape[1] != n_freq:
        raise ValueError(f"Expected update shape (n_stages, {n_freq}), got {tuple(update.shape)}")

    widths = torch.tensor(shape, device=update.device)
    masks = update.to(dtype=torch.float).repeat_interleave(widths, dim=1)
    return masks


def make_update_full(n_stages: int, n_freq: int, *, device=None) -> torch.Tensor:
    """Full update matrix (all True).

    Args:
        n_stages: Number of attractor stages.
        n_freq: Number of frequency modules.

    Returns:
        Full update matrix of shape (n_stages, n_freq).
    """
    return torch.ones((n_stages, n_freq), dtype=torch.bool, device=device)


def make_update_hierarchical(n_stages: int, n_freq: int, *, ramp_len: int | None = None, device=None) -> torch.Tensor:
    """Hierarchical update matrix.

    ramp_len controls how many frequency modules participate in the triangular ramp.
    If None, uses min(n_stages, n_freq).
    """
    ramp_len = min(n_stages, n_freq) if ramp_len is None else ramp_len
    if not (0 <= ramp_len <= n_freq):
        raise ValueError("ramp_len must be in [0, n_freq]")

    i = torch.arange(n_stages, device=device)[:, None]  # (n_stages, 1)
    f = torch.arange(ramp_len, device=device)[None, :]  # (1, ramp_len)

    # Triangular ramp: freq 0 updates n_stages times, freq 1 updates n_stages-1, etc.
    update_ramp = i < (n_stages - f)  # (n_stages, ramp_len)

    if ramp_len < n_freq:
        update_tail = torch.ones((n_stages, n_freq - ramp_len), dtype=torch.bool, device=device)
        return torch.cat([update_ramp, update_tail], dim=1)

    return update_ramp
