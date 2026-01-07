from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import LECSettings


@dataclass
class LECState:
    """State container for TEM model components."""

    c: Tensor  # Compressed observation
    x: List[Tensor]  # Multi-frequency filtered features
    x_filtered: List[Tensor]  # Unweighted filtered features


class LECModel(nn.Module):
    """Lateral Entorhinal Cortex: Temporal filtering of compressed sensory features.

    Responsibility: Apply multi-frequency exponential smoothing to compressed features.
    Input: c (compressed observations from autoencoder)
    Output: x (multi-frequency filtered features)

    External dependencies:
    - Autoencoder (o -> c encoding) is handled externally
    - Projection (x -> x_ for memory) is handled externally
    """

    def __init__(self, n_c: int, n_x: List[int], settings: LECSettings, f_init: Optional[List[float]] = None):
        super().__init__()

        # Store hyperparameters
        self._n_c = n_c
        self._n_x = n_x
        self.settings = settings

        # Initialize temporal filtering factors
        # Store as logit(f) so that sigmoid(alpha) recovers the desired frequency
        alpha_freq = f_init if f_init is not None else _alpha_init(settings, len(n_x))
        alpha_logit = [np.log(f / (1 - f)) for f in alpha_freq]
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(a, dtype=torch.float)) for a in alpha_logit])

        # Frequency module specific scaling of filtered sensory experience
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(len(n_x))])  # w_p in legacy

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(self._n_c))  # Bias for reconstructing c from x

    @property
    def n_in(self) -> int:
        """Dimensionality of compressed features."""
        return self._n_c

    @property
    def n_out(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._n_x

    def x_prev2x(self, c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        # Calculate factor for filtering from sigmoid of learned parameter
        alpha = [torch.sigmoid(self.alpha[f]) for f, _ in enumerate(self.n_out)]
        # Do exponential temporal filtering for each frequency module
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * c for f, _ in enumerate(self.n_out)]
        return x

    def f_n(self, x: List[Tensor]) -> List[Tensor]:
        # Normalize using global mean across entire batch (legacy behavior)
        normalised = [utils.normalise(utils.relu(x[f] - torch.mean(x[f]))) for f, _ in enumerate(self.n_out)]
        return normalised

    def f_w(self, x: List[Tensor]) -> List[Tensor]:
        # Apply sigmoid-constrained scaling like legacy
        weighted = [torch.sigmoid(self.w_f[f]) * x[f] for f, _ in enumerate(self.n_out)]
        return weighted

    def forward(self, c: Tensor, state: LECState) -> LECState:
        # Temporally filter sensory observation by mixing it with previous experience
        x_filtered = self.x_prev2x(c, state.x_filtered)
        # Normalize and weight filtered sensory experience for memory
        x_normalized = self.f_n(x_filtered)
        x = self.f_w(x_normalized)
        return LECState(c=c, x=x, x_filtered=x_filtered)

    def reconstruct(self, x: List[Tensor]) -> Tensor:
        """Reconstruct compressed features from filtered features.

        Applies affine transformation to approximate compressed features from
        the highest-frequency filtered features. Used in generative model (p -> x -> c -> o).

        Legacy equivalent: w_x * x + b_x

        Note: Uses only the first (highest-frequency, most responsive) module.
        This is an approximation that does NOT invert normalization/weighting.
        """
        # Use highest frequency module (most responsive to current input)
        # This matches legacy behavior of using x[0] for reconstruction
        return self.w_x * x[0] + self.b_x


def _alpha_init(settings: LECSettings, n_f: int) -> List[float]:
    """Initialize temporal filtering factors based on desired time constants.

    Returns frequencies in [0, 1] range (NOT logit-transformed).
    The calling code will apply the logit transform.
    """
    if settings.frequencies_init == "linear":  # Linearly spaced time constants between min and max
        return np.linspace(0.9, 0.1, n_f).tolist()
    raise ValueError(f"Unknown frequencies_init method: {settings.frequencies_init}")
