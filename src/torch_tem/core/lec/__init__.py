from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

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

    def __init__(self, n_c: int, shape: List[int], f_init: List[float], settings: LECSettings):
        super().__init__()

        # Store hyperparameters
        self._n_c = n_c
        self._n_x = shape
        self.settings = settings

        # Initialize temporal filtering factors
        # Store as logit(f) so that sigmoid(alpha) recovers the desired frequency
        alpha_logit = [np.log(f / (1 - f)) for f in f_init]
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(a, dtype=torch.float)) for a in alpha_logit])

        # Frequency module specific scaling of filtered sensory experience
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_freq)])

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(self._n_c))  # Bias for reconstructing c from x

    def init_state(self, batch_size: int, device: torch.device) -> LECState:
        """Initialize LEC state with zeros."""
        x0 = [torch.zeros((batch_size, n), device=device) for n in self._n_x]
        return LECState(x=x0, x_filtered=x0)

    @property
    def n_in(self) -> int:
        """Dimensionality of compressed features."""
        return self._n_c

    @property
    def shape(self) -> List[int]:
        """Dimensionality of features per frequency module."""
        return self._n_x

    @property
    def n_freq(self) -> int:
        """Number of frequency modules."""
        return len(self._n_x)

    def forward(self, c: Tensor, state: LECState) -> Tuple[List[Tensor], LECState]:
        # Delegate to inference method
        return self.inference(c, state)

    def inference(self, c: Tensor, state: LECState) -> Tuple[List[Tensor], LECState]:
        # Temporally filter sensory observation by mixing it with previous experience
        x_filtered = self.x_prev2x(c, state.x_filtered)
        # Normalize and weight filtered sensory experience for memory
        x_normalized = self.f_n(x_filtered)
        x = self.f_w(x_normalized)
        return x, LECState(x=x, x_filtered=x_filtered)

    def x_prev2x(self, c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        # Calculate factor for filtering from sigmoid of learned parameter
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.n_freq)]
        # Do exponential temporal filtering for each frequency module
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * c for f in range(self.n_freq)]
        return x

    def f_n(self, x: List[Tensor]) -> List[Tensor]:
        # Normalize per sample: subtract mean along feature dimension (dim=1), then L2-normalize
        # This keeps each environment's representation independent
        normalised = [utils.normalise(utils.relu(x[f] - torch.mean(x[f], dim=1, keepdim=True))) for f in range(self.n_freq)]
        return normalised

    def f_w(self, x: List[Tensor]) -> List[Tensor]:
        # Apply sigmoid-constrained scaling like legacy
        weighted = [torch.sigmoid(self.w_f[f]) * x[f] for f in range(self.n_freq)]
        return weighted

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
