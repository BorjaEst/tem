from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from torch_tem.settings import FreqFilterSettings


class FrequencyFilter(nn.Module):
    def __init__(self, lec_shape: List[int], f_init: List[float], settings: FreqFilterSettings):
        super().__init__()
        self._n_freq = len(lec_shape)
        self._settings = settings

        # Initialize temporal filtering factors
        # Store as logit(f) so that sigmoid(alpha) recovers the desired frequency
        alpha_logit = [np.log(f / (1 - f)) for f in f_init]
        self.alpha = nn.ParameterList([nn.Parameter(torch.tensor(a, dtype=torch.float)) for a in alpha_logit])

    @property
    def settings(self) -> FreqFilterSettings:
        return self._settings

    @property
    def n_freq(self) -> int:
        """Return the number of frequency modules."""
        return len(self.alpha)

    def forward(self, c: Tensor, x_prev: List[Tensor]) -> List[Tensor]:
        # Calculate factor for filtering from sigmoid of learned parameter
        alpha = [torch.sigmoid(self.alpha[f]) for f in range(self.n_freq)]
        # Do exponential temporal filtering for each frequency module
        x = [(1 - alpha[f]) * x_prev[f] + alpha[f] * c for f in range(self.n_freq)]
        return x
