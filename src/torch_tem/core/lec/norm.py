from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import FeatureNormSettings


class FeatureNorm(nn.Module):
    def __init__(self, lec_shape: List[int], settings: FeatureNormSettings):
        super().__init__()
        self._n_freq = len(lec_shape)
        self._settings = settings

    @property
    def settings(self) -> FeatureNormSettings:
        return self._settings

    @property
    def n_freq(self) -> int:
        return self._n_freq

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        positive_centered = [utils.relu(x[f] - torch.mean(x[f])) for f in range(self.n_freq)]
        normalised = [utils.normalise(positive_centered[f]) for f in range(self.n_freq)]
        return normalised
