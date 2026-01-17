from __future__ import annotations

from typing import List

import torch
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.settings import FeatureNormSettings


class FeatureNorm(nn.Module):
    def __init__(self, settings: FeatureNormSettings):
        super().__init__()
        self._settings = settings

    @property
    def settings(self) -> FeatureNormSettings:
        return self._settings

    def forward(self, x: List[Tensor]) -> List[Tensor]:
        n_freq = len(x)
        positive_centered = [utils.relu(x[f] - torch.mean(x[f])) for f in range(n_freq)]
        normalised = [utils.normalise(positive_centered[f]) for f in range(n_freq)]
        return normalised
