""" """

from typing import List, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from torch_tem.types import MultiScaleCode


class ProcessorParams(Protocol):
    n_f: int
    f_extended: List[float]


class Processor(nn.Module):
    def __init__(self, params: ProcessorParams):
        super().__init__()
        self.n_f = params.n_f

        # Learnable decay rates (inverse sigmoid)
        self.f_tensor = torch.tensor(params.f_extended, dtype=torch.float)
        logits = torch.logit(self.f_tensor)
        self.alpha_logit = nn.ParameterList([nn.Parameter(logits[i : i + 1]) for i in range(self.n_f)])

    def filter_temporal(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        alpha = [torch.sigmoid(self.alpha_logit[f]) for f in range(self.n_f)]
        return [alpha[f] * x_c + (1 - alpha[f]) * x_prev[f] for f in range(self.n_f)]

    def normalize(self, x_f: MultiScaleCode) -> MultiScaleCode:
        return [torch.nn.functional.normalize(torch.relu(x - x.mean(dim=-1, keepdim=True)), p=2, dim=-1) for x in x_f]

    def forward(self, x_c: Tensor, x_prev: MultiScaleCode) -> MultiScaleCode:
        x_f = self.filter_temporal(x_c, x_prev)  # Exponential smoothing for each frequency channel
        x_normalized = self.normalize(x_f)  # Per-channel L2 normalization with learnable affine transform
        return x_normalized
