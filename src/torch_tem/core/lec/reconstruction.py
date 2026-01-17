from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from torch_tem.settings import ReconstructionSettings


class Reconstruction(nn.Module):
    def __init__(self, n_c: int, settings: ReconstructionSettings):
        super().__init__()
        self._settings = settings

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(n_c))  # Bias for reconstructing c from x

    @property
    def settings(self) -> ReconstructionSettings:
        return self._settings

    def forward(self, x: List[Tensor]) -> Tensor:
        return self.w_x * x[0] + self.b_x
