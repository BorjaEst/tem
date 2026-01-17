from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.core.lec.filter import FrequencyFilter
from torch_tem.core.lec.norm import FeatureNorm
from torch_tem.modules import MLP
from torch_tem.settings import LECSettings


@dataclass
class LECState:
    cells: List[Tensor]  # LEC cell activations per frequency
    filtered: List[Tensor]  # Unweighted filtered features

    def new(self, **kwargs) -> "LECState":
        copy = self.__dict__.copy()
        copy.update(kwargs)
        return LECState(**copy)

    def detach(self) -> "LECState":
        return LECState(
            cells=[v.detach() for v in self.cells],
            filtered=[v.detach() for v in self.filtered],
        )


class LECModel(nn.Module):
    def __init__(self, n_c: int, shape: List[int], f_init: List[float], settings: LECSettings):
        super().__init__()
        self._n_c = n_c
        self._shape, self._n_freq = shape, len(shape)
        self._settings = settings

        # Composable submodules (single responsibility each)
        self.filter = FrequencyFilter(shape, f_init, settings.filter)
        self.norm = FeatureNorm(shape, settings.norm)
        self.reconstructor = None  # Reconstruction module

        # Frequency module specific scaling of filtered sensory experience
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_freq)])

        # Reconstruction parameters
        self.w_x = torch.nn.Parameter(torch.tensor(1.0))  # For reconstructing c from x
        self.b_x = torch.nn.Parameter(torch.zeros(self._n_c))  # Bias for reconstructing c from x

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> LECState:
        x0 = [torch.zeros((batch_size, n), device=device) for n in self.shape]
        return LECState(cells=x0, filtered=x0)

    def set_runtime(self, *, _):
        pass

    @property
    def settings(self) -> LECSettings:
        return self._settings

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def n_freq(self) -> int:
        return len(self._shape)

    def forward(self, *, _) -> Tuple[List[Tensor], LECState]:
        raise NotImplementedError("LEC forward not implemented. Use inference().")

    def generative(self, *, _) -> Tuple[List[Tensor], LECState]:
        raise NotImplementedError("LEC does not support geenrative. Use inference().")

    def inference(self, c: Tensor, state: LECState) -> Tuple[List[Tensor], LECState]:
        filtered = self.filter(c, state.filtered)
        normalized = self.norm(filtered)
        x_inf = next_cells = self.f_w(normalized)
        return x_inf, state.new(cells=next_cells, filtered=filtered)

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
