from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.core.lec.filter import FrequencyFilter
from torch_tem.core.lec.norm import FeatureNorm
from torch_tem.core.lec.reconstruction import Reconstruction
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
    def __init__(self, n_c: int, f_init: List[float], settings: LECSettings):
        super().__init__()
        self._n_c, self._n_freq = n_c, len(f_init)
        self._shape = [n_c] * self.n_freq
        self._settings = settings

        # Composable submodules (single responsibility each)
        self.filter = FrequencyFilter(f_init, settings.filter)
        self.norm = FeatureNorm(settings.norm)
        self.reconstructor = Reconstruction(n_c, settings.reconstruction)

        # Frequency module specific scaling of filtered sensory experience
        self.w_f = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_freq)])

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
        return self._n_freq

    def forward(self, *, _) -> Tuple[List[Tensor], LECState]:
        raise NotImplementedError("LEC forward not implemented. Use inference().")

    def generative(self, x: List[Tensor]) -> Tensor:
        return self.reconstructor(x)

    def inference(self, c: Tensor, state: LECState) -> Tuple[List[Tensor], LECState]:
        filtered = self.filter(c, state.filtered)
        normalized = self.norm(filtered)
        x_inf = next_cells = [torch.sigmoid(self.w_f[f]) * normalized[f] for f in range(self.n_freq)]
        return x_inf, state.new(cells=next_cells, filtered=filtered)
