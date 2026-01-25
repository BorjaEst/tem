from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from itertools import islice
from typing import Any, Generic, Iterable, List, Optional, TypeVar

import numpy as np
from torch import Tensor

from torch_tem.data.datamodule import DataModule, DataStep
from torch_tem.data.rollout import RolloutStep, SimulationStep
from torch_tem.data.world import World
from torch_tem.model import Model as TEMModel
from torch_tem.model import Prediction, RolloutStream, TEMAction, TEMGenerative, TEMInference, TEMLabel, TEMOutput, TEMReconstruction, TEMState, TEMStep
from torch_tem.modules.hpc import HPCState
from torch_tem.modules.lec import LECState
from torch_tem.modules.mec import MECState
from torch_tem.types import AbstractLocation, GroundedLocation, LocationLabel, Matrix, MultiScaleCode, Observation, Walk
from torch_tem.utils.walks import downsample_env_major, time_major_location_ids, time_major_to_env_major_walks

TStep = TypeVar("TStep")
TraceT = TypeVar("TraceT", bound="TraceBase")


@dataclass
class TraceBase(Sequence[TStep], Generic[TStep], ABC):
    meta: dict[str, Any] = field(default_factory=dict)
    _n_steps: int = field(default=0)

    def __getitem__(self, idx: int | slice) -> TStep | TraceBase[TStep]:
        if isinstance(idx, int):
            return self.get_item(idx)
        out = self.__class__(meta=self.meta.copy())
        return out.attach(list(self)[idx])

    @abstractmethod
    def get_item(self, idx: int) -> TStep:
        raise NotImplementedError

    def attach(self, steps: Iterable[TStep]) -> TraceBase[TStep]:
        for step in steps:
            self.append(step)
        return self

    def append(self, step: TStep) -> None:
        self._append(step)
        self._n_steps += 1

    @abstractmethod
    def _append(self, step: TStep) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        return self._n_steps

    @abstractmethod
    @property
    def batch_size(self) -> int:
        raise NotImplementedError

    def downsample_time(self, stride: int) -> "TraceBase":
        iterable = (x for i, x in enumerate(self) if i % stride == 0)
        return self.__class__.from_iter(iterable, stop=None, meta=self.meta.copy())

    @classmethod
    def from_iter(cls, iterable: Iterable[TStep], *, stop: int | None = None, meta: dict[str, Any] | None = None) -> TraceT:
        trace = cls(meta=(meta or {}))
        return trace.attach(islice(iterable, stop))


@dataclass
class TEMLabelTrace(TraceBase[TEMLabel]):
    observation: list[Observation] = field(default_factory=list)
    locations: list[LocationLabel] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.observation[0].shape[0]) if self.observation else 0

    def get_item(self, idx: int) -> TStep:
        return TEMLabel(self.observation[idx], self.locations)

    def _append(self, step: TEMLabel) -> None:
        self.observation.append(step.observation)
        self.locations = step.locations


@dataclass
class TEMInferenceTrace(TraceBase[TEMInference]):
    g_inf: list[AbstractLocation] = field(default_factory=list)
    p_inf: list[GroundedLocation] = field(default_factory=list)
    p_xi: list[GroundedLocation] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.g_inf[0].shape[0]) if self.g_inf else 0

    def get_item(self, idx: int) -> TStep:
        return TEMInference(self.g_inf[idx], self.p_inf[idx], self.p_xi[idx])

    def _append(self, step: TEMInference) -> None:
        self.g_inf.append(step.g_inf)
        self.p_inf.append(step.p_inf)
        self.p_xi.append(step.p_xi)


@dataclass
class TEMGenerativeTrace(TraceBase[TEMGenerative]):
    """generative trace."""

    g_gen: list[AbstractLocation] = field(default_factory=list)
    p_gen_gg: list[GroundedLocation] = field(default_factory=list)
    p_gen_gi: list[GroundedLocation] = field(default_factory=list)

    def _append(self, step: TEMGenerative) -> None:
        self.g_gen.append(step.g_gen)
        self.p_gen_gg.append(step.p_gen_gg)
        self.p_gen_gi.append(step.p_gen_gi)


@dataclass
class PredictionTrace(TraceBase[Prediction]):
    """prediction trace."""

    prediction: list[Observation] = field(default_factory=list)
    logits: list[Tensor] = field(default_factory=list)

    def _append(self, step: Prediction) -> None:
        self.prediction.append(step.prediction)
        self.logits.append(step.logits)


@dataclass
class TEMReconstructionTrace(TraceBase[TEMReconstruction]):
    """reconstruction trace.

    Each reconstruction output is a `Prediction` (prediction + logits), so we
    store three `PredictionTrace` sub-traces.
    """

    y_p_inf: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gi: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gg: PredictionTrace = field(default_factory=PredictionTrace)

    def _append(self, step: TEMReconstruction) -> None:
        self.y_p_inf.append(step.y_p_inf)
        self.y_gen_gi.append(step.y_gen_gi)
        self.y_gen_gg.append(step.y_gen_gg)


@dataclass
class TEMOutputTrace(TraceBase[TEMOutput]):
    """top-level output trace."""

    inference: TEMInferenceTrace = field(default_factory=TEMInferenceTrace)
    generative: TEMGenerativeTrace = field(default_factory=TEMGenerativeTrace)
    reconstruction: TEMReconstructionTrace = field(default_factory=TEMReconstructionTrace)

    def _append(self, step: TEMOutput) -> None:
        self.inference.append(step.inference)
        self.generative.append(step.generative)
        self.reconstruction.append(step.reconstruction)


@dataclass
class LECStateTrace(TraceBase[LECState]):
    """LEC state trace."""

    cells: list[MultiScaleCode] = field(default_factory=list)
    filtered: list[MultiScaleCode] = field(default_factory=list)

    def _append(self, step: LECState) -> None:
        self.cells.append(step.cells)
        self.filtered.append(step.filtered)


@dataclass
class MECStateTrace(TraceBase[MECState]):
    """MEC state trace."""

    cells: list[AbstractLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)

    def _append(self, step: MECState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)


@dataclass
class HPCStateTrace(TraceBase[HPCState]):
    """HPC state trace."""

    cells: list[GroundedLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)
    memory: list[Optional[list[Matrix]]] = field(default_factory=list)

    def _append(self, step: HPCState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)
        self.memory.append(step.memory)


@dataclass
class TEMStateTrace(TraceBase[TEMState]):
    """full state trace.

    Stores the full `TEMState` objects, and also provides per-module sub-traces
    for convenience.
    """

    lec: LECStateTrace = field(default_factory=LECStateTrace)
    mec: MECStateTrace = field(default_factory=MECStateTrace)
    hpc: HPCStateTrace = field(default_factory=HPCStateTrace)

    def _append(self, step: TEMState) -> None:
        self.lec.append(step.lec)
        self.mec.append(step.mec)
        self.hpc.append(step.hpc)


@dataclass
class TEMTrace(TraceBase[TEMStep]):
    """full model trace.

    Stores all per-step outputs and states from a TEM rollout.

    Attributes:
        actions: List of TEM actions taken at each step.
        labels: TEMLabelTrace storing observations and locations.
        output: TEMOutputTrace storing model outputs.
        state: TEMStateTrace storing model states.
    """

    actions: list[TEMAction] = field(default_factory=list)
    labels: TEMLabelTrace = field(default_factory=TEMLabelTrace)
    output: TEMOutputTrace = field(default_factory=TEMOutputTrace)
    state: TEMStateTrace = field(default_factory=TEMStateTrace)


@dataclass
class DataTrace(TraceBase[DataStep]):
    """Plot-ready trace for environment and walk data.

    Container-style trace assembled from already-collected arrays.
    Does not support incremental construction via append().
    """

    worlds: list[World] = field(default_factory=list)
    walks: list[Walk] = field(default_factory=list)
    visited: list[list[bool]] | None = None

    def batch_size(self) -> int:
        return len(self.worlds)


@dataclass
class SimulationTrace(TraceBase[SimulationStep]):
    """Plot-ready combined trace for model+position figures.

    Container-style trace combining walk data with model outputs.
    Does not support incremental construction via append().
    """

    worlds: list[World] = field(default_factory=list)
    location_ids: list[list[int]] = field(default_factory=list)
    walks: list[Walk] = field(default_factory=list)
    model: TEMTrace = field(default_factory=TEMTrace)
