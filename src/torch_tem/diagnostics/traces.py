from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Generic, Iterable, List, Optional, Tuple, TypeVar

from torch import Tensor

from torch_tem.data.datamodule import WorldStep
from torch_tem.data.world import World
from torch_tem.model import Model, Prediction, RolloutStep, RolloutStream, TEMGenerative, TEMInference, TEMOutput, TEMReconstruction, TEMState
from torch_tem.modules.hpc import HPCState
from torch_tem.modules.lec import LECState
from torch_tem.modules.mec import MECState
from torch_tem.types import *

TStep = TypeVar("TStep")
TraceT = TypeVar("TraceT", bound="TraceBase")
Batch = Tuple[WalkBatch, list[list[bool]]]


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

    @property
    @abstractmethod
    def batch_size(self) -> int:
        raise NotImplementedError

    def downsample_time(self, stride: int) -> TraceBase[TStep]:
        if stride <= 1:
            return self
        iterable = (x for i, x in enumerate(self) if i % stride == 0)
        return type(self).from_iter(iterable, stop=None, meta=self.meta.copy())

    @classmethod
    def from_iter(cls, iterable: Iterable[TStep], *, stop: int | None = None, meta: dict[str, Any] | None = None) -> TraceT:
        trace = cls(meta=(meta or {}))
        return trace.attach(islice(iterable, stop))


@dataclass
class TEMInferenceTrace(TraceBase[TEMInference]):
    g_inf: List[AbstractLocation] = field(default_factory=list)
    p_inf: List[GroundedLocation] = field(default_factory=list)
    p_xi: List[GroundedLocation] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return _batch_size_from_multiscale_trace(self.g_inf)

    def get_item(self, idx: int) -> TEMInference:
        return TEMInference(self.g_inf[idx], self.p_inf[idx], self.p_xi[idx])

    def _append(self, step: TEMInference) -> None:
        self.g_inf.append(step.g_inf)
        self.p_inf.append(step.p_inf)
        self.p_xi.append(step.p_xi)


@dataclass
class TEMGenerativeTrace(TraceBase[TEMGenerative]):
    g_gen: List[AbstractLocation] = field(default_factory=list)
    p_gen_gg: List[GroundedLocation] = field(default_factory=list)
    p_gen_gi: List[GroundedLocation] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return _batch_size_from_multiscale_trace(self.g_gen)

    def get_item(self, idx: int) -> TEMGenerative:
        return TEMGenerative(self.g_gen[idx], self.p_gen_gg[idx], self.p_gen_gi[idx])

    def _append(self, step: TEMGenerative) -> None:
        self.g_gen.append(step.g_gen)
        self.p_gen_gg.append(step.p_gen_gg)
        self.p_gen_gi.append(step.p_gen_gi)


@dataclass
class PredictionTrace(TraceBase[Prediction]):
    prediction: List[Observation] = field(default_factory=list)
    logits: List[Tensor] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.logits[0].shape[0]) if self.logits else 0

    def get_item(self, idx: int) -> Prediction:
        return Prediction(self.prediction[idx], self.logits[idx])

    def _append(self, step: Prediction) -> None:
        self.prediction.append(step.prediction)
        self.logits.append(step.logits)


@dataclass
class TEMReconstructionTrace(TraceBase[TEMReconstruction]):
    y_p_inf: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gi: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gg: PredictionTrace = field(default_factory=PredictionTrace)

    @property
    def batch_size(self) -> int:
        return self.y_p_inf.batch_size

    def get_item(self, idx: int) -> TEMReconstruction:
        return TEMReconstruction(self.y_p_inf[idx], self.y_gen_gi[idx], self.y_gen_gg[idx])

    def _append(self, step: TEMReconstruction) -> None:
        self.y_p_inf.append(step.y_p_inf)
        self.y_gen_gi.append(step.y_gen_gi)
        self.y_gen_gg.append(step.y_gen_gg)


@dataclass
class TEMOutputTrace(TraceBase[TEMOutput]):
    inference: TEMInferenceTrace = field(default_factory=TEMInferenceTrace)
    generative: TEMGenerativeTrace = field(default_factory=TEMGenerativeTrace)
    reconstruction: TEMReconstructionTrace = field(default_factory=TEMReconstructionTrace)

    @property
    def batch_size(self) -> int:
        return self.inference.batch_size

    def get_item(self, idx: int) -> TEMOutput:
        return TEMOutput(self.inference[idx], self.generative[idx], self.reconstruction[idx])

    def _append(self, step: TEMOutput) -> None:
        self.inference.append(step.inference)
        self.generative.append(step.generative)
        self.reconstruction.append(step.reconstruction)


@dataclass
class LECStateTrace(TraceBase[LECState]):
    cells: List[MultiScaleCode] = field(default_factory=list)
    filtered: List[MultiScaleCode] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return _batch_size_from_multiscale_trace(self.cells)

    def get_item(self, idx: int) -> LECState:
        return LECState(self.cells[idx], self.filtered[idx])

    def _append(self, step: LECState) -> None:
        self.cells.append(step.cells)
        self.filtered.append(step.filtered)


@dataclass
class MECStateTrace(TraceBase[MECState]):
    cells: List[AbstractLocation] = field(default_factory=list)
    uncertainty: List[Optional[MultiScaleCode]] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return _batch_size_from_multiscale_trace(self.cells)

    def get_item(self, idx: int) -> MECState:
        location = LocationBelief(mean=self.cells[idx], uncertainty=self.uncertainty[idx])
        return MECState(location)

    def _append(self, step: MECState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)


@dataclass
class HPCStateTrace(TraceBase[HPCState]):
    cells: List[GroundedLocation] = field(default_factory=list)
    uncertainty: List[Optional[MultiScaleCode]] = field(default_factory=list)
    memory: List[MemoryState] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return _batch_size_from_multiscale_trace(self.cells)

    def get_item(self, idx: int) -> HPCState:
        location = LocationBelief(mean=self.cells[idx], uncertainty=self.uncertainty[idx])
        return HPCState(location, self.memory[idx])

    def _append(self, step: HPCState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)
        self.memory.append(step.memory)


@dataclass
class TEMStateTrace(TraceBase[TEMState]):
    lec: LECStateTrace = field(default_factory=LECStateTrace)
    mec: MECStateTrace = field(default_factory=MECStateTrace)
    hpc: HPCStateTrace = field(default_factory=HPCStateTrace)

    @property
    def batch_size(self) -> int:
        return self.lec.batch_size

    def get_item(self, idx: int) -> TEMState:
        return TEMState(self.lec[idx], self.mec[idx], self.hpc[idx])

    def _append(self, step: TEMState) -> None:
        self.lec.append(step.lec)
        self.mec.append(step.mec)
        self.hpc.append(step.hpc)


@dataclass
class WorldTrace(TraceBase[WorldStep]):
    environments: List[World] = field(default_factory=list)  # (B, )
    visited: Optional[List[List[bool]]] = None  # (B, )

    locations: List[List[LocationLabel]] = field(default_factory=list)  # (T, B)
    observations: List[Observation] = field(default_factory=list)  # (T, B)
    actions: List[List[Action]] = field(default_factory=list)  # (T, B)

    @property
    def batch_size(self) -> int:
        return int(self.observations[0].shape[0]) if self.observations else 0

    def get_item(self, idx: int) -> WorldStep:
        return WorldStep(self.locations[idx], self.observations[idx], self.actions[idx])

    def _append(self, step: WorldStep) -> None:
        self.locations.append(step.locations)
        self.observations.append(step.observation)
        self.actions.append(step.action)

    @property
    def location_ids(self) -> list[list[int]]:
        ids_t = [[int(loc["id"]) for loc in locs_t] for locs_t in self.locations]  # (T, B)
        return [list(env_series) for env_series in zip(*ids_t)] if ids_t else []

    @classmethod
    def from_batch(cls, batch: Batch, environments: list[World], **kwargs) -> WorldTrace:
        iterable = (WorldStep(locations=step[0], observation=step[1], action=step[2]) for step in batch[0])
        trace: WorldTrace = cls.from_iter(iterable, **kwargs)
        trace.environments, trace.visited = environments, batch[1]
        return trace


@dataclass
class RolloutTrace(TraceBase[RolloutStep]):
    world_step: WorldTrace = field(default_factory=WorldTrace)  # (T, B)
    output: TEMOutputTrace = field(default_factory=TEMOutputTrace)  # (T, B)
    state: TEMStateTrace = field(default_factory=TEMStateTrace)  # (T, B)

    @property
    def batch_size(self) -> int:
        return self.world_step.batch_size

    def get_item(self, idx: int) -> RolloutStep:
        return RolloutStep(self.world_step[idx], self.output[idx], self.state[idx])

    def _append(self, step: RolloutStep) -> None:
        self.world_step.append(step.world_step)
        self.output.append(step.output)
        self.state.append(step.state)

    @classmethod
    def from_batch(cls, batch: Batch, environments: list[World], model: Model, *, initial: TEMState | None = None, **kwargs) -> "RolloutTrace":
        trace: RolloutTrace = cls.from_iter(RolloutStream(model, batch[0], initial), **kwargs)
        trace.world_step.environments, trace.world_step.visited = environments, batch[1]
        return trace


def _batch_size_from_multiscale(code: MultiScaleCode | None) -> int:
    if not code:
        return 0
    return int(code[0].shape[0]) if len(code) > 0 else 0


def _batch_size_from_multiscale_trace(trace: Sequence[MultiScaleCode]) -> int:
    if not trace:
        return 0
    return _batch_size_from_multiscale(trace[0])
