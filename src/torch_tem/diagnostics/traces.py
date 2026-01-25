from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Generic, Iterable, List, Optional, TypeVar

from torch import Tensor

from torch_tem.data.datamodule import AgentStep, DataStep
from torch_tem.data.rollout import RolloutStep, SimulationStep
from torch_tem.data.world import World
from torch_tem.model import Prediction, TEMGenerative, TEMInference, TEMLabel, TEMOutput, TEMReconstruction, TEMState, TEMStep
from torch_tem.modules.hpc import HPCState
from torch_tem.modules.lec import LECState
from torch_tem.modules.mec import MECState
from torch_tem.types import AbstractLocation, Action, GroundedLocation, LocationBelief, LocationLabel, MemoryState, MultiScaleCode, Observation

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
class TEMLabelTrace(TraceBase[TEMLabel]):
    observation: List[Observation] = field(default_factory=list)
    locations: Optional[List[LocationLabel]] = None

    @property
    def batch_size(self) -> int:
        return int(self.observation[0].shape[0]) if self.observation else 0

    def get_item(self, idx: int) -> TEMLabel:
        return TEMLabel(self.observation[idx], self.locations[idx])

    def _append(self, step: TEMLabel) -> None:
        self.observation.append(step.observation)
        self.locations = step.locations


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
class TEMTrace(TraceBase[TEMStep]):
    actions: List[Action] = field(default_factory=list)
    labels: TEMLabelTrace = field(default_factory=TEMLabelTrace)
    output: TEMOutputTrace = field(default_factory=TEMOutputTrace)
    state: TEMStateTrace = field(default_factory=TEMStateTrace)

    @property
    def batch_size(self) -> int:
        return self.state.batch_size

    def get_item(self, idx: int) -> TStep:
        return RolloutStep(self.actions[idx], self.output[idx], self.labels[idx], self.state[idx])

    def _append(self, step: TEMStep) -> None:
        self.actions.append(step.actions)
        self.labels.append(step.label)
        self.output.append(step.output)
        self.state.append(step.state)


class AgentTrace(TraceBase[AgentStep]):
    locations: List[LocationLabel] = field(default_factory=list)
    observation: List[Observation] = field(default_factory=list)
    action: List[Action] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return int(self.observation[0].shape[0]) if self.observation else 0

    def get_item(self, idx: int) -> AgentStep:
        return AgentStep(self.locations[idx], self.observation[idx], self.action[idx])

    def _append(self, step: AgentStep) -> None:
        self.locations.append(step.locations)
        self.observation.append(step.observation)
        self.action.append(step.action)


@dataclass
class DataTrace(TraceBase[DataStep]):
    environments: List[World] = field(default_factory=list)  # Batch of environments
    walks: AgentTrace = field(default_factory=AgentTrace)
    visited: List[Optional[List[List[bool]]]] = field(default_factory=list)

    def get_item(self, idx: int) -> TStep:
        return DataStep(self.environments, self.walks[idx], self.visited[idx])

    def _append(self, step: DataStep) -> None:
        self.environments = step.environments
        self.walks.append(step.agent_info)
        self.visited.append(step.visited)


@dataclass
class SimulationTrace(TraceBase[SimulationStep]):
    """Plot-ready combined trace for model+position figures.

    Container-style trace combining walk data with model outputs.
    Does not support incremental construction via append().
    """

    location_ids: List[List[int]] = field(default_factory=list)
    data: DataTrace = field(default_factory=DataTrace)
    model: TEMTrace = field(default_factory=TEMTrace)


def _batch_size_from_multiscale(code: MultiScaleCode | None) -> int:
    """Infer batch size from a multi-scale code.

    A multi-scale code is a list of tensors, one per frequency module, each of
    shape `(B, n_cells_f)`.

    Args:
        code: Multi-scale code or `None`.

    Returns:
        The inferred batch size `B`, or 0 if the code is missing/empty.
    """

    if not code:
        return 0
    return int(code[0].shape[0]) if len(code) > 0 else 0


def _batch_size_from_multiscale_trace(trace: Sequence[MultiScaleCode]) -> int:
    """Infer batch size from a trace of multi-scale codes."""

    if not trace:
        return 0
    return _batch_size_from_multiscale(trace[0])
