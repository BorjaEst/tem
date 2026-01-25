from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from itertools import islice
from typing import Any, Generic, Iterable, List, Optional, TypeVar

import numpy as np
from torch import Tensor

from torch_tem.data.datamodule import DataModule
from torch_tem.data.world import World
from torch_tem.model import Model as TEMModel
from torch_tem.model import Prediction, RolloutStream, TEMAction, TEMGenerative, TEMInference, TEMLabel, TEMOutput, TEMReconstruction, TEMState, TEMStep
from torch_tem.modules.hpc import HPCState
from torch_tem.modules.lec import LECState
from torch_tem.modules.mec import MECState
from torch_tem.types import AbstractLocation, GroundedLocation, LocationLabel, Matrix, MultiScaleCode, Observation
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
        return TEMLabel(
            observation=self.observation[idx],
            locations=self.locations,
        )

    def _append(self, step: TEMLabel) -> None:
        self.observation.append(step.observation)
        self.locations = step.locations


@dataclass
class TEMInferenceTrace(_TraceBase[TEMInference]):
    """inference trace."""

    g_inf: list[AbstractLocation] = field(default_factory=list)
    p_inf: list[GroundedLocation] = field(default_factory=list)
    p_xi: list[GroundedLocation] = field(default_factory=list)

    def append(self, step: TEMInference) -> None:
        self.g_inf.append(step.g_inf)
        self.p_inf.append(step.p_inf)
        self.p_xi.append(step.p_xi)


@dataclass
class TEMGenerativeTrace(_TraceBase[TEMGenerative]):
    """generative trace."""

    g_gen: list[AbstractLocation] = field(default_factory=list)
    p_gen_gg: list[GroundedLocation] = field(default_factory=list)
    p_gen_gi: list[GroundedLocation] = field(default_factory=list)

    def append(self, step: TEMGenerative) -> None:
        self.g_gen.append(step.g_gen)
        self.p_gen_gg.append(step.p_gen_gg)
        self.p_gen_gi.append(step.p_gen_gi)


@dataclass
class PredictionTrace(_TraceBase[Prediction]):
    """prediction trace."""

    prediction: list[Observation] = field(default_factory=list)
    logits: list[Tensor] = field(default_factory=list)

    def append(self, step: Prediction) -> None:
        self.prediction.append(step.prediction)
        self.logits.append(step.logits)


@dataclass
class TEMReconstructionTrace(_TraceBase[TEMReconstruction]):
    """reconstruction trace.

    Each reconstruction output is a `Prediction` (prediction + logits), so we
    store three `PredictionTrace` sub-traces.
    """

    y_p_inf: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gi: PredictionTrace = field(default_factory=PredictionTrace)
    y_gen_gg: PredictionTrace = field(default_factory=PredictionTrace)

    def append(self, step: TEMReconstruction) -> None:
        self.y_p_inf.append(step.y_p_inf)
        self.y_gen_gi.append(step.y_gen_gi)
        self.y_gen_gg.append(step.y_gen_gg)


@dataclass
class TEMOutputTrace(_TraceBase[TEMOutput]):
    """top-level output trace."""

    inference: TEMInferenceTrace = field(default_factory=TEMInferenceTrace)
    generative: TEMGenerativeTrace = field(default_factory=TEMGenerativeTrace)
    reconstruction: TEMReconstructionTrace = field(default_factory=TEMReconstructionTrace)

    def append(self, step: TEMOutput) -> None:
        self.inference.append(step.inference)
        self.generative.append(step.generative)
        self.reconstruction.append(step.reconstruction)


@dataclass
class LECStateTrace(_TraceBase[LECState]):
    """LEC state trace."""

    cells: list[MultiScaleCode] = field(default_factory=list)
    filtered: list[MultiScaleCode] = field(default_factory=list)

    def append(self, step: LECState) -> None:
        self.cells.append(step.cells)
        self.filtered.append(step.filtered)


@dataclass
class MECStateTrace(_TraceBase[MECState]):
    """MEC state trace."""

    cells: list[AbstractLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)

    def append(self, step: MECState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)


@dataclass
class HPCStateTrace(_TraceBase[HPCState]):
    """HPC state trace."""

    cells: list[GroundedLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)
    memory: list[Optional[list[Matrix]]] = field(default_factory=list)

    def append(self, step: HPCState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)
        self.memory.append(step.memory)


@dataclass
class TEMStateTrace(_TraceBase[TEMState]):
    """full state trace.

    Stores the full `TEMState` objects, and also provides per-module sub-traces
    for convenience.
    """

    lec: LECStateTrace = field(default_factory=LECStateTrace)
    mec: MECStateTrace = field(default_factory=MECStateTrace)
    hpc: HPCStateTrace = field(default_factory=HPCStateTrace)

    def append(self, step: TEMState) -> None:
        self.lec.append(step.lec)
        self.mec.append(step.mec)
        self.hpc.append(step.hpc)


@dataclass
class ModelTrace(_TraceBase[TEMStep]):
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

    def append(self, step: TEMStep) -> None:
        self.actions.append(step.action)
        self.labels.append(step.label)
        self.output.append(step.output)

    @property
    def batch_size(self) -> int:
        # TODO

    @property
    def n_steps(self) -> int:
        # TODO

    def select_env(self, env_idx: int) -> "ModelTrace":
        _validate_env_idx(env_idx, self.batch_size)

        actions = [[a[env_idx]] for a in self.actions]
        labels = self.labels.select_env(env_idx) if self.labels.batch_size else self.labels
        output = self.output.select_env(env_idx) if self.output.batch_size else self.output

        return ModelTrace(
            actions=actions,
            labels=labels,
            output=output,
            meta=self.meta.copy(),
        )

    def downsample_time(self, stride: int) -> "ModelTrace":
        self.validate()
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if stride == 1:
            return self

        # TODO
        if self.actions and len(self.actions) != base_len:
            raise ValueError(f"Inconsistent ModelTrace time series lengths: expected {base_len}, got {len(self.actions)}")

        return ModelTrace(
            actions=[self.actions[i] for i in keep_idxs] if self.actions else [],
            labels=self.labels.downsample_time(stride),
            output=self.output.downsample_time(stride),
            meta=self.meta.copy(),
        )


@dataclass
class DataTrace(TraceBase):
    """Plot-ready trace for environment and walk data.

    Container-style trace assembled from already-collected arrays.
    Does not support incremental construction via append().
    """

    worlds: list[World] = field(default_factory=list)
    walks: list[list[list[Any]]] = field(default_factory=list)  # env-major: walks[env][t] = [location, obs, action]
    visited: list[list[bool]] | None = None

    @classmethod
    def from_chunk(
        cls,
        *,
        worlds: list[World],
        chunk: list[list[Any]],
        visited: list[list[bool]] | None = None,
        meta: dict[str, Any] | None = None,
        max_steps: int | None = None,
        downsample_stride: int = 1,
    ) -> "DataTrace":
        if downsample_stride < 1:
            raise ValueError("downsample_stride must be >= 1")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be >= 0 or None")

        chunk_limited = chunk[:max_steps] if max_steps is not None else chunk
        walks = time_major_to_env_major_walks(chunk_limited)
        if downsample_stride > 1:
            walks = downsample_env_major(walks, downsample_stride)

        trace = cls(worlds=worlds, walks=walks, visited=visited, meta=(meta or {}))
        trace.validate()
        return trace

    @property
    def batch_size(self) -> int:
        return len(self.worlds)

    @property
    def n_steps(self) -> int:
        if len(self.walks) == 0 or len(self.walks[0]) == 0:
            return 0
        return len(self.walks[0])

    def select_env(self, env_idx: int) -> "DataTrace":
        _validate_env_idx(env_idx, self.batch_size)

        visited = [self.visited[env_idx]] if self.visited is not None else None

        return DataTrace(
            worlds=[self.worlds[env_idx]],
            walks=[self.walks[env_idx]],
            visited=visited if visited is not None else None,
            meta=self.meta.copy(),
        )

    def downsample_time(self, stride: int) -> "DataTrace":
        if stride < 1:
            raise ValueError(f"stride must be >= 1 (got {stride})")
        if stride == 1:
            return self

        downsampled_walks = downsample_env_major(self.walks, stride)
        return DataTrace(
            worlds=self.worlds,
            walks=downsampled_walks,
            visited=self.visited,
            meta=self.meta.copy(),
        )

    def validate(self) -> None:
        """Validate DataTrace invariants."""
        if len(self.worlds) != len(self.walks):
            raise ValueError(f"Inconsistent batch_size: {len(self.worlds)} worlds but {len(self.walks)} walks")

        if self.visited is not None:
            if not isinstance(self.visited, list):
                raise ValueError(f"Inconsistent visited: expected list[list[bool]]; got {type(self.visited).__name__}")
            if len(self.visited) != len(self.worlds):
                raise ValueError(f"Inconsistent visited: expected {len(self.worlds)} items, got {len(self.visited)}")
            for env_i, visited_env in enumerate(self.visited):
                if not isinstance(visited_env, list):
                    raise ValueError("Inconsistent visited: expected list[list[bool]] " f"but visited[{env_i}] is {type(visited_env).__name__}")
                n_locations = getattr(self.worlds[env_i], "n_locations", None)
                if n_locations is not None and len(visited_env) != int(n_locations):
                    raise ValueError(f"Inconsistent visited[{env_i}] length: expected worlds[{env_i}].n_locations={int(n_locations)}, got {len(visited_env)}")

        if self.batch_size > 0:
            expected_steps = len(self.walks[0])
            for i, walk in enumerate(self.walks):
                if len(walk) != expected_steps:
                    raise ValueError(f"Inconsistent walk lengths: walks[0]={expected_steps} but walks[{i}]={len(walk)}")


@dataclass
class RolloutTrace(TraceBase):
    """Plot-ready combined trace for model+position figures.

    Container-style trace combining walk data with model outputs.
    Does not support incremental construction via append().
    """

    worlds: list[World] = field(default_factory=list)
    walks: list[list[list[Any]]] = field(default_factory=list)  # env-major: walks[env][t] = [location, obs, action]
    location_ids: list[list[int]] = field(default_factory=list)  # location_ids[env][t] = int
    model: ModelTrace = field(default_factory=ModelTrace)

    @classmethod
    def from_rollout(
        cls,
        *,
        worlds: list[World],
        rollout: Iterable[TEMStep],
        chunk: list[list[Any]],
        max_steps: int | None = None,
        downsample_stride: int = 1,
        meta: dict[str, Any] | None = None,
    ) -> "RolloutTrace":
        """Build a RolloutTrace from a rollout iterator and its originating chunk.

        The same `max_steps` and `downsample_stride` are applied to both the model
        trace and the derived walk/location arrays.
        """
        if downsample_stride < 1:
            raise ValueError("downsample_stride must be >= 1")
        if max_steps is not None and max_steps < 0:
            raise ValueError("max_steps must be >= 0 or None")

        chunk_limited = chunk[:max_steps] if max_steps is not None else chunk

        model_trace = ModelTrace.from_rollout(
            rollout=rollout,
            max_steps=max_steps,
            downsample_stride=downsample_stride,
            meta=meta,
        )

        walks = time_major_to_env_major_walks(chunk_limited)
        location_ids = time_major_location_ids(chunk_limited)

        if downsample_stride > 1:
            walks = downsample_env_major(walks, downsample_stride)
            location_ids = downsample_env_major(location_ids, downsample_stride)

        trace = cls(
            worlds=worlds,
            walks=walks,
            location_ids=location_ids,
            model=model_trace,
            meta=(meta or {}),
        )
        trace.validate()
        return trace

    @property
    def batch_size(self) -> int:
        return len(self.worlds)

    @property
    def n_steps(self) -> int:
        if self.batch_size == 0 or not self.walks:
            return 0
        return len(self.walks[0])

    def select_env(self, env_idx: int) -> "RolloutTrace":
        _validate_env_idx(env_idx, self.batch_size)

        return RolloutTrace(
            worlds=[self.worlds[env_idx]],
            walks=[self.walks[env_idx]],
            location_ids=[self.location_ids[env_idx]],
            model=self.model.select_env(env_idx),
            meta=self.meta.copy(),
        )

    def downsample_time(self, stride: int) -> "RolloutTrace":
        if stride < 1:
            raise ValueError(f"stride must be >= 1 (got {stride})")
        if stride == 1:
            return self

        walks_ds = downsample_env_major(self.walks, stride)
        locs_ds = downsample_env_major(self.location_ids, stride)
        model_ds = self.model.downsample_time(stride)

        return RolloutTrace(
            worlds=self.worlds,
            walks=walks_ds,
            location_ids=locs_ds,
            model=model_ds,
            meta=self.meta.copy(),
        )

    def validate(self) -> None:
        """Validate RolloutTrace invariants."""
        if len(self.worlds) != len(self.walks):
            raise ValueError(f"Inconsistent batch_size: {len(self.worlds)} worlds but {len(self.walks)} walks")

        if len(self.location_ids) != len(self.walks):
            raise ValueError(f"Inconsistent location_ids: expected {len(self.walks)}, got {len(self.location_ids)}")

        if self.batch_size > 0:
            expected_steps = len(self.walks[0])
            for i in range(self.batch_size):
                if len(self.walks[i]) != expected_steps:
                    raise ValueError(f"Inconsistent walk lengths: walks[0]={expected_steps} but walks[{i}]={len(self.walks[i])}")
                if len(self.location_ids[i]) != expected_steps:
                    raise ValueError(f"Inconsistent location_ids: expected {expected_steps}, got {len(self.location_ids[i])} for env {i}")

        # Validate model trace alignment when the model carries time/batch information
        if self.n_steps and self.model.n_steps and self.model.n_steps != self.n_steps:
            raise ValueError(f"Model trace misalignment: walk has {self.n_steps} steps but model has {self.model.n_steps}")

        if self.model.batch_size and self.model.batch_size != self.batch_size:
            raise ValueError(f"Model batch size mismatch: walk has {self.batch_size} envs but model has {self.model.batch_size}")


__all__ = [
    "TraceBase",
    "AppendableTraceBase",
    "TEMLabelTrace",
    "TEMInferenceTrace",
    "TEMGenerativeTrace",
    "TEMOutputTrace",
    "ModelTrace",
    "DataTrace",
    "RolloutTrace",
]


