"""trace definitions - kept for backward compatibility.

These traces are lightweight wrappers around rollouts of model objects.

Notes on efficiency:
    The original implementation repeatedly materialized iterators via ``list(rollout)``
    in each trace class. This version materializes once (and avoids copying when the
    rollout is already a list) and uses small projection helpers to keep the code
    compact.

New code should use:
    - torch_tem.diagnostics.traces.ModelTrace (plot-ready, CPU/NumPy)
    - torch_tem.diagnostics.extractors.ModelTraceExtractor
"""

from dataclasses import dataclass, field
from typing import Generic, Iterable, Optional, TypeVar

from torch_tem.model import Prediction, TEMAction, TEMGenerative, TEMInference, TEMLabel, TEMOutput, TEMReconstruction, TEMState
from torch_tem.modules.hpc import HPCState
from torch_tem.modules.lec import LECState
from torch_tem.modules.mec import MECState
from torch_tem.types import *

TStep = TypeVar("TStep")


@dataclass
class _TraceBase(Generic[TStep]):
    """Appendable trace base with optional rollout ingestion.

    This base class exists to keep traces elegant and efficient:
    - Dataclasses manage default factories correctly (no shared mutable defaults)
    - Traces can be built in a single pass via `append`
    - Backward-compatible construction via `Trace(rollout)` is supported

    Subclasses must implement `append(step)`.
    """

    _rollout: Optional[Iterable[TStep]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._rollout is None:
            return
        for step in self._rollout:
            self.append(step)
        self._rollout = None

    def append(self, step: TStep) -> None:  # pragma: no cover
        raise NotImplementedError


@dataclass
class ActionTrace(_TraceBase[TEMAction]):
    """action trace.

    Stores per-step TEM actions.
    """

    actions: list[TEMAction] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMAction]) -> "ActionTrace":
        return cls(rollout)

    def append(self, step: TEMAction) -> None:
        self.actions.append(step)


@dataclass
class TEMLabelTrace(_TraceBase[TEMLabel]):
    """label trace.

    Stores observations over time and caches the (typically constant) locations
    metadata from the first label.
    """

    observation: list[Observation] = field(default_factory=list)
    locations: list[LocationLabel] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMLabel]) -> "TEMLabelTrace":
        return cls(rollout)

    def append(self, step: TEMLabel) -> None:
        self.locations = self.locations or step.locations
        self.observation.append(step.observation)


@dataclass
class TEMInferenceTrace(_TraceBase[TEMInference]):
    """inference trace."""

    g_inf: list[AbstractLocation] = field(default_factory=list)
    p_inf: list[GroundedLocation] = field(default_factory=list)
    p_xi: list[GroundedLocation] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMInference]) -> "TEMInferenceTrace":
        return cls(rollout)

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

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMGenerative]) -> "TEMGenerativeTrace":
        return cls(rollout)

    def append(self, step: TEMGenerative) -> None:
        self.g_gen.append(step.g_gen)
        self.p_gen_gg.append(step.p_gen_gg)
        self.p_gen_gi.append(step.p_gen_gi)


@dataclass
class PredictionTrace(_TraceBase[Prediction]):
    """prediction trace."""

    prediction: list[Observation] = field(default_factory=list)
    logits: list[Tensor] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[Prediction]) -> "PredictionTrace":
        return cls(rollout)

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

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMReconstruction]) -> "TEMReconstructionTrace":
        return cls(rollout)

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

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMOutput]) -> "TEMOutputTrace":
        return cls(rollout)

    def append(self, step: TEMOutput) -> None:
        self.inference.append(step.inference)
        self.generative.append(step.generative)
        self.reconstruction.append(step.reconstruction)


@dataclass
class LECStateTrace(_TraceBase[LECState]):
    """LEC state trace."""

    cells: list[MultiScaleCode] = field(default_factory=list)
    filtered: list[MultiScaleCode] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[LECState]) -> "LECStateTrace":
        return cls(rollout)

    def append(self, step: LECState) -> None:
        self.cells.append(step.cells)
        self.filtered.append(step.filtered)


@dataclass
class MECStateTrace(_TraceBase[MECState]):
    """MEC state trace."""

    cells: list[AbstractLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[MECState]) -> "MECStateTrace":
        return cls(rollout)

    def append(self, step: MECState) -> None:
        self.cells.append(step.cells)
        self.uncertainty.append(step.uncertainty)


@dataclass
class HPCStateTrace(_TraceBase[HPCState]):
    """HPC state trace."""

    cells: list[GroundedLocation] = field(default_factory=list)
    uncertainty: list[Optional[MultiScaleCode]] = field(default_factory=list)
    memory: list[MemoryState] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Iterable[HPCState]) -> "HPCStateTrace":
        return cls(rollout)

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

    @classmethod
    def from_rollout(cls, rollout: Iterable[TEMState]) -> "TEMStateTrace":
        return cls(rollout)

    def append(self, step: TEMState) -> None:
        self.lec.append(step.lec)
        self.mec.append(step.mec)
        self.hpc.append(step.hpc)
