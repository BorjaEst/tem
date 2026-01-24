"""Generic rollout step container and extraction utilities.

This module provides:
- Generic RolloutStep dataclass parameterized by action, output, label, and state types
- Generic extraction functions for extracting components from rollout steps
- Type-safe specializations for TEM and other models

Architecture Note:
    RolloutStep uses Generic typing to support different model architectures while
    maintaining type safety. The extraction functions provide unified access to
    step components regardless of whether actions are present.

Example:
    ```python
    # TEM with actions
    TEMRollout = RolloutStep[list[int | None], TEMOutput, TEMLabel, TEMState]

    # HPC without actions (action type is None)
    HPCRollout = RolloutStep[None, HPCOutput, HPCLabel, HPCState]

    # Extract components generically
    actions = extract_action(step)  # Returns None if no actions
    output = extract_output(step)
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

# Type parameters for generic RolloutStep
ActionT = TypeVar("ActionT")
OutputT = TypeVar("OutputT")
LabelT = TypeVar("LabelT")
StateT = TypeVar("StateT")


@dataclass
class RolloutStep(Generic[ActionT, OutputT, LabelT, StateT]):
    """Generic container for a single rollout timestep.

    Represents the complete state of a model at one timestep, including
    the action taken, model output, supervision labels, and recurrent state.
    """

    action: ActionT
    output: OutputT
    label: LabelT
    state: StateT


__all__ = ["RolloutStep"]
