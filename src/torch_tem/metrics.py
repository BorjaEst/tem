"""TEM evaluation metrics.

This module provides diagnostic metrics for TEM training, separated from loss
computation to maintain clean boundaries between optimization objectives and
evaluation diagnostics.

Conventions:
    - `reduction="none"` returns per-environment vectors of shape `[B]`.
    - Accuracies are float values in [0.0, 1.0] representing correctness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class AccuracyO:
    """Sensory prediction accuracy metrics with weighted averaging support.

    Attributes:
        o_p_inf: Accuracy for inference pathway (float in [0.0, 1.0]).
        o_gen_gi: Accuracy for retrieved pathway (float in [0.0, 1.0]).
        o_gen_gg: Accuracy for ancestral pathway (float in [0.0, 1.0]).

    Note:
        Internal weight tracking (_total) is used for weighted averaging.
        Users should not access or modify this field directly.
    """

    o_p_inf: Tensor  # inference pathway
    o_gen_gi: Tensor  # retrieved pathway
    o_gen_gg: Tensor  # ancestral pathway
    _total: Tensor | None = None  # weight for averaging (internal)

    @classmethod
    def zero(cls, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "AccuracyO":
        """Create a zero-initialized accuracy.

        Args:
            device: Device for tensor allocation.
            dtype: Data type for tensors.

        Returns:
            Zero-initialized :class:`AccuracyO`.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(o_p_inf=z.clone(), o_gen_gi=z.clone(), o_gen_gg=z.clone(), _total=z.clone())

    def __post_init__(self):
        """Set default _total to 1 if not provided."""
        if self._total is None:
            # Use device and dtype from first accuracy tensor
            self._total = torch.ones((), device=self.o_p_inf.device, dtype=self.o_p_inf.dtype)

    def __add__(self, other: "AccuracyO") -> "AccuracyO":
        """Add two accuracies with weighted averaging.

        Combines weighted accuracies: (acc1 * weight1 + acc2 * weight2) / (weight1 + weight2)

        Args:
            other: Another :class:`AccuracyO` to add.

        Returns:
            New :class:`AccuracyO` with weighted-averaged components.
        """
        total_new = self._total + other._total
        # Handle zero denominator case (both weights are 0)
        # Clamp to minimum 1.0 to avoid NaN
        denom = torch.clamp(total_new, min=1.0)
        # Weighted average: (a1*w1 + a2*w2) / (w1 + w2)
        return AccuracyO(
            o_p_inf=(self.o_p_inf * self._total + other.o_p_inf * other._total) / denom,
            o_gen_gi=(self.o_gen_gi * self._total + other.o_gen_gi * other._total) / denom,
            o_gen_gg=(self.o_gen_gg * self._total + other.o_gen_gg * other._total) / denom,
            _total=total_new,
        )

    def __truediv__(self, divisor: int | float) -> "AccuracyO":
        """Divide internal weight by a scalar (accuracies unchanged).

        Args:
            divisor: Scalar divisor.

        Returns:
            New :class:`AccuracyO` with scaled internal weight.
        """
        return AccuracyO(
            o_p_inf=self.o_p_inf,
            o_gen_gi=self.o_gen_gi,
            o_gen_gg=self.o_gen_gg,
            _total=self._total / divisor,
        )


class SensoryAccuracy(nn.Module):
    """Compute sensory prediction accuracies.

    This computes categorical prediction accuracy for each of the three TEM
    sensory prediction pathways, matching the structure of `SensoryReconstructionLoss`.
    """

    def __init__(self, reduction: Literal["none", "mean"] = "none"):
        """Initialize the accuracy metric.

        Args:
            reduction: Output reduction.
                - "none": return per-environment vectors of shape `[B]`.
                - "mean": return a scalar mean across environments.
        """
        super().__init__()
        self.reduction = reduction

    def forward(self, o_logits: list[Tensor], o: Tensor) -> AccuracyO:
        """Compute `AccuracyO` from logits and ground-truth observations.

        Args:
            o_logits: Three logit tensors `[infer, retrieved, ancestral]`, each
                shaped `(B, n_classes)`.
            o: Ground-truth observation. Accepts either:
                - one-hot: `(B, n_classes)`
                - class indices: `(B,)` or `(B, 1)`

        Returns:
            AccuracyO: Per-pathway prediction accuracies (float in [0.0, 1.0]).

        Raises:
            ValueError: If `o_logits` does not contain exactly 3 tensors.
        """
        if len(o_logits) != 3:
            raise ValueError(f"Expected 3 logit tensors, got {len(o_logits)}")

        # Convert o to class indices if one-hot
        if o.dim() == 2 and o.shape[1] > 1:
            labels = torch.argmax(o, dim=1)
        else:
            labels = o.squeeze(-1) if o.dim() == 2 else o

        # Compute predictions for each pathway
        pred_p = torch.argmax(o_logits[0], dim=1)  # infer pathway
        pred_g = torch.argmax(o_logits[1], dim=1)  # retrieved pathway
        pred_gt = torch.argmax(o_logits[2], dim=1)  # ancestral pathway

        # Compute per-environment correctness (float 0.0 or 1.0)
        acc_o_p_inf = (pred_p == labels).float()
        acc_o_gen_gi = (pred_g == labels).float()
        acc_o_gen_gg = (pred_gt == labels).float()

        # Apply reduction if requested
        if self.reduction == "mean":
            acc_o_p_inf = acc_o_p_inf.mean()
            acc_o_gen_gi = acc_o_gen_gi.mean()
            acc_o_gen_gg = acc_o_gen_gg.mean()

        return AccuracyO(o_p_inf=acc_o_p_inf, o_gen_gi=acc_o_gen_gi, o_gen_gg=acc_o_gen_gg)
