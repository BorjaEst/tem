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
class AccuracyX:
    """Sensory prediction accuracy metrics.

    Attributes:
        p: Accuracy of prediction from inferred grounded location (x_p pathway).
        g: Accuracy of prediction from memory retrieval (x_g pathway).
        gt: Accuracy of prediction from ancestral/generative rollout (x_gen pathway).
    """

    p: Tensor  # x_p accuracy (infer pathway)
    g: Tensor  # x_g accuracy (retrieved pathway)
    gt: Tensor  # x_gen accuracy (ancestral/generative pathway)


@dataclass
class AccuracyCounts:
    """Accumulator for sensory accuracy computation.

    Stores sums of correct predictions (numerators) and the number of evaluated
    predictions (denominator). This mirrors the loss accumulation pattern and
    can be converted to :class:`AccuracyX` at the end of a rollout.

    Attributes:
        p: Sum of correct predictions from inference pathway.
        g: Sum of correct predictions from retrieved pathway.
        gt: Sum of correct predictions from ancestral pathway.
        total: Total number of evaluated predictions.
    """

    p: Tensor
    g: Tensor
    gt: Tensor
    total: Tensor

    @classmethod
    def zero(cls, *, device: torch.device | str, dtype: torch.dtype = torch.float32) -> "AccuracyCounts":
        """Create a zero-initialized accumulator.

        Args:
            device: Device for tensor allocation.
            dtype: Data type for tensors.

        Returns:
            Zero-initialized :class:`AccuracyCounts`.
        """
        z = torch.zeros((), device=device, dtype=dtype)
        return cls(p=z.clone(), g=z.clone(), gt=z.clone(), total=z.clone())

    def __add__(self, other: "AccuracyCounts") -> "AccuracyCounts":
        """Add two accuracy accumulators element-wise.

        Args:
            other: Another :class:`AccuracyCounts` to add.

        Returns:
            New :class:`AccuracyCounts` with summed components.
        """
        return AccuracyCounts(
            p=self.p + other.p,
            g=self.g + other.g,
            gt=self.gt + other.gt,
            total=self.total + other.total,
        )

    def __truediv__(self, divisor: int | float) -> "AccuracyCounts":
        """Divide all components by a scalar.

        Args:
            divisor: Scalar divisor.

        Returns:
            New :class:`AccuracyCounts` with scaled components.
        """
        return AccuracyCounts(
            p=self.p / divisor,
            g=self.g / divisor,
            gt=self.gt / divisor,
            total=self.total / divisor,
        )

    def to_accuracy(self) -> AccuracyX:
        """Convert accumulated counts to mean accuracies.

        Divides summed correct predictions by total count, handling the
        zero-denominator case.

        Returns:
            :class:`AccuracyX` with mean accuracies in [0.0, 1.0].
        """
        denom = torch.clamp(self.total, min=1.0)
        return AccuracyX(p=self.p / denom, g=self.g / denom, gt=self.gt / denom)


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

    def forward(self, x_logits: list[Tensor], x: Tensor) -> AccuracyX:
        """Compute `AccuracyX` from logits and ground-truth observations.

        Args:
            x_logits: Three logit tensors `[infer, retrieved, ancestral]`, each
                shaped `(B, n_classes)`.
            x: Ground-truth observation. Accepts either:
                - one-hot: `(B, n_classes)`
                - class indices: `(B,)` or `(B, 1)`

        Returns:
            AccuracyX: Per-pathway prediction accuracies (float in [0.0, 1.0]).

        Raises:
            ValueError: If `x_logits` does not contain exactly 3 tensors.
        """
        if len(x_logits) != 3:
            raise ValueError(f"Expected 3 logit tensors, got {len(x_logits)}")

        # Convert x to class indices if one-hot
        if x.dim() == 2 and x.shape[1] > 1:
            labels = torch.argmax(x, dim=1)
        else:
            labels = x.squeeze(-1) if x.dim() == 2 else x

        # Compute predictions for each pathway
        pred_p = torch.argmax(x_logits[0], dim=1)  # infer pathway
        pred_g = torch.argmax(x_logits[1], dim=1)  # retrieved pathway
        pred_gt = torch.argmax(x_logits[2], dim=1)  # ancestral pathway

        # Compute per-environment correctness (float 0.0 or 1.0)
        acc_p = (pred_p == labels).float()
        acc_g = (pred_g == labels).float()
        acc_gt = (pred_gt == labels).float()

        # Apply reduction if requested
        if self.reduction == "mean":
            acc_p = acc_p.mean()
            acc_g = acc_g.mean()
            acc_gt = acc_gt.mean()

        return AccuracyX(p=acc_p, g=acc_g, gt=acc_gt)
