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
