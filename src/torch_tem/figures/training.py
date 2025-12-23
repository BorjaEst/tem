"""Training diagnostics plotting utilities.

This module contains plotting helpers that are specific to the training loop and
its logged metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import matplotlib.pyplot as plt


def plot_loss_curves(trainer: Any, output_path: Optional[Path] = None) -> plt.Figure:
    """Plot a summary of the final logged loss values.

    This plot is intentionally simple: it uses the most recently logged metrics
    available on the trainer and shows a bar chart for total and component
    losses.

    Notes:
        - This function reads from ``trainer.logged_metrics``.
        - Keys vary depending on logging conventions. We support both the newer
          ``train/loss`` style (used by ``TEMLightningModule``) and the older
          ``train_loss`` style.
        - For full training curves, prefer TensorBoard.

    Args:
        trainer: A PyTorch Lightning Trainer-like object exposing
            ``logged_metrics``.
        output_path: Optional path to save the resulting figure.

    Returns:
        A Matplotlib Figure containing the loss summary.
    """

    metrics: Mapping[str, Any] = getattr(trainer, "logged_metrics", {}) or {}

    def _metric(*keys: str) -> Optional[float]:
        for key in keys:
            if key in metrics:
                try:
                    return float(metrics[key])
                except (TypeError, ValueError):
                    # Some Lightning metrics are tensors or objects with .item()
                    value = metrics[key]
                    item = getattr(value, "item", None)
                    if callable(item):
                        return float(item())
                    return None
        return None

    total = _metric("train/loss", "train_loss")
    lx = _metric("train/lx", "train_lx")
    lg = _metric("train/lg", "train_lg")
    lp = _metric("train/lp", "train_lp")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle("Training Loss Summary (Final Values)", fontsize=16, fontweight="bold")

    loss_names = []
    loss_values = []

    if total is not None:
        loss_names.append("Total")
        loss_values.append(total)
    if lx is not None:
        loss_names.append("Sensory (L_x)")
        loss_values.append(lx)
    if lg is not None:
        loss_names.append("Abstract (L_g)")
        loss_values.append(lg)
    if lp is not None:
        loss_names.append("Grounded (L_p)")
        loss_values.append(lp)

    if loss_names:
        # Use matplotlib defaults to avoid introducing any new styling tokens.
        bars = ax.bar(loss_names, loss_values)
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.set_title("Loss Components at Final Step")
        ax.grid(True, alpha=0.3, axis="y")

        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    else:
        ax.text(0.5, 0.5, "No loss metrics available", ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.annotate(
        "Note: For full training curves, view TensorBoard logs",
        xy=(0.5, -0.15),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        style="italic",
    )

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
