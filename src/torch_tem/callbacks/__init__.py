"""PyTorch Lightning callbacks for TEM training.

This package provides custom callbacks that extend TEM training with:
    - Periodic figure generation and logging (FiguresCallback)
"""

from torch_tem.callbacks.figures import FigureCallbackSettings, FiguresCallback

__all__ = [
    "FiguresCallback",
    "FigureCallbackSettings",
]
