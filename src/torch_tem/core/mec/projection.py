"""MEC Projection to hippocampal input space.

Projects abstract location (grid cells) to hippocampal input space using
learned downsampling and expansion matrices. This enables the hippocampus
to form conjunctive place cell representations (p = g ⊙ x).

The projection pipeline:
    1. Downsample: g → g_downsampled (dimensionality reduction)
    2. Repeat/Expand: g_downsampled → g_ (hippocampal input space)

This module also provides inverse operations for memory-based inference.

Typical usage example:
    >>> config = ProjectionConfig(learn_alpha=True)
    >>> projection = Projection(W_down, W_repeat, config)
    >>> g_ = projection(abstract_location)
    >>> g_reconstructed = projection.inverse(place_cells)
"""

from typing import List

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.types import AbstractLocation, Matrix, MultiScaleCode

__all__ = ["ProjectionConfig", "Projection"]


class ProjectionConfig(BaseModel):
    """Protocol for Projection initialization parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    # Learnable parameters initialization
    learn_alpha: bool = Field(default=True, description="If True, Laplacian scales alpha are learnable; if False, frozen at initial values")


class Projection(nn.Module):
    """MEC projection head for abstract location to hippocampal input.

    Projects grid cell representations to hippocampal input space through
    learned downsampling and expansion transformations. Each frequency module
    has its own projection pathway with learnable Laplacian scales.
    ...
    """

    def __init__(self, W_down: List[Matrix], W_repeat: List[Matrix], config: ProjectionConfig):
        """Initialize MEC projection module.
        ...
        """
        super().__init__()
        self._config = config
        self._W_down = W_down
        self._W_repeat = W_repeat

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return len(self._W_down)

    @property
    def n_p(self) -> List[int]:
        """Hippocampal input dimensions per frequency module."""
        return [matrix.shape[1] for matrix in self._W_repeat]

    @property
    def n_g(self) -> List[int]:
        """Abstract location dimensions per frequency module."""
        return [matrix.shape[0] for matrix in self._W_down]

    def downsample(self, g: AbstractLocation) -> MultiScaleCode:
        """Downsample abstract location for dimensionality reduction.

        Args:
            g: Abstract location List[n_f] of (batch, n_g[f]).
        ...
        """
        return [torch.matmul(g[f], W) for f, W in enumerate(self._W_down)]

    def down_inv(self, g_downsampled: MultiScaleCode) -> AbstractLocation:
        """Inverse downsample operation for memory-based inference.

        Args:
            g_downsampled: Downsampled abstract location List[n_f] of (batch, n_g_downsampled[f]).
        ...
        """
        return [torch.matmul(g_downsampled[f], W.t()) for f, W in enumerate(self._W_down)]

    def repeat(self, g_downsampled: MultiScaleCode) -> MultiScaleCode:
        """Expand downsampled abstract location to hippocampal input space.

        Args:
            g_downsampled: Downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).

        Returns:
            Expanded hippocampal input List[n_f] of (batch, n_p[f]).
        """
        return [torch.matmul(g_downsampled[f], W) for f, W in enumerate(self._W_repeat)]

    def repeat_inv(self, p: MultiScaleCode) -> MultiScaleCode:
        """Inverse expansion operation for memory-based inference.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Reconstructed downsampled abstract location List[n_f] of (batch, n_g_subsampled[f]).
        """
        return [torch.matmul(p[f], W.t()) for f, W in enumerate(self._W_repeat)]

    def forward(self, g: AbstractLocation) -> MultiScaleCode:
        """Project abstract location to hippocampal input space.

        Args:
            g: Abstract location List[n_f] of (batch, n_g[f]).

        Returns:
            Hippocampal input List[n_f] of (batch, n_p[f]).
        """
        g_downsampled = self.downsample(g)
        return self.repeat(g_downsampled)

    def inverse(self, p: MultiScaleCode) -> MultiScaleCode:
        """Full inverse projection from hippocampal place cells to abstract location.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Reconstructed abstract location List[n_f] of (batch, n_g[f]).
        """
        p_downsampled = self.repeat_inv(p)
        return self.down_inv(p_downsampled)
