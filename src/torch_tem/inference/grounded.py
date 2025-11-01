"""Grounded location inference for torch_tem package."""

from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ..config.facets import GroundedInferenceParams


class GroundedLocationInference(nn.Module):
    """Infers grounded location p from sensory and abstract location.

    Computes outer product: p = g ⊗ x
    """

    def __init__(self, params: GroundedInferenceParams):
        """Initialize grounded location inference.

        Args:
            params: Configuration satisfying GroundedInferenceParams protocol
        """
        super().__init__()
        self.n_f = params.n_f_calculated
        self.n_p = params.n_p_calculated

        # Store W_repeat and W_tile matrices as buffers
        W_repeat = params.W_repeat_calculated
        W_tile = params.W_tile_calculated

        for f in range(self.n_f):
            self.register_buffer(f"W_repeat_{f}", W_repeat[f])
            self.register_buffer(f"W_tile_{f}", W_tile[f])

        # Learnable weights for sensory preference
        self.w_p = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(self.n_f)])

    def forward(self, g_downsampled: List[Tensor], x_filtered: List[Tensor]) -> List[Tensor]:
        """Compute p via outer product.

        Args:
            g_downsampled: Downsampled abstract location [n_f] of [B, n_g_subsampled[f]]
            x_filtered: Temporally filtered sensory [n_f] of [B, n_x_c]

        Returns:
            p: Grounded location [n_f] of [B, n_p[f]]
        """
        p = []
        for f in range(self.n_f):
            # Outer product via matrices
            # g_repeated: [B, n_p[f]] by repeating g for each x dimension
            # x_tiled: [B, n_p[f]] by tiling x for each g dimension
            W_repeat = getattr(self, f"W_repeat_{f}")
            W_tile = getattr(self, f"W_tile_{f}")

            g_repeated = torch.matmul(g_downsampled[f], W_repeat)
            x_tiled = torch.matmul(x_filtered[f], W_tile)
            p_f = g_repeated * x_tiled

            # Apply learnable weights
            p_f = self.w_p[f] * p_f

            p.append(p_f)

        return p
