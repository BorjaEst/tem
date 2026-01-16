"""MEC memory inference: p→g correction from hippocampal place cells."""

from __future__ import annotations

from typing import List, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import P2GMemSettings


class P2GMemoryModel(nn.Module):
    """Memory-based inference: infer grid code from retrieved place cells.

    Predicts grid cell activations from hippocampal place cell patterns,
    with uncertainty modulated by memory quality (reconstruction error + norm).
    """

    def __init__(self, n_p: List[int], n_g: List[int], settings: P2GMemSettings):
        super().__init__()
        self._n_g, self._n_freq = n_g, len(n_g)
        self._settings = settings

        # Mean prediction from place cells
        self.MLP_mu_g_mem = MLP(n_p, n_g, hidden_dim=[2 * g for g in n_g])

        # Initialize last layer with truncated normal (legacy parity)
        init_w = lambda f: truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=settings.sigma_init)
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(init_w(f), dtype=torch.float32) for f in range(self._n_freq)])

        # Uncertainty from memory quality indicators
        self.MLP_sigma_g_mem = MLP([2 for _ in n_p], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def set_runtime(self, *, p2g_scale_offset: float):
        """Set runtime variance offset for curriculum training."""
        self.settings.curriculum_sigma *= p2g_scale_offset

    @property
    def settings(self) -> P2GMemSettings:
        """Place-to-grid memory inference settings."""
        return self._settings

    def forward(self, p_x: List[Tensor], g_ref: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Infer grid code from place cells with uncertainty estimation.

        Args:
            p_x: Retrieved place cell activations per frequency
            g_ref: Reference grid code for computing reconstruction error

        Returns:
            mu_g_mem: Predicted grid cell means
            sigma_g_mem: Predicted grid cell uncertainties
        """
        # Predict mean from place cells
        mu_g_mem = self.MLP_mu_g_mem(p_x)

        # Compute memory quality features
        err = utils.squared_error(mu_g_mem, g_ref)
        sigma_g_input = [torch.cat((torch.sum(mu_f**2, dim=1, keepdim=True), torch.unsqueeze(err[f], dim=1)), dim=1) for f, mu_f in enumerate(mu_g_mem)]

        # Predict uncertainty from quality
        sigma = self.MLP_sigma_g_mem(sigma_g_input)
        sigma_g_mem = [sigma[f] + self.settings.curriculum_sigma for f in range(self._n_freq)]

        return mu_g_mem, sigma_g_mem
