"""MEC memory inference: p→g correction from hippocampal place cells."""

from __future__ import annotations

from typing import List, Tuple

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem import utils
from torch_tem.modules import MLP
from torch_tem.settings import P2GMemSettings
from torch_tem.types import Transition


class P2GMemoryModel(nn.Module):
    """Memory-based inference: infer grid code from retrieved place cells.

    Predicts grid cell activations from hippocampal place cell patterns,
    with uncertainty modulated by memory quality (reconstruction error + norm).
    """

    def __init__(self, n_p: List[int], n_g: List[int], settings: P2GMemSettings):
        super().__init__()
        self._n_g, self._n_freq = n_g, len(n_g)
        self._settings = settings
        self._uncertainty_constant = settings.curriculum_sigma

        # Mean prediction from place cells
        self.MLP_mu_g_mem = MLP(n_p, n_g, hidden_dim=[2 * g for g in n_g])

        # Initialize last layer with truncated normal (legacy parity)
        init_w = lambda f: truncnorm.rvs(-2, 2, size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape), loc=0, scale=settings.sigma_init)
        self.MLP_mu_g_mem.set_weights(-1, [torch.tensor(init_w(f), dtype=torch.float32) for f in range(self._n_freq)])

        # Uncertainty from memory quality indicators
        self.MLP_sigma_g_mem = MLP([2 for _ in n_p], n_g, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

    def set_runtime(self, *, p2g_scale_offset: float):
        """Set runtime variance offset for curriculum training."""
        self._uncertainty_constant = p2g_scale_offset * self.settings.curriculum_sigma

    @property
    def settings(self) -> P2GMemSettings:
        """Place-to-grid memory inference settings."""
        return self._settings

    def forward(self, p_x: List[Tensor], transition: Transition) -> Transition:
        """Infer grid code from place cells with uncertainty estimation.

        Args:
            p_x: Retrieved place cell activations per frequency

        Returns:
            transition: Inferred grid cell distribution (mean, uncertainty)
        """
        g_ref, sigma_ref = transition.mean, transition.uncertainty  # Unpack for clarity

        mu = self._inference_mean(p_x)
        sigma = self._inference_uncertainty(g_ref, err=utils.squared_error(mu, g_ref))

        correction = Transition(mean=mu, uncertainty=sigma)
        return utils.inv_var_trans(transition, correction)

    def _inference_mean(self, p_x: List[Tensor]) -> List[Tensor]:
        """Infer grid cell means from place cells."""
        return self.MLP_mu_g_mem(p_x)

    def _inference_uncertainty(self, g: List[Tensor], err: List[Tensor]) -> List[Tensor]:
        """Infer grid cell uncertainties from place cells and reconstruction error."""
        sigma_g_input = [torch.cat((torch.sum(mu_f**2, dim=1, keepdim=True), torch.unsqueeze(err[f], dim=1)), dim=1) for f, mu_f in enumerate(g)]
        sigma = self.MLP_sigma_g_mem(sigma_g_input)
        return [sigma[f] + self._uncertainty_constant for f in range(self._n_freq)]
