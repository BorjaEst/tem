"""Loss functions for the Tolman-Eichenbaum Machine (TEM).

This module implements the Evidence Lower Bound (ELBO) components used to train TEM.
The total loss is composed of three parts:
1. Sensory Reconstruction Loss (L_x): Negative log-likelihood of observations.
2. Abstract Location Loss (L_g): KL divergence between inferred and predicted abstract locations.
3. Grounded Location Loss (L_p): KL divergence between inferred and retrieved grounded locations.

These losses enforce consistency between the "What" (sensory) and "Where" (spatial) pathways.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .types import AbstractLocation, GroundedLocation, SensoryPrediction, Transition


@dataclass
class LossOutput:
    """Container for TEM loss components.

    Attributes:
        total: Weighted sum of all losses (for backprop).
        lx: Sensory reconstruction loss.
        lg: Abstract location loss.
        lp: Grounded location loss.
        l_reg_g: Abstract location regularization loss.
        l_reg_p: Grounded location regularization loss.
    """

    total: Tensor
    lx: Tensor
    lg: Tensor
    lp: Tensor
    l_reg_g: Optional[Tensor] = None
    l_reg_p: Optional[Tensor] = None


class SensoryReconstructionLoss(nn.Module):
    """Computes the reconstruction loss for sensory observations (L_x).

    Minimizes the negative log-likelihood of the true observation x given the
    predicted distribution p(x|p).

    L_x = - E_q [ ln p(x | p) ]
    """

    def __init__(self):
        super().__init__()

    def forward(self, prediction: SensoryPrediction, target: Tensor) -> Tensor:
        """Compute L_x.

        Args:
            prediction: Predicted sensory observation (logits and values).
            target: Ground truth observation.

        Returns:
            Scalar loss value.
        """
        # Assuming prediction.logits is a list of tensors (multi-scale),
        # but typically we decode from the combined or highest scale.
        # Here we assume the first element is the main prediction.
        logits = prediction.logits[0]

        # Check target shape and type to choose appropriate loss
        if target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1):
            # Class indices: (B,) or (B, 1)
            loss = F.cross_entropy(logits, target.view(-1).long())
        elif target.dim() == 2 and target.shape[1] == logits.shape[1]:
            # One-hot or soft targets: (B, C)
            # Use BCEWithLogitsLoss for multi-label or CrossEntropy for soft targets
            # For TEM, observations are usually categorical, so CrossEntropy is appropriate.
            loss = F.cross_entropy(logits, target)
        else:
            raise ValueError(f"Shape mismatch: logits {logits.shape}, target {target.shape}")

        return loss


class AbstractLocationLoss(nn.Module):
    """Computes the consistency loss for abstract locations (L_g).

    Minimizes the KL divergence between the posterior (inferred from x) and
    the prior (predicted from path integration).

    L_g = D_KL( q(g | x, a) || p(g | a) )
    """

    def __init__(self):
        super().__init__()

    def forward(self, g: AbstractLocation, g_gen: Transition) -> Tensor:
        """Compute L_g.

        Args:
            g: Posterior abstract location (inferred). List of means.
            g_gen: Prior abstract location (transition). Contains means and uncertainty.

        Returns:
            Scalar loss value (summed over frequencies).
        """
        total_kl = torch.tensor(0.0, device=g[0].device)

        # Iterate over frequency modules
        for i, (mu_post, mu_prior, sigma_prior) in enumerate(zip(g, g_gen.mean, g_gen.uncertainty)):
            # We assume the posterior 'g' is a point estimate (delta function) or
            # has the same variance as the prior for simplification (Mahalanobis distance).
            # If we treat it as a point estimate, we maximize its log-likelihood under the prior.
            # L_g ~ -ln p(g_post | g_prior)
            #     = 0.5 * ((mu_post - mu_prior) / sigma_prior)^2 + ln(sigma_prior)

            # Avoid division by zero
            sigma_prior = torch.clamp(sigma_prior, min=1e-6)

            # Mahalanobis distance term
            diff = mu_post - mu_prior
            mahalanobis = 0.5 * (diff / sigma_prior).pow(2)

            # Log-determinant term (entropy of prior)
            log_det = torch.log(sigma_prior)

            # Sum over batch and dimensions
            kl_f = (mahalanobis + log_det).sum(dim=-1).mean()
            total_kl += kl_f

        return total_kl


class GroundedLocationLoss(nn.Module):
    """Computes the consistency loss for grounded locations (L_p).

    Ensures the inferred place cells (p) are consistent with memory retrieval (p_g)
    and optionally with memory retrieval from sensory input (p_x).

    L_p = D_KL( q(p | x, g) || p(p | g) ) + D_KL( q(p | x, g) || p(p | x) )
    """

    def __init__(self):
        super().__init__()

    def forward(self, p: GroundedLocation, p_g: Optional[GroundedLocation] = None, p_x: Optional[GroundedLocation] = None) -> Tensor:
        """Compute L_p.

        Args:
            p: Inferred grounded location (from x and g).
            p_g: Retrieved grounded location (from memory via g).
            p_x: Retrieved grounded location (from memory via x).

        Returns:
            Scalar loss value.
        """
        total_loss = torch.tensor(0.0, device=p[0].device)

        if p_g is not None:
            for p_inf, p_ret in zip(p, p_g):
                total_loss += F.mse_loss(p_inf, p_ret)

        if p_x is not None:
            for p_inf, p_ret in zip(p, p_x):
                total_loss += F.mse_loss(p_inf, p_ret)

        return total_loss


class RegularizationLoss(nn.Module):
    """Computes regularization losses for latent representations.

    Enforces:
    1. L2 penalty on abstract locations (g) to prevent exploding values.
    2. L1 penalty on grounded locations (p) to enforce sparsity.
    """

    def forward(self, g: AbstractLocation, p: GroundedLocation) -> Tuple[Tensor, Tensor]:
        # L2 on g: sum(g^2)
        l_reg_g = sum(g_i.pow(2).sum(dim=1).mean() for g_i in g)

        # L1 on p: sum(|p|)
        l_reg_p = sum(p_i.abs().sum(dim=1).mean() for p_i in p)

        return l_reg_g, l_reg_p


class TEMLoss(nn.Module):
    """Aggregates all TEM loss components."""

    def __init__(self, w_x: float = 1.0, w_g: float = 1.0, w_p: float = 1.0, w_reg_g: float = 0.1, w_reg_p: float = 0.1):
        super().__init__()
        self.w_x = w_x
        self.w_g = w_g
        self.w_p = w_p
        self.w_reg_g = w_reg_g
        self.w_reg_p = w_reg_p

    def forward(self, lx: Tensor, lp: Tensor, lg: Tensor, l_reg_g: Optional[Tensor] = None, l_reg_p: Optional[Tensor] = None) -> LossOutput:
        total = self.w_x * lx + self.w_p * lp + self.w_g * lg

        if l_reg_g is not None:
            total += self.w_reg_g * l_reg_g

        if l_reg_p is not None:
            total += self.w_reg_p * l_reg_p

        return LossOutput(total=total, lx=lx, lp=lp, lg=lg, l_reg_g=l_reg_g, l_reg_p=l_reg_p)
