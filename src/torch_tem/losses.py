"""Modern loss modules for TEM training.

This module provides flexible, pathway-based loss computation following PyTorch best practices.
The TEM model has two main computational pathways that can be trained independently or jointly:

Pathways:
    - Generative: g → p → x (path integration, memory retrieval, sensory generation)
    - Inference: x → p, g (sensory encoding, location inference, memory grounding)

Training Modes:
    1. Generative-only: Pre-train transition dynamics and sensory generation
    2. Inference-only: Pre-train sensory encoding and location inference
    3. Joint: Full TEM training with consistency between pathways (teacher forcing)

Examples:
    >>> # Generative-only training
    >>> gen_loss = GenerativeLoss()
    >>> output = gen_loss(gen_outputs, x_target)
    >>> output.total.backward()
    >>>
    >>> # Inference-only training
    >>> inf_loss = InferenceLoss()
    >>> output = inf_loss(inf_outputs, x_target)
    >>> output.total.backward()
    >>>
    >>> # Full TEM training with teacher forcing
    >>> tem_loss = TEMLoss()
    >>> output = tem_loss(gen_outputs, inf_outputs, x_target)
    >>> output.total.backward()

Architecture:
    LossOutput: Structured output with total loss and component dict
    GenerativeOutputs: Protocol for generative pathway outputs
    InferenceOutputs: Protocol for inference pathway outputs
    GenerativeLoss: Loss for generative pathway (independent)
    InferenceLoss: Loss for inference pathway (independent)
    TEMLoss: Combined loss with pathway consistency (teacher forcing)

Legacy Compatibility:
    Losses: Original 8-component dataclass (deprecated, use pathway losses)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .types import SensoryPrediction

# =============================================================================
# Pathway Output Protocols
# =============================================================================


class GenerativeOutputs(Protocol):
    """Protocol defining outputs from the generative pathway.

    The generative pathway produces abstract and grounded locations through
    transition dynamics and memory retrieval, plus sensory predictions.
    """

    g: List[Tensor]  # Abstract locations List[n_f] of [B, n_g[f]]
    p: List[Tensor]  # Grounded locations List[n_f] of [B, n_p[f]]
    x: SensoryPrediction  # Sensory predictions


class InferenceOutputs(Protocol):
    """Protocol defining outputs from the inference pathway.

    The inference pathway produces abstract and grounded locations through
    sensory encoding and belief propagation, plus sensory reconstructions.
    """

    g: List[Tensor]  # Abstract locations List[n_f] of [B, n_g[f]]
    p: List[Tensor]  # Grounded locations List[n_f] of [B, n_p[f]]
    x: SensoryPrediction  # Sensory reconstructions
    p_x: Optional[List[Tensor]] = None  # Optional sensory-retrieved locations


class GenerativeLossWeights(Protocol):
    """Protocol defining weight configuration for GenerativeLoss."""

    w_x_gen: float  # Weight for sensory generation quality
    w_reg_g: float  # Weight for abstract location regularization
    w_g: float  # Weight for teacher forcing consistency


class InferenceLossWeights(Protocol):
    """Protocol defining weight configuration for InferenceLoss."""

    w_x_inf: float  # Weight for sensory reconstruction quality
    w_p_x: float  # Weight for sensory-grounded consistency
    w_reg_p: float  # Weight for grounded location regularization


class TEMLossWeights(Protocol):
    """Protocol defining weight configuration for TEMLoss."""

    w_x_gen: float  # Weight for generative sensory loss
    w_x_inf: float  # Weight for inference sensory loss
    w_p_consistency: float  # Weight for grounded consistency
    w_g_consistency: float  # Weight for abstract consistency
    w_p_x: float  # Weight for sensory-grounded consistency
    w_reg_g: float  # Weight for abstract regularization
    w_reg_p: float  # Weight for grounded regularization


# =============================================================================
# Default Weight Configurations
# =============================================================================


@dataclass
class DefaultGenerativeWeights:
    """Default weight configuration for GenerativeLoss."""

    w_x_gen: float = 1.0
    w_reg_g: float = 0.01
    w_g: float = 1.0


@dataclass
class DefaultInferenceWeights:
    """Default weight configuration for InferenceLoss."""

    w_x_inf: float = 1.0
    w_p_x: float = 1.0
    w_reg_p: float = 0.01


@dataclass
class DefaultTEMWeights:
    """Default weight configuration for TEMLoss."""

    w_x_gen: float = 1.0
    w_x_inf: float = 1.0
    w_p_consistency: float = 1.0
    w_g_consistency: float = 1.0
    w_p_x: float = 1.0
    w_reg_g: float = 0.01
    w_reg_p: float = 0.01


# =============================================================================
# Base Types
# =============================================================================


@dataclass(frozen=True)
class LossOutput:
    """Structured output from loss computation.

    This unified output format enables:
    - Automatic logging of all components
    - Easy integration with PyTorch Lightning
    - Clear separation between optimization target and monitoring metrics

    Attributes:
        total: Total scalar loss for backpropagation
        components: Dictionary of individual loss values for logging
        metrics: Optional additional metrics (e.g., accuracy, correlation)

    Example:
        >>> output = loss_fn(predictions, targets)
        >>> output.total.backward()
        >>> for name, value in output.components.items():
        ...     logger.log(f'train/{name}', value)
    """

    total: Tensor
    components: Dict[str, Tensor]
    metrics: Optional[Dict[str, float]] = None


# =============================================================================
# Generative Pathway Loss
# =============================================================================


class GenerativeLoss(nn.Module):
    """Loss for the generative pathway: g → p → x.

    The generative pathway can be trained independently to learn:
    1. Transition dynamics: g_prev + action → g_gen
    2. Memory retrieval: g_gen → p_gen (via attractor)
    3. Sensory generation: p_gen → x_gen (via LEC decoder)

    This pathway does NOT require sensory input or inference, making it suitable
    for pre-training on simulated trajectories or reinforcement learning scenarios.

    Loss Components:
        L_x_gen: Cross-entropy between generated and true observations
        L_reg_g: L2 regularization on abstract location codes (sparsity)
        L_g: Optional consistency with inferred g (teacher forcing)

    Args:
        weights: Weight configuration (defaults to DefaultGenerativeWeights if None)

    Example:
        >>> # Using defaults
        >>> loss_fn = GenerativeLoss()
        >>>
        >>> # Using custom weights
        >>> loss_fn = GenerativeLoss(weights=DefaultGenerativeWeights(w_reg_g=0.001))
        >>>
        >>> output = loss_fn(gen_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[GenerativeLossWeights] = None):
        super().__init__()
        config = weights or DefaultGenerativeWeights()
        self.w_x_gen = config.w_x_gen
        self.w_reg_g = config.w_reg_g
        self.w_g = config.w_g

    def forward(
        self,
        outputs: GenerativeOutputs,
        x_target: Tensor,
        g_target: Optional[List[Tensor]] = None,
    ) -> LossOutput:
        """Compute generative pathway loss.

        Args:
            outputs: Generative pathway outputs (g, p, x)
            x_target: Ground truth observation [B, n_x] (one-hot)
            g_target: Optional inferred g for teacher forcing

        Returns:
            LossOutput with total loss and individual components
        """
        # L_x_gen: Sensory generation quality
        labels = torch.argmax(x_target, dim=1)
        L_x_gen = F.cross_entropy(outputs.x.logits[0], labels, reduction="mean")

        # L_reg_g: Abstract location regularization (L2)
        L_reg_g = sum((g**2).sum() / g.numel() for g in outputs.g)

        components = {"L_x_gen": L_x_gen.detach(), "L_reg_g": L_reg_g.detach()}

        # Base loss
        total = self.w_x_gen * L_x_gen + self.w_reg_g * L_reg_g

        # Optional: L_g for teacher forcing
        if g_target is not None:
            L_g = sum(F.mse_loss(outputs.g[f], g_target[f], reduction="mean") for f in range(len(outputs.g)))
            components["L_g"] = L_g.detach()
            total = total + self.w_g * L_g

        return LossOutput(total=total, components=components)


# =============================================================================
# Inference Pathway Loss
# =============================================================================


class InferenceLoss(nn.Module):
    """Loss for the inference pathway: x → p, g.

    The inference pathway can be trained independently to learn:
    1. Sensory encoding: x → x_f (LEC processing)
    2. Memory retrieval: x_ → p_x (sensory-based location)
    3. Abstract inference: p_x + transition → g_inf
    4. Grounded inference: g_, x_ → p_inf (hippocampal binding)

    This pathway does NOT require the generative pathway, making it suitable
    for supervised learning from observations or behavioral cloning.

    Loss Components:
        L_x_inf: Cross-entropy for sensory reconstruction from inferred p
        L_p_x: MSE between inferred p and sensory-retrieved p (optional)
        L_reg_p: L1 regularization on grounded location codes (sparsity)

    Args:
        weights: Weight configuration (defaults to DefaultInferenceWeights if None)

    Example:
        >>> # Using defaults
        >>> loss_fn = InferenceLoss()
        >>>
        >>> # Using custom weights
        >>> loss_fn = InferenceLoss(weights=DefaultInferenceWeights(w_reg_p=0.001))
        >>>
        >>> output = loss_fn(inf_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[InferenceLossWeights] = None):
        super().__init__()
        config = weights or DefaultInferenceWeights()
        self.w_x_inf = config.w_x_inf
        self.w_p_x = config.w_p_x
        self.w_reg_p = config.w_reg_p

    def forward(self, outputs: InferenceOutputs, x_target: Tensor) -> LossOutput:
        """Compute inference pathway loss.

        Args:
            outputs: Inference pathway outputs (g, p, x, optional p_x)
            x_target: Ground truth observation [B, n_x] (one-hot)

        Returns:
            LossOutput with total loss and individual components
        """
        # L_x_inf: Sensory reconstruction quality from inferred p
        labels = torch.argmax(x_target, dim=1)
        L_x_inf = F.cross_entropy(outputs.x.logits[0], labels, reduction="mean")

        # L_reg_p: Grounded location regularization (L1)
        L_reg_p = sum(p.abs().sum() / p.numel() for p in outputs.p)

        components = {"L_x_inf": L_x_inf.detach(), "L_reg_p": L_reg_p.detach()}

        # Base loss
        total = self.w_x_inf * L_x_inf + self.w_reg_p * L_reg_p

        # Optional: L_p_x for sensory-grounded consistency
        if outputs.p_x is not None:
            L_p_x = sum(F.mse_loss(outputs.p[f], outputs.p_x[f], reduction="mean") for f in range(len(outputs.p)))
            components["L_p_x"] = L_p_x.detach()
            total = total + self.w_p_x * L_p_x

        return LossOutput(total=total, components=components)


# =============================================================================
# Full TEM Training Loss
# =============================================================================


class TEMLoss(nn.Module):
    """Full TEM training loss combining generative and inference pathways.

    This loss enables complete TEM training with consistency constraints between
    pathways, implementing teacher forcing where inference guides generation.

    The key insight: during training, the generative pathway can use inferred
    values as targets, stabilizing learning through:
    1. Inference provides high-quality g_inf and p_inf
    2. Generative learns to match these via transition dynamics
    3. Consistency losses align the pathways

    Loss Components:
        L_x_gen: Generative sensory prediction quality
        L_x_inf: Inference sensory reconstruction quality
        L_p_consistency: MSE between inferred and generated p (teacher forcing)
        L_g_consistency: MSE between inferred and generated g (teacher forcing)
        L_p_x: Sensory-grounded consistency (optional)
        L_reg_g: Abstract location regularization
        L_reg_p: Grounded location regularization

    Args:
        weights: Weight configuration (defaults to DefaultTEMWeights if None)

    Example:
        >>> # Using defaults
        >>> loss_fn = TEMLoss()
        >>>
        >>> # Using custom weights
        >>> loss_fn = TEMLoss(weights=DefaultTEMWeights(w_reg_g=0.001, w_reg_p=0.001))
        >>>
        >>> output = loss_fn(gen_outputs, inf_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[TEMLossWeights] = None):
        super().__init__()
        config = weights or DefaultTEMWeights()
        self.w_x_gen = config.w_x_gen
        self.w_x_inf = config.w_x_inf
        self.w_p_consistency = config.w_p_consistency
        self.w_g_consistency = config.w_g_consistency
        self.w_p_x = config.w_p_x
        self.w_reg_g = config.w_reg_g
        self.w_reg_p = config.w_reg_p

    def forward(self, gen_outputs: GenerativeOutputs, inf_outputs: InferenceOutputs, x_target: Tensor) -> LossOutput:
        """Compute full TEM loss with pathway consistency.

        Args:
            gen_outputs: Generative pathway outputs (g, p, x)
            inf_outputs: Inference pathway outputs (g, p, x, optional p_x)
            x_target: Ground truth observation [B, n_x] (one-hot)

        Returns:
            LossOutput with total loss and all components
        """
        labels = torch.argmax(x_target, dim=1)

        # Reconstruction losses
        L_x_gen = F.cross_entropy(gen_outputs.x.logits[0], labels, reduction="mean")
        L_x_inf = F.cross_entropy(inf_outputs.x.logits[0], labels, reduction="mean")

        # Consistency losses (teacher forcing)
        L_p_consistency = sum(F.mse_loss(inf_outputs.p[f], gen_outputs.p[f], reduction="mean") for f in range(len(inf_outputs.p)))

        L_g_consistency = sum(F.mse_loss(inf_outputs.g[f], gen_outputs.g[f], reduction="mean") for f in range(len(inf_outputs.g)))

        # Regularization
        L_reg_g = sum((g**2).sum() / g.numel() for g in inf_outputs.g)
        L_reg_p = sum(p.abs().sum() / p.numel() for p in inf_outputs.p)

        components = {
            "L_x_gen": L_x_gen.detach(),
            "L_x_inf": L_x_inf.detach(),
            "L_p_consistency": L_p_consistency.detach(),
            "L_g_consistency": L_g_consistency.detach(),
            "L_reg_g": L_reg_g.detach(),
            "L_reg_p": L_reg_p.detach(),
        }

        # Base total
        total = (
            self.w_x_gen * L_x_gen
            + self.w_x_inf * L_x_inf
            + self.w_p_consistency * L_p_consistency
            + self.w_g_consistency * L_g_consistency
            + self.w_reg_g * L_reg_g
            + self.w_reg_p * L_reg_p
        )

        # Optional: L_p_x for sensory-grounded consistency
        if inf_outputs.p_x is not None:
            L_p_x = sum(F.mse_loss(inf_outputs.p[f], inf_outputs.p_x[f], reduction="mean") for f in range(len(inf_outputs.p)))
            components["L_p_x"] = L_p_x.detach()
            total = total + self.w_p_x * L_p_x

        return LossOutput(total=total, components=components)
