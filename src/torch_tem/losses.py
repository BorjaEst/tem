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
from typing import Dict, List, Optional, Protocol

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
    transition dynamics and memory retrieval, plus sensory predictions via decoder.

    All x predictions come from the generative decoder, using different p sources:
    - x: Main reconstruction from p_gen (g_gen → p_gen → x)
    - x_from_g: Reconstruction from p via g_inf (g_inf → p → x) for L_x_g
    - x_from_inf_p: Reconstruction from p_inf (p_inf → x) for L_x_p
    """

    g: List[Tensor]  # Abstract locations List[n_f] of [B, n_g[f]]
    p: List[Tensor]  # Grounded locations List[n_f] of [B, n_p[f]]
    x: SensoryPrediction  # Primary sensory prediction from p_gen
    x_from_g: Optional[SensoryPrediction] = None  # Reconstruction from g_inf → p → x
    x_from_inf_p: Optional[SensoryPrediction] = None  # Reconstruction from p_inf → x


class InferenceOutputs(Protocol):
    """Protocol defining outputs from the inference pathway.

    The inference pathway produces abstract and grounded locations through
    sensory encoding and belief propagation.

    Note: The inference pathway does NOT generate observations. It only produces
    location codes (g, p) from sensory input. All observation reconstruction happens
    in the generative pathway via the decoder.
    """

    g: List[Tensor]  # Inferred abstract locations List[n_f] of [B, n_g[f]]
    p: List[Tensor]  # Inferred grounded locations List[n_f] of [B, n_p[f]]
    p_x: Optional[List[Tensor]] = None  # Optional sensory-retrieved locations


class GenerativeLossWeights(Protocol):
    """Protocol defining weight configuration for GenerativeLoss."""

    w_x_gen: float  # Weight for sensory generation quality
    w_reg_g: float  # Weight for abstract location regularization
    w_g: float  # Weight for teacher forcing consistency


class InferenceLossWeights(Protocol):
    """Protocol defining weight configuration for InferenceLoss."""

    w_p_x: float  # Weight for sensory-grounded consistency
    w_reg_p: float  # Weight for grounded location regularization


class TEMLossWeights(Protocol):
    """Protocol defining weight configuration for TEMLoss."""

    w_x_gen: float  # Weight for primary reconstruction (g_gen → p_gen → x)
    w_x_g: float  # Weight for reconstruction from g_inf (g_inf → p → x)
    w_x_p: float  # Weight for reconstruction from p_inf (p_inf → x)
    w_p_consistency: float  # Weight for grounded consistency
    w_g_consistency: float  # Weight for abstract consistency
    w_p_x: float  # Weight for sensory-grounded consistency (optional)
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

    w_p_x: float = 1.0
    w_reg_p: float = 0.01


@dataclass
class DefaultTEMWeights:
    """Default weight configuration for TEMLoss."""

    w_x_gen: float = 1.0
    w_x_g: float = 1.0
    w_x_p: float = 1.0
    w_p_consistency: float = 1.0
    w_g_consistency: float = 1.0
    w_p_x: float = 1.0
    w_reg_g: float = 0.01
    w_reg_p: float = 0.01


# =============================================================================
# Base Types
# =============================================================================


@dataclass(frozen=True)
class PathwayMetrics:
    """Structured metrics for pathway quality assessment.

    Attributes:
        p_agreement: Cosine similarity between generative and inference grounded locations
        g_agreement: Cosine similarity between generative and inference abstract locations
        gen_reconstruction_accuracy: Percentage of correct predictions from generative pathway
        inf_reconstruction_accuracy: Percentage of correct predictions from inference pathway
    """

    p_agreement: float
    g_agreement: float
    gen_reconstruction_accuracy: float
    inf_reconstruction_accuracy: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging frameworks."""
        return {
            "p_agreement": self.p_agreement,
            "g_agreement": self.g_agreement,
            "gen_reconstruction_accuracy": self.gen_reconstruction_accuracy,
            "inf_reconstruction_accuracy": self.inf_reconstruction_accuracy,
        }


@dataclass(frozen=True)
class LossOutput:
    """Structured output from loss computation.

    This unified output format enables:
    - Automatic logging of all components
    - Easy integration with PyTorch Lightning
    - Clear separation between optimization target and monitoring metrics
    - Gradient monitoring for debugging training dynamics

    Attributes:
        total: Total scalar loss for backpropagation
        components: Dictionary of individual loss values for logging
        metrics: Optional additional metrics (e.g., accuracy, correlation)
        grad_norms: Optional gradient norms per component for debugging

    Example:
        >>> output = loss_fn(predictions, targets)
        >>> output.total.backward()
        >>> for name, value in output.components.items():
        ...     logger.log(f'train/{name}', value)
        >>> if output.metrics:
        ...     for name, value in output.metrics.items():
        ...         logger.log(f'metrics/{name}', value)
    """

    total: Tensor
    components: Dict[str, Tensor]
    metrics: Optional[Dict[str, float]] = None
    grad_norms: Optional[Dict[str, float]] = None


# =============================================================================
# Base Loss Class
# =============================================================================


class BaseTEMLoss(nn.Module):
    """Base class providing common functionality for TEM loss modules.

    Extracts shared logic for:
    - Weight management with annealing
    - Component freezing for staged training
    - Metrics computation control

    Subclasses should implement the forward method.
    """

    def __init__(self, annealing_factor: float = 1.0, frozen_components: Optional[set] = None, compute_metrics: bool = False):
        """Initialize base loss configuration.

        Args:
            annealing_factor: Multiplicative factor for regularization (0.0-1.0)
            frozen_components: Set of component names to exclude from loss
            compute_metrics: Whether to compute diagnostic metrics
        """
        super().__init__()
        self.annealing_factor = annealing_factor
        self.frozen_components = frozen_components or set()
        self.compute_metrics = compute_metrics

    def _get_effective_weight(self, component_name: str, base_weight: float, is_regularization: bool = False) -> float:
        """Compute effective weight considering freezing and annealing.

        Args:
            component_name: Name of the loss component
            base_weight: Base weight from configuration
            is_regularization: Whether this is a regularization term

        Returns:
            Effective weight (0.0 if frozen, annealed if regularization)
        """
        if component_name in self.frozen_components:
            return 0.0
        if is_regularization:
            return base_weight * self.annealing_factor
        return base_weight


# =============================================================================
# Generative Pathway Loss
# =============================================================================


class GenerativeLoss(BaseTEMLoss):
    """Loss for the generative pathway: g → p → x.

    The generative pathway can be trained independently to learn:
    1. Transition dynamics: g_prev + action → g_gen
    2. Memory retrieval: g_gen → p_gen (via attractor)
    3. Sensory generation: p_gen → x_gen (via LEC decoder)

    This pathway does NOT require sensory input or inference, making it suitable
    for pre-training on simulated trajectories or reinforcement learning scenarios.

    Loss Components (ELBO terms):
        L_x_gen: -log p(x|p) - Cross-entropy between generated and true observations
        L_reg_g: ||g||² - L2 regularization on abstract location codes (sparsity)
        L_g: MSE with inferred g - Optional consistency with inferred g (teacher forcing)

    Args:
        weights: Weight configuration (defaults to DefaultGenerativeWeights if None)
        annealing_factor: Multiplicative factor for regularization (curriculum learning)
        compute_metrics: Whether to compute diagnostic metrics (adds overhead)

    Example:
        >>> # Using defaults
        >>> loss_fn = GenerativeLoss()
        >>>
        >>> # Using custom weights with annealing
        >>> loss_fn = GenerativeLoss(
        ...     weights=DefaultGenerativeWeights(w_reg_g=0.001),
        ...     annealing_factor=0.5,
        ... )
        >>>
        >>> output = loss_fn(gen_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[GenerativeLossWeights] = None, annealing_factor: float = 1.0, compute_metrics: bool = False):
        super().__init__(annealing_factor, None, compute_metrics)
        config = weights or DefaultGenerativeWeights()
        self.w_x_gen = config.w_x_gen
        self.w_reg_g = config.w_reg_g
        self.w_g = config.w_g

    def forward(self, outputs: GenerativeOutputs, x_target: Tensor, g_target: Optional[List[Tensor]] = None) -> LossOutput:
        """Compute generative pathway loss.

        Args:
            outputs: Generative pathway outputs (g, p, x)
            x_target: Ground truth observation [B, n_x] (one-hot)
            g_target: Optional inferred g for teacher forcing

        Returns:
            LossOutput with total loss and individual components
        """
        # L_x_gen: Sensory generation quality (ELBO: -log p(x|p))
        L_x_gen = _compute_reconstruction_loss(outputs.x, x_target)

        # L_reg_g: Abstract location regularization (ELBO: approximate KL to prior)
        L_reg_g = _compute_l2_regularization(outputs.g)

        components = {"L_x_gen": L_x_gen.detach(), "L_reg_g": L_reg_g.detach()}

        # Base loss with annealing applied via base class method
        total = self._get_effective_weight("L_x_gen", self.w_x_gen) * L_x_gen + self._get_effective_weight("L_reg_g", self.w_reg_g, is_regularization=True) * L_reg_g

        # Optional: L_g for teacher forcing
        if g_target is not None:
            L_g = _compute_consistency_loss(outputs.g, g_target)
            components["L_g"] = L_g.detach()
            total = total + self._get_effective_weight("L_g", self.w_g) * L_g

        # Compute metrics if requested
        metrics = None
        if self.compute_metrics and g_target is not None:
            labels = torch.argmax(x_target, dim=1)
            gen_preds = torch.argmax(outputs.x.logits[0], dim=1)
            gen_accuracy = (gen_preds == labels).float().mean().item()
            # Vectorized cosine similarity computation
            g_similarities = torch.stack([F.cosine_similarity(outputs.g[f], g_target[f], dim=1).mean() for f in range(len(outputs.g))])
            g_agreement = g_similarities.mean().item()
            metrics = {"gen_reconstruction_accuracy": gen_accuracy, "g_agreement": g_agreement}

        return LossOutput(total=total, components=components, metrics=metrics)


# =============================================================================
# Inference Pathway Loss
# =============================================================================


class InferenceLoss(BaseTEMLoss):
    """Loss for the inference pathway: x → p, g.

    The inference pathway can be trained independently to learn:
    1. Sensory encoding: x → x_f (LEC processing)
    2. Memory retrieval: x_ → p_x (sensory-based location)
    3. Abstract inference: p_x + transition → g_inf
    4. Grounded inference: g_, x_ → p_inf (hippocampal binding)

    This pathway does NOT require the generative pathway, making it suitable
    for supervised learning from observations or behavioral cloning.

    Loss Components (ELBO KL terms only):
        L_p_x: MSE between f(p|g,x) and f(p_x|x) - Sensory-grounded consistency (optional)
        L_reg_p: ||p||₁ - L1 regularization on grounded location codes (sparsity)

    Note: Reconstruction losses (L_x) are computed in GenerativeLoss, not here.
    The inference pathway only computes KL divergence terms (consistency losses).

    Args:
        weights: Weight configuration (defaults to DefaultInferenceWeights if None)
        annealing_factor: Multiplicative factor for regularization (curriculum learning)
        compute_metrics: Whether to compute diagnostic metrics (adds overhead)

    Example:
        >>> # Using defaults
        >>> loss_fn = InferenceLoss()
        >>>
        >>> # Using custom weights with annealing
        >>> loss_fn = InferenceLoss(
        ...     weights=DefaultInferenceWeights(w_reg_p=0.001),
        ...     annealing_factor=0.5,
        ... )
        >>>
        >>> output = loss_fn(inf_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[InferenceLossWeights] = None, annealing_factor: float = 1.0, compute_metrics: bool = False):
        super().__init__(annealing_factor, None, compute_metrics)
        config = weights or DefaultInferenceWeights()
        self.w_p_x = config.w_p_x
        self.w_reg_p = config.w_reg_p

    def forward(self, outputs: InferenceOutputs) -> LossOutput:
        """Compute inference pathway loss (KL divergences only).

        Args:
            outputs: Inference pathway outputs (g, p, optional p_x)

        Returns:
            LossOutput with total loss and individual components
        """
        # L_reg_p: Grounded location regularization (ELBO: approximate KL to prior)
        L_reg_p = _compute_l1_regularization(outputs.p)

        components = {"L_reg_p": L_reg_p.detach()}

        # Base loss with annealing applied via base class method
        total = self._get_effective_weight("L_reg_p", self.w_reg_p, is_regularization=True) * L_reg_p

        # Optional: L_p_x for sensory-grounded consistency
        if outputs.p_x is not None:
            L_p_x = _compute_consistency_loss(outputs.p, outputs.p_x)
            components["L_p_x"] = L_p_x.detach()
            total = total + self._get_effective_weight("L_p_x", self.w_p_x) * L_p_x

        return LossOutput(total=total, components=components, metrics=None)


# =============================================================================
# Full TEM Training Loss
# =============================================================================


class TEMLoss(BaseTEMLoss):
    """Full TEM training loss combining generative and inference pathways.

    This loss enables complete TEM training with consistency constraints between
    pathways, implementing teacher forcing where inference guides generation.

    The ELBO (Evidence Lower Bound) following Gemici et al. (2017):
        log p(x|a) ≥ E_q[log p(x|p)] - KL[q(g|a) || f(g|x)] - KL[q(p|g) || f(p|x)]

    Where:
        - p(x|p): Generative model (decoder)
        - q(g|a): Transition model (path integration)
        - q(p|g): Memory retrieval (attractor dynamics)
        - f(g|x): Abstract inference (sensory → abstract)
        - f(p|x): Grounded inference (sensory → grounded)

    Loss Components (ELBO terms):
        L_x_gen: -log p(x|p_gen) - Primary reconstruction (g_gen → p_gen → x)
        L_x_g: -log p(x|p) where p from g_inf - Reconstruction from inferred g
        L_x_p: -log p(x|p_inf) - Reconstruction from inferred p
        L_p_consistency: MSE between q(p|g_gen) and f(p|x) - Grounded KL
        L_g_consistency: MSE between q(g|a) and f(g|x) - Abstract KL
        L_p_x: MSE between f(p|x) and f(p|g_inf,x) - Sensory-grounded consistency (optional)
        L_reg_g: ||g||² - Abstract location sparsity (approximate KL to prior)
        L_reg_p: ||p||₁ - Grounded location sparsity (approximate KL to prior)

    Args:
        weights: Weight configuration (defaults to DefaultTEMWeights if None)
        annealing_factor: Multiplicative factor for regularization (curriculum learning)
        frozen_components: Set of component names to freeze (zero weight)
        compute_metrics: Whether to compute diagnostic metrics (adds overhead)

    Example:
        >>> # Using defaults
        >>> loss_fn = TEMLoss()
        >>>
        >>> # Using custom configuration
        >>> loss_fn = TEMLoss(
        ...     weights=DefaultTEMWeights(w_reg_g=0.001, w_reg_p=0.001),
        ...     annealing_factor=0.5,  # Reduce regularization early in training
        ...     frozen_components={"L_x_gen"},  # Freeze generative reconstruction
        ...     compute_metrics=True,  # Enable pathway agreement metrics
        ... )
        >>>
        >>> output = loss_fn(gen_outputs, inf_outputs, x_target)
        >>> output.total.backward()
    """

    def __init__(self, weights: Optional[TEMLossWeights] = None, annealing_factor: float = 1.0, frozen_components: Optional[set] = None, compute_metrics: bool = False):
        super().__init__(annealing_factor, frozen_components, compute_metrics)
        config = weights or DefaultTEMWeights()
        self.w_x_gen = config.w_x_gen
        self.w_x_g = config.w_x_g
        self.w_x_p = config.w_x_p
        self.w_p_consistency = config.w_p_consistency
        self.w_g_consistency = config.w_g_consistency
        self.w_p_x = config.w_p_x
        self.w_reg_g = config.w_reg_g
        self.w_reg_p = config.w_reg_p

    def forward(self, gen_outputs: GenerativeOutputs, inf_outputs: InferenceOutputs, x_target: Tensor) -> LossOutput:
        """Compute full TEM loss with pathway consistency.

        Args:
            gen_outputs: Generative pathway outputs (g, p, x, x_from_g, x_from_inf_p)
            inf_outputs: Inference pathway outputs (g, p, optional p_x)
            x_target: Ground truth observation [B, n_x] (one-hot)

        Returns:
            LossOutput with total loss, components, and optional metrics
        """
        # Reconstruction losses (ELBO: -log p(x|p) terms - all from generative decoder)
        L_x_gen = _compute_reconstruction_loss(gen_outputs.x, x_target)

        # Consistency losses (ELBO: KL divergence approximations via MSE)
        L_p_consistency = _compute_consistency_loss(inf_outputs.p, gen_outputs.p)
        L_g_consistency = _compute_consistency_loss(inf_outputs.g, gen_outputs.g)

        # Regularization (ELBO: approximate KL to prior)
        L_reg_g = _compute_l2_regularization(inf_outputs.g)
        L_reg_p = _compute_l1_regularization(inf_outputs.p)

        # Build components dict
        components = {
            "L_x_gen": L_x_gen.detach(),
            "L_p_consistency": L_p_consistency.detach(),
            "L_g_consistency": L_g_consistency.detach(),
            "L_reg_g": L_reg_g.detach(),
            "L_reg_p": L_reg_p.detach(),
        }

        # Base total using base class method for weight management
        total = (
            self._get_effective_weight("L_x_gen", self.w_x_gen) * L_x_gen
            + self._get_effective_weight("L_p_consistency", self.w_p_consistency) * L_p_consistency
            + self._get_effective_weight("L_g_consistency", self.w_g_consistency) * L_g_consistency
            + self._get_effective_weight("L_reg_g", self.w_reg_g, is_regularization=True) * L_reg_g
            + self._get_effective_weight("L_reg_p", self.w_reg_p, is_regularization=True) * L_reg_p
        )

        # L_x_g: Reconstruction from g_inf → p → x (legacy parity)
        if gen_outputs.x_from_g is not None:
            L_x_g = _compute_reconstruction_loss(gen_outputs.x_from_g, x_target)
            components["L_x_g"] = L_x_g.detach()
            total = total + self._get_effective_weight("L_x_g", self.w_x_g) * L_x_g

        # L_x_p: Reconstruction from p_inf → x (legacy parity)
        if gen_outputs.x_from_inf_p is not None:
            L_x_p = _compute_reconstruction_loss(gen_outputs.x_from_inf_p, x_target)
            components["L_x_p"] = L_x_p.detach()
            total = total + self._get_effective_weight("L_x_p", self.w_x_p) * L_x_p

        # Optional: L_p_x for sensory-grounded consistency
        if inf_outputs.p_x is not None:
            L_p_x = _compute_consistency_loss(inf_outputs.p, inf_outputs.p_x)
            components["L_p_x"] = L_p_x.detach()
            total = total + self._get_effective_weight("L_p_x", self.w_p_x) * L_p_x

        # Compute metrics if requested using helper function
        metrics_dict = _compute_metrics(gen_outputs, inf_outputs, x_target) if self.compute_metrics else None

        return LossOutput(total=total, components=components, metrics=metrics_dict)


# =============================================================================
# Metrics Computation
# =============================================================================


def _compute_metrics(gen_outputs: GenerativeOutputs, inf_outputs: InferenceOutputs, x_target: Tensor) -> Dict[str, float]:
    """Compute diagnostic metrics for pathway quality assessment.

    Computes agreement between pathways and reconstruction accuracy to monitor
    training progress and detect pathway divergence.

    Args:
        gen_outputs: Generative pathway outputs with p, g, and x predictions
        inf_outputs: Inference pathway outputs with p and g (no x)
        x_target: Ground truth observations for accuracy computation

    Returns:
        Dictionary containing:
            - p_agreement: Cosine similarity between generative and inference grounded codes
            - g_agreement: Cosine similarity between generative and inference abstract codes
            - gen_reconstruction_accuracy: Accuracy of generative pathway predictions
    """
    labels = torch.argmax(x_target, dim=1)
    gen_preds = torch.argmax(gen_outputs.x.logits[0], dim=1)

    gen_accuracy = (gen_preds == labels).float().mean().item()

    # Compute mean cosine similarity across all frequency scales using vectorized operations
    # Stack all frequencies and compute cosine similarity in batch
    p_similarities = torch.stack([F.cosine_similarity(gen_outputs.p[f], inf_outputs.p[f], dim=1).mean() for f in range(len(gen_outputs.p))])
    p_agreement = p_similarities.mean().item()

    g_similarities = torch.stack([F.cosine_similarity(gen_outputs.g[f], inf_outputs.g[f], dim=1).mean() for f in range(len(gen_outputs.g))])
    g_agreement = g_similarities.mean().item()

    # Create structured metrics object (inf_reconstruction_accuracy removed)
    metrics = PathwayMetrics(
        p_agreement=p_agreement,
        g_agreement=g_agreement,
        gen_reconstruction_accuracy=gen_accuracy,
        inf_reconstruction_accuracy=0.0,  # Not applicable - inference doesn't reconstruct
    )

    # Return as dict for backward compatibility with LossOutput
    return metrics.to_dict()


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_reconstruction_loss(predictions: SensoryPrediction, targets: Tensor, logits_index: int = 0, reduction: str = "mean") -> Tensor:
    """Compute cross-entropy reconstruction loss.

    Args:
        predictions: Sensory predictions with logits
        targets: Ground truth observation (one-hot encoded)
        logits_index: Index of logits tensor to use
        reduction: Reduction strategy ('mean', 'sum', 'none')

    Returns:
        Scalar loss tensor
    """
    labels = torch.argmax(targets, dim=1)
    return F.cross_entropy(predictions.logits[logits_index], labels, reduction=reduction)


def _compute_l2_regularization(codes: List[Tensor]) -> Tensor:
    """Compute L2 (squared) regularization over multi-scale codes.

    Args:
        codes: List of tensors representing multi-scale location codes

    Returns:
        Mean squared magnitude across all codes (averaged per frequency)
    """
    # Compute mean per frequency (since n_g[f] varies), then average across frequencies
    return torch.stack([c.pow(2).mean() for c in codes]).mean()


def _compute_l1_regularization(codes: List[Tensor]) -> Tensor:
    """Compute L1 (absolute) regularization over multi-scale codes.

    Args:
        codes: List of tensors representing multi-scale location codes

    Returns:
        Mean absolute magnitude across all codes (averaged per frequency)
    """
    # Compute mean per frequency (since n_p[f] varies), then average across frequencies
    return torch.stack([c.abs().mean() for c in codes]).mean()


def _compute_consistency_loss(codes1: List[Tensor], codes2: List[Tensor], reduction: str = "mean") -> Tensor:
    """Compute MSE consistency loss between two multi-scale codes.

    Args:
        codes1: First multi-scale code
        codes2: Second multi-scale code
        reduction: Reduction strategy

    Returns:
        Mean consistency loss across all frequencies
    """
    # Compute MSE per frequency (since n[f] varies), then aggregate across frequencies
    per_freq_mse = torch.stack([F.mse_loss(c1, c2) for c1, c2 in zip(codes1, codes2)])
    return per_freq_mse.mean() if reduction == "mean" else per_freq_mse.sum()
