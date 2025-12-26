"""Observation decoder for torch_tem package.

This module implements the final generative step from grounded locations (place cells)
to sensory observations. It completes the generative pathway: g → p → x, allowing
the model to predict what sensory input should be experienced given an internal
spatial representation.

The Decoder performs p→x decoding using learned sensory processing
parameters (w_x, b_x) and tiling matrices (W_tile).
"""

from typing import List, Protocol

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from torch_tem import utils
from torch_tem.core.mlp import MLP
from torch_tem.types import Matrix, MultiScaleCode, SensoryPrediction


class DecoderConfig(BaseModel):
    """Decoder configuration parameters."""

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    hidden_multiplier: int = Field(default=20, ge=1, frozen=True, description="Hidden dim = hidden_multiplier * n_x_c")
    activation: str = Field(default="elu", frozen=True, description="Activation function name")
    use_bias: bool = Field(default=True, frozen=True, description="Use bias in MLP layers")


class DecoderContext(Protocol):
    """Protocol defining required parameters for Decoder initialization."""

    n_x: int  # Number of sensory observation neurons x
    W_tile: List[Matrix]  # Tiling matrices for each frequency


class Decoder(nn.Module):
    """Decodes grounded location (place cells) to sensory predictions.

    Implements the generative pathway: p → x̂

    The decoder performs:
    1. Untiling: Project place cells to compressed sensory (p → x_c)
    2. Scaling: Apply learned weights and biases (w_x, b_x)
    3. Decoding: MLP expansion to full observation space (x_c → x̂)

    Args:
        n_x: Number of sensory observation neurons
        W_tile: Tiling matrices shared from parent LECModel
        config: Decoder configuration parameters
    """

    def __init__(self, n_x: int, W_tile: List[Matrix], config: DecoderConfig):
        """Initialize Decoder module.

        Args:
            n_x: Number of sensory observation neurons.
            W_tile: Tiling matrices for projection (managed by parent LECModel).
            config: Decoder configuration parameters.
        """
        super().__init__()
        self._config = config
        self._W_tile = W_tile
        self._n_x = n_x

        # Learnable sensory decoding parameters
        self._w_x = nn.Parameter(torch.ones(1, self.n_x_c))
        self._b_x = nn.Parameter(torch.zeros(1, self.n_x_c))

        # MLP decoder from compressed sensory to full observation
        activation_fn = utils.get_activation_function(config.activation.lower())
        hidden_dim = config.hidden_multiplier * self.n_x_c  # Hidden layer size
        bias = (True, True) if config.use_bias else (False, False)
        self._mlp_decoder = MLP(self.n_x_c, n_x, (activation_fn, None), hidden_dim, bias)

    @property
    def n_x(self) -> int:
        """Number of sensory observation neurons x."""
        return self._n_x

    @property
    def n_x_c(self) -> int:
        """Number of compressed sensory neurons x_c."""
        return self._W_tile[0].size(0)

    def untiling(self, p: MultiScaleCode) -> Tensor:
        """Untile grounded locations to compressed sensory space.

        Projects place cell activations back to compressed sensory representation
        by inverting the tiling operation: x_c = p @ W_tile^T

        Args:
            p: Grounded locations (place cells) List[n_f] of (batch, n_p[f]).

        Returns:
            Compressed sensory representation (batch, n_x_c).

        Note:
            We only untile the highest frequency for decoding, as it contains
            the most detailed sensory information to reduce computation costs.
        """
        return torch.matmul(p[0], self._W_tile[0].t())

    def decode(self, x: Tensor) -> SensoryPrediction:
        """Decode compressed sensory to full observation predictions.

        Args:
            x: Compressed sensory representation (batch, n_x_c).

        Returns:
            Sensory prediction with observation probabilities and logits.
        """
        x_logits = self._mlp_decoder(x)
        x_probs = torch.nn.functional.softmax(x_logits, dim=-1)
        return SensoryPrediction(values=[x_probs], logits=[x_logits])

    def forward(self, p: MultiScaleCode) -> SensoryPrediction:
        """Decode grounded locations to sensory predictions.

        Complete generative pathway: p → x_c → x̂

        Args:
            p: Grounded locations (place cells) List[n_f] of (batch, n_p[f]).

        Returns:
            Sensory prediction with observation probabilities and logits.
        """
        x_proj = self.untiling(p)
        return self.decode(self._w_x * x_proj + self._b_x)


__all__ = ["Decoder", "DecoderConfig"]


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Decoder usage example: Place cells to sensory predictions.

    Demonstrates how the decoder transforms grounded locations (place cells)
    back into sensory observation predictions via untiling and MLP decoding.
    """
    print("=" * 80)
    print("Decoder Example - Generative Pathway (p → x̂)")
    print("=" * 80)

    # Configuration
    n_x = 45  # Observation space size
    n_x_c = 10  # Compressed dimension
    n_p = [96, 80, 64]  # Place cells per frequency
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Observation space: {n_x}")
    print(f"  Compressed dimension: {n_x_c}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Batch size: {batch_size}")

    # Create tiling matrices (normally from context)
    W_tile = [torch.randn(n_x_c, n_p_f) for n_p_f in n_p]
    print(f"\n✓ Tiling matrices: {[W.shape for W in W_tile]}")

    # Create decoder
    config = DecoderConfig()
    decoder = Decoder(n_x, W_tile, config)
    print(f"✓ Decoder initialized (n_x={decoder.n_x}, n_x_c={decoder.n_x_c})")

    # Create grounded locations (place cells)
    p = [torch.randn(batch_size, n_p_f) for n_p_f in n_p]
    print(f"✓ Grounded locations: {[p_f.shape for p_f in p]}")

    # Decode to sensory predictions
    with torch.no_grad():
        prediction = decoder(p)

    print(f"\n✓ Sensory prediction:")
    print(f"  Probabilities shape: {prediction.values[0].shape}")
    print(f"  Logits shape: {prediction.logits[0].shape}")
    print(f"  Probabilities sum to 1: {torch.allclose(prediction.values[0].sum(dim=-1), torch.ones(batch_size))}")

    # Show top predictions
    print(f"\nTop-3 predicted observations per sample:")
    for b in range(min(2, batch_size)):
        top_k = torch.topk(prediction.values[0][b], k=3)
        print(f"  Sample {b}: indices={top_k.indices.tolist()}, probs={top_k.values.tolist()}")

    print("\n" + "=" * 80)
