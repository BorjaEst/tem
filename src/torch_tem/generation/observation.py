"""Observation generator for torch_tem package.

This module implements the final generative step from grounded locations (place cells)
to sensory observations. It completes the generative pathway: g → p → x, allowing
the model to predict what sensory input should be experienced given an internal
spatial representation.

The ObservationGenerator provides a clean interface to the ObservationDecoder,
making the generative path explicit in the TEM architecture. This separation allows
for modular testing and clear information flow in the model.
"""

from typing import List, Tuple

import torch.nn as nn
from torch import Tensor

from torch_tem.core.decoder import ObservationDecoder


class ObservationGenerator(nn.Module):
    """Generates sensory observations (x) from grounded locations (p).

    This component implements the p→x generative pathway in the TEM architecture,
    decoding hippocampal place cell representations into predicted sensory
    observations. It wraps the ObservationDecoder with clear generative semantics.

    The generator uses only the highest-frequency place cell module (p[0]) for
    observation prediction, as this module has the finest spatial resolution and
    contains the most detailed location information.

    Architecture:
        - Thin wrapper around ObservationDecoder
        - Uses only p[0] (highest frequency) for decoding
        - Returns both softmax probabilities and raw logits
        - No learnable parameters (all in decoder)

    Attributes:
        decoder: ObservationDecoder that performs p[0]→x transformation
    """

    def __init__(self, decoder: ObservationDecoder):
        """Initialize observation generator.

        Simple wrapper initialization that stores the decoder component.

        Args:
            decoder: ObservationDecoder instance configured with appropriate
                    dimensions (n_x, n_x_c, n_x_f). The decoder handles the
                    actual p[0]→x transformation.

        Note:
            Unlike LocationGenerator, this component has no configurable parameters
            as it's a pure wrapper. All parameters reside in the decoder.
        """
        super().__init__()
        self.decoder = decoder

    def generate(self, p: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Generate sensory observation from grounded location.

        Decodes the highest-frequency place cell representation into a distribution
        over possible sensory observations. Returns both normalized probabilities
        (for sampling/prediction) and raw logits (for loss computation).

        Args:
            p: Grounded location as list of [n_f] tensors, each with shape [B, n_p[f]].
               Only p[0] (highest frequency module) is used for decoding.

        Returns:
            Tuple of (x_probs, x_logits):
                - x_probs: Softmax probabilities over observations, shape [B, n_x].
                          Sum to 1.0 across n_x dimension.
                - x_logits: Raw unnormalized logits, shape [B, n_x].
                           Used for computing cross-entropy loss.

        Note:
            Using only p[0] assumes the finest-scale place cells contain sufficient
            information to uniquely identify observations. Lower-frequency modules
            (p[1], p[2], ...) represent more abstract spatial patterns.
        """
        # Delegate to decoder (uses only p[0])
        return self.decoder(p)

    def forward(self, p: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward pass (alias for generate).

        PyTorch convention: defines the computation performed at every call.
        Simply delegates to generate() for consistent interface.

        Args:
            p: Grounded location (place cells)

        Returns:
            Tuple of (x_probs, x_logits):
                - x_probs: Observation probabilities [B, n_x]
                - x_logits: Observation logits [B, n_x]
        """
        return self.generate(p)


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """
    Example usage of ObservationGenerator for p→x generative decoding.

    This demonstrates how to:
    1. Set up the observation decoder with proper configuration
    2. Create the observation generator wrapper
    3. Generate sensory predictions from place cell activity
    4. Use both probability and logit outputs
    """
    import torch

    from torch_tem.config.parameters import Parameters
    from torch_tem.core.decoder import ObservationDecoder

    # Create configuration
    params = Parameters(
        n_x=45,  # Number of unique observations
        n_x_c=10,  # Compressed observation dimension
        n_g_subsampled=[10, 10, 8, 6, 6],  # Grid cell dimensions per frequency
    )

    # Create decoder component
    # (In practice, this would be created by TEM orchestrator)
    decoder = ObservationDecoder(params)

    # Create observation generator
    generator = ObservationGenerator(decoder)

    # Example: Generate observations from grounded locations
    batch_size = 4
    p = [torch.randn(batch_size, n_p) for n_p in params.n_p]

    # Generate observation predictions
    x_probs, x_logits = generator.generate(p)

    print(f"Input: {len(p)} place cell modules")
    print(f"  p[0] shape: {p[0].shape} (highest frequency, used for decoding)")
    print(f"\nOutput:")
    print(f"  x_probs: {x_probs.shape} (probabilities sum to 1.0)")
    print(f"  x_logits: {x_logits.shape} (raw logits for loss)")
    print(f"  Probability sum per batch: {x_probs.sum(dim=1)}")

    # Sample from predicted distribution
    x_sample = torch.multinomial(x_probs, num_samples=1)
    print(f"\nSampled observations: {x_sample.squeeze()}")

    # Using logits for loss computation
    target_x = torch.randint(0, params.n_x, (batch_size,))
    loss = torch.nn.functional.cross_entropy(x_logits, target_x)
    print(f"\nExample cross-entropy loss: {loss.item():.4f}")

    # Demonstrate forward() method (PyTorch convention)
    x_probs_fwd, x_logits_fwd = generator(p)
    print(f"\nforward() produces same output: {torch.allclose(x_probs, x_probs_fwd)}")
