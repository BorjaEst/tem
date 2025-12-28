"""Lateral Entorhinal Cortex (LEC) module for sensory processing.

This module implements the LEC pathway for sensory observation processing through
multi-stage transformation:
    - Encoding: o → o_c (one-hot to two-hot compression)
    - Processing: o_c → x (temporal filtering at multiple frequencies)
    - Projection: x → x_ (tiling to hippocampal input space)
    - Decoding: p → x (place cells to sensory predictions)

The LEC provides sensory context that combines with spatial information from MEC
to form conjunctive hippocampal representations (place cells).

Typical usage example:
    >>> config = LECConfig(learn_W_tile=False)
    >>> context = ... # LECContext with n_o, f_initial, W_tile
    >>> lec = LECModel(context, config)
    >>> state = lec.init_state(batch_size=4, device=torch.device('cpu'))
    >>> state = lec.forward(observation, state)
    >>> prediction = lec.decode(place_cells)
"""

from dataclasses import dataclass
from typing import List, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from torch_tem.core.lec.decoder import Decoder, DecoderConfig
from torch_tem.core.lec.encoder import Encoder, EncoderConfig
from torch_tem.core.lec.processor import Processor, ProcessorConfig
from torch_tem.core.lec.projection import Projection, ProjectionConfig
from torch_tem.types import Matrix, MultiScaleCode, Observation, SensoryPrediction

__all__ = ["LECConfig", "LECState", "LECModel"]


class LECConfig(BaseModel):
    """LEC model configuration parameters."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # Learning tiling matrices
    learn_W_tile: bool = Field(default=False, description="If True, tiling matrices W_tile are learnable")

    # Submodule configurations
    encoder: EncoderConfig = Field(default_factory=EncoderConfig, description="Encoder configuration")
    decoder: DecoderConfig = Field(default_factory=DecoderConfig, description="Decoder configuration")
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig, description="Processor configuration")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig, description="Projection configuration")


class LECContext(Protocol):
    """Protocol for LEC model initialization parameters.

    Attributes:
        n_o: Number of sensory observation neurons
        f_initial: Initial frequency values for temporal filtering
        W_tile: Tiling matrices for projection, one per frequency module
    """

    n_o: int
    f_initial: List[float]
    W_tile: List[Matrix]


@dataclass(frozen=True)
class LECState:
    """State container for LEC pathway.

    Attributes:
        compressed_observation: Two-hot compressed sensory representation.
        filtered_observation: Temporally filtered sensory representation.
        projection: Projected sensory to hippocampal input space.
    """

    compressed_observation: MultiScaleCode
    filtered_observation: MultiScaleCode
    projection: MultiScaleCode

    def detach(self) -> "LECState":
        """Detach all tensors in the state from the computation graph.

        Returns:
            New LECState with all tensors detached from gradients.
        """
        return LECState(
            compressed_observation=[x.detach() for x in self.compressed_observation],
            filtered_observation=[x.detach() for x in self.filtered_observation],
            projection=[x.detach() for x in self.projection],
        )


class LECModel(nn.Module):
    """Lateral Entorhinal Cortex (LEC) pathway for sensory processing.

    Implements sensory observation processing through:
    - Encoding: o → o_c (one-hot to two-hot compression)
    - Processing: o_c → x (temporal filtering at multiple frequencies)
    - Projection: x → x_ (tiling to hippocampal input space)
    - Decoding: p → x (place cells to sensory predictions)
    """

    def __init__(self, context: LECContext, config: LECConfig):
        """Initialize LEC model."""
        super().__init__()
        self._config = config

        # Register W_tile matrices as parameters or buffers based on config
        p = [nn.Parameter(matrix, requires_grad=config.learn_W_tile) for matrix in context.W_tile]
        self._W_tile = nn.ParameterList(p)

        self.projection = Projection(self.W_tile, config.projection)
        self.decoder = Decoder(context.n_o, self.W_tile, config.decoder)
        self.encoder = Encoder(context.n_o, self.decoder.n_o_c, config.encoder)
        self.processor = Processor(context.f_initial, config.processor)

    @property
    def W_tile(self) -> nn.ParameterList:
        """Tiling matrices for projection and decoder.

        Returns:
            Parameter list of tiling matrices, one per frequency module.
        """
        return self._W_tile

    def set_tile_learning(self, learn: bool):
        """Set learning state for W_tile matrices.

        Args:
            learn: If True, enable gradients; if False, freeze parameters
        """
        self._config.learn_W_tile = learn
        for param in self.W_tile:
            param.requires_grad_(learn)

    def init_state(self, batch_size: int, device: torch.device) -> LECState:
        """Initialize LEC state with zeros.

        Args:
            batch_size: Batch size for tensor allocation
            device: Device for tensor allocation

        Returns:
            Initial LECState with zero-initialized compressed, filtered, and projected observations
        """
        x_c = [torch.zeros((batch_size, self.encoder.n_o_c), dtype=torch.float, device=device) for _ in range(self.processor.n_f)]
        x = [torch.zeros((batch_size, self.encoder.n_o_c), dtype=torch.float, device=device) for _ in range(self.processor.n_f)]
        x_ = [torch.zeros((batch_size, self.projection.n_p[f]), dtype=torch.float, device=device) for f in range(self.processor.n_f)]
        return LECState(compressed_observation=x_c, filtered_observation=x, projection=x_)

    def forward(self, x: Observation, state: LECState) -> LECState:
        """Forward pass through LEC pathway.

        Args:
            x: Sensory observation (one-hot encoded).
            state: Previous LEC state.

        Returns:
            Updated LEC state with new sensory representations.
        """
        x_c = self.encoder(x)  # Compress sensory observation: x → x_c (one-hot to two-hot)
        x = self.processor(x_c, state.filtered_observation)  # Temporally filter sensorium: x_c → x
        x_ = self.projection(x)  # Project to hippocampal input: x → x_
        return LECState(compressed_observation=x_c, filtered_observation=x, projection=x_)

    def decode(self, p: MultiScaleCode) -> SensoryPrediction:
        """Decode hippocampal place cells to sensory predictions.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Sensory prediction with observation probabilities and logits.
        """
        return self.decoder(p)


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """LEC module usage example.

    Demonstrates sensory processing through encoding, temporal filtering,
    projection, and decoding.
    """
    print("=" * 80)
    print("LEC Module Example - Sensory Processing")
    print("=" * 80)

    # Configuration
    n_o = 45  # Observation space
    n_o_c = 10  # Compressed dimension
    n_p = [96, 80, 64]  # Place cells per frequency
    f_initial = [0.8, 0.5, 0.3]  # Initial frequencies
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Observation space: {n_o}")
    print(f"  Compressed dimension: {n_o_c}")
    print(f"  Frequencies: {len(f_initial)}")
    print(f"  Place cells per frequency: {n_p}")
    print(f"  Batch size: {batch_size}")

    # Create tiling matrices (context)
    W_tile = [torch.randn(n_o_c, n_p_f) for n_p_f in n_p]

    # Create mock context
    from dataclasses import dataclass

    @dataclass
    class MockContext:
        n_o: int
        f_initial: list
        W_tile: list

    context = MockContext(n_o=n_o, f_initial=f_initial, W_tile=W_tile)

    # Create LEC model
    config = LECConfig(learn_W_tile=False)
    lec = LECModel(context, config)
    print(f"\n✓ LEC model initialized")

    # Initialize state
    device = torch.device("cpu")
    state = lec.init_state(batch_size, device)
    print(f"✓ LEC state initialized")

    # Create observation (one-hot)
    obs_indices = torch.randint(0, n_o, (batch_size,))
    observation = torch.nn.functional.one_hot(obs_indices, num_classes=n_o).float()
    print(f"\n✓ Observation created: {observation.shape}")

    # Forward pass
    with torch.no_grad():
        state = lec.forward(observation, state)

    print(f"\n✓ Forward pass complete")
    print(f"  Compressed: {[x.shape for x in state.compressed_observation]}")
    print(f"  Filtered: {[x.shape for x in state.filtered_observation]}")
    print(f"  Projected: {[x.shape for x in state.projection]}")

    # Decode from place cells
    p = [torch.randn(batch_size, n_p_f) for n_p_f in n_p]
    with torch.no_grad():
        prediction = lec.decode(p)

    print(f"\n✓ Decoding complete")
    print(f"  Prediction shape: {prediction.values[0].shape}")
    print(f"  Sum to 1: {torch.allclose(prediction.values[0].sum(dim=-1), torch.ones(batch_size))}")

    print("\n" + "=" * 80)
