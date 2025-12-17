from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from .. import utils
from ..types import MultiScaleCode, Observation, SensoryPrediction
from . import decoder, encoder, processor, projection


class LECParams(encoder.EncoderParams, decoder.DecoderParams, processor.ProcessorParams, projection.ProjectionParams):
    """Protocol for LEC model initialization parameters.

    Combines parameters from all LEC submodules (encoder, decoder, processor, projection).

    Attributes:
        batch_size: Batch size for state initialization.
        n_g_subsampled_combined: Grid cell dimensions for W_tile creation.
    """

    batch_size: int
    n_g_subsampled_combined: List[int]


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
        """Detach all tensors in the state from the computation graph."""
        return LECState(
            compressed_observation=[x.detach() for x in self.compressed_observation],
            filtered_observation=[x.detach() for x in self.filtered_observation],
            projection=[x.detach() for x in self.projection],
        )


class LECModel(nn.Module):
    """Lateral Entorhinal Cortex (LEC) pathway for sensory processing.

    Implements sensory observation processing through:
    - Encoding: x → x_c (one-hot to two-hot compression)
    - Processing: x_c → x_f (temporal filtering at multiple frequencies)
    - Projection: x_f → x_ (tiling to hippocampal input space)
    - Decoding: p → x (place cells to sensory predictions)
    """

    def __init__(self, params: LECParams):
        """Initialize LEC model.

        Args:
            params: Configuration with n_x, n_x_c, n_f, and all submodule parameters.
        """
        super().__init__()

        # Initialize components
        self.encoder = encoder.Encoder(params)  # Sensory encoder module: x → x_c
        self.decoder = decoder.Decoder(params)  # Observation decoder module: p → x
        self.processor = processor.Processor(params)  # Sensory processor module: x
        self.projection = projection.Projection(params)  # Tiling module for location inference
        self.batch_size = params.batch_size

        # Register W_tile matrices as buffers for centralized device management
        # This ensures W_tile moves with the model when calling .to(device)
        W_tile = utils.create_W_tile(params.n_g_subsampled_combined, self.projection.n_x_f)
        for i, matrix in enumerate(W_tile):
            self.register_buffer(f"W_tile_{i}", matrix)

    @property
    def n_x_c(self) -> List[int]:
        """Number of compressed sensory neurons per frequency."""
        return self.encoder.n_x_c

    @property
    def n_f(self) -> int:
        """Number of frequency modules."""
        return self.processor.n_f

    def init_state(self, device: torch.device) -> LECState:
        """Initialize LEC state with zeros.

        Args:
            device: Device for tensor allocation

        Returns:
            Initial LECState with zero-initialized compressed, filtered, and projected observations
        """
        n_x_c_val = self.n_x_c[0] if isinstance(self.n_x_c, list) else self.n_x_c
        x_c = [torch.zeros((self.batch_size, n_x_c_val), dtype=torch.float, device=device) for f in range(self.n_f)]
        x_f = [torch.zeros((self.batch_size, n_x_c_val), dtype=torch.float, device=device) for f in range(self.n_f)]
        x_ = [torch.zeros((self.batch_size, self.projection.n_p[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        return LECState(compressed_observation=x_c, filtered_observation=x_f, projection=x_)

    def forward(self, x: Observation, state: LECState) -> LECState:
        """Forward pass through LEC pathway.

        Args:
            x: Sensory observation (one-hot encoded).
            state: Previous LEC state.

        Returns:
            Updated LEC state with new sensory representations.
        """
        x_c = self.encoder(x)  # Compress sensory observation: x → x_c (one-hot to two-hot)
        x_f = self.processor(x_c, state.filtered_observation)  # Temporally filter sensorium: x_c → x_f
        x_ = self.projection(x_f, self.get_W_tile())  # Project to hippocampal input: x_f → x_
        return LECState(compressed_observation=x_c, filtered_observation=x_f, projection=x_)

    def decode(self, p: MultiScaleCode) -> SensoryPrediction:
        """Decode hippocampal place cells to sensory predictions.

        Args:
            p: Hippocampal place cells List[n_f] of (batch, n_p[f]).

        Returns:
            Sensory prediction with observation probabilities and logits.
        """
        return self.decoder(p, self.get_tile_matrix(0))

    def get_W_tile(self) -> List[Tensor]:
        """Get all W_tile matrices as a list.

        Returns:
            List of W_tile matrices, one per frequency module.
        """
        return [getattr(self, f"W_tile_{f}") for f in range(self.n_f)]

    def get_tile_matrix(self, f: int) -> Tensor:
        """Get W_tile matrix for a specific frequency module.

        Args:
            f: Frequency module index (0 to n_f-1)

        Returns:
            W_tile matrix for frequency f
        """
        return getattr(self, f"W_tile_{f}")


__all__ = ["LECParams", "LECState", "LECModel"]
