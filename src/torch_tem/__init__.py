"""PyTorch implementation of the Tolman-Eichenbaum Machine (TEM)."""

from .model import Parameters
from .utils import (
    create_downsampling_matrix,
    create_hierarchical_connections,
    create_hierarchical_mask,
    create_outer_product_repeat_matrix,
    create_outer_product_tile_matrix,
    create_retrieval_masks,
    generate_two_hot_codes,
)

__all__ = [
    "Parameters",
    "create_hierarchical_mask",
    "create_retrieval_masks",
    "create_hierarchical_connections",
    "create_outer_product_repeat_matrix",
    "create_outer_product_tile_matrix",
    "generate_two_hot_codes",
    "create_downsampling_matrix",
]
