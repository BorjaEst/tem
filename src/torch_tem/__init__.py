"""PyTorch implementation of the Tolman-Eichenbaum Machine (TEM).

This package provides a modern, modular implementation of the TEM architecture
for spatial navigation and memory research.

Modules:
    config: Configuration management using Pydantic
    core: Core neural network components (grounded location inference)
    data: Data generation and loading utilities
    hpc: Hippocampus components (memory storage and retrieval)
    lec: Lateral Entorhinal Cortex components (sensory processing)
    mec: Medial Entorhinal Cortex components (abstract location processing)
    losses: Pathway-based loss functions for flexible training
    model: High-level TEMModel orchestrator
    types: Type definitions and protocols
    utils: Utility functions

Example:
    >>> from torch_tem import TEMModel
    >>> from torch_tem.losses import JointLoss
    >>> from torch_tem.config import ModelConfig
    >>>
    >>> # Initialize model and loss
    >>> config = ModelConfig(n_x=25, n_g=[32, 16, 8])
    >>> model = TEMModel(config)
    >>> loss_fn = JointLoss()
    >>>
    >>> # Training
    >>> output = loss_fn(g_gen, g_inf, p_gen, p_inf, x_gen, x_inf, x_target)
    >>> output.total.backward()
"""

from . import config, core, data, hpc, lec, losses, mec, model, types, utils

__all__ = [
    "config",
    "core",
    "data",
    "hpc",
    "lec",
    "losses",
    "mec",
    "model",
    "types",
    "utils",
]
