"""Configuration system for torch_tem package.

Provides:
- ModelConfig: Model structure (dimensions, connectivity, memory)
- TrainingConfig: Optimization schedule (LR, curricula, loss weights)
- EnvironmentConfig: Task definition (action space, shiny objects)
"""

from torch_tem.config.architecture import ModelConfig
from torch_tem.config.environment import EnvironmentConfig
from torch_tem.config.training import TrainingConfig

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "EnvironmentConfig",
]
