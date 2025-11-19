"""Configuration system for torch_tem package.

Provides:
- ArchitectureConfig: Model structure (dimensions, connectivity, memory)
- TrainingConfig: Optimization schedule (LR, curricula, loss weights)
- EnvironmentConfig: Task definition (action space, shiny objects)
- InferenceConfig: Runtime behavior (sampling, memory dynamics)
"""

from torch_tem.config.architecture import ArchitectureConfig
from torch_tem.config.environment import EnvironmentConfig
from torch_tem.config.inference import InferenceConfig
from torch_tem.config.training import TrainingConfig

__all__ = [
    "ArchitectureConfig",
    "TrainingConfig",
    "EnvironmentConfig",
    "InferenceConfig",
]
