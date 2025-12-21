"""Configuration system for torch_tem package.

Provides:
- ModelConfig: Model structure (dimensions, connectivity, memory)
- TrainingConfig: Optimization schedule (LR, curricula, loss weights)
- EnvironmentConfig: Task definition (action space, shiny objects)
- DataModuleConfig: Data generation (walk length, policy, batching, splits)
- Policy Configs: RandomPolicyConfig, DistancePolicyConfig, QLearningPolicyConfig, ShinyPolicyConfig, MixedPolicyConfig
"""

from torch_tem.config.architecture import ModelConfig
from torch_tem.config.datamodule import DataModuleConfig
from torch_tem.config.environment import EnvironmentConfig
from torch_tem.config.policies import DistancePolicyConfig, MixedPolicyConfig, PolicyConfig, QLearningPolicyConfig, RandomPolicyConfig, ShinyPolicyConfig
from torch_tem.config.training import TrainingConfig

__all__ = [
    "ModelConfig",
    "TrainingConfig",
    "EnvironmentConfig",
    "DataModuleConfig",
    "PolicyConfig",
    "RandomPolicyConfig",
    "DistancePolicyConfig",
    "QLearningPolicyConfig",
    "ShinyPolicyConfig",
    "MixedPolicyConfig",
]
