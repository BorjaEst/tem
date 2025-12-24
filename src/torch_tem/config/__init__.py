"""Configuration system for torch_tem package.

Provides:
- ModelConfig: Model structure (dimensions, connectivity, memory)
- TrainingConfig: Optimization schedule (LR, loss weights, truncated BPTT rollout)
- EnvironmentConfig: Task definition (action space, shiny objects)
- DataModuleConfig: Data generation (walk length, policy, batching, splits)
- Policy Configs: RandomPolicyConfig, DistancePolicyConfig, QLearningPolicyConfig, ShinyPolicyConfig, MixedPolicyConfig
"""

from torch_tem.config.architecture import ModelConfig
from torch_tem.config.datamodule import (
    DataModuleConfig,
    DistancePolicyConfig,
    EnvironmentConfig,
    MixedPolicyConfig,
    PolicyConfig,
    QLearningPolicyConfig,
    RandomPolicyConfig,
    ShinyPolicyConfig,
)
from torch_tem.config.training import LossConfig, TrainingConfig

__all__ = [
    "ModelConfig",
    "DataModuleConfig",
    "EnvironmentConfig",
    "PolicyConfig",
    "RandomPolicyConfig",
    "DistancePolicyConfig",
    "QLearningPolicyConfig",
    "ShinyPolicyConfig",
    "MixedPolicyConfig",
    "LossConfig",
    "TrainingConfig",
]
