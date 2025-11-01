"""Data generation system for TEM training.

Provides synthetic graph-world environments, policy generation, walk sampling,
and synthetic grid cell pattern generation.
Replaces the legacy world.py with a modular, Protocol-based architecture.
"""

from torch_tem.data.datamodule import InfiniteWalkDataset, TEMDataModule
from torch_tem.data.environment import Action, Environment, Location
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyConfig, ShinyEnvironmentBuilder
from torch_tem.data.synthetic import SyntheticGridGenerator
from torch_tem.data.walks import Walk, WalkGenerator

__all__ = [
    "Environment",
    "Location",
    "Action",
    "PolicyGenerator",
    "ShinyConfig",
    "ShinyEnvironmentBuilder",
    "Walk",
    "WalkGenerator",
    "TEMDataModule",
    "InfiniteWalkDataset",
    "SyntheticGridGenerator",
]
