"""Data generation system for TEM training.

Provides synthetic graph-world environments, policy generation, walk sampling,
synthetic grid cell pattern generation, and memory training pattern generation.
Replaces the legacy world.py with a modular, Protocol-based architecture.
"""

from torch_tem.data.abstract import MemorySignalGenerator, ShinySignalGenerator, TransitionPredictionGenerator
from torch_tem.data.datamodule import TEMDataModule
from torch_tem.data.environment import EnvAction, Environment, EnvironmentConfig, EnvLocation
from torch_tem.data.memory import MemoryMatrixGenerator
from torch_tem.data.patterns import GridCellPatternGenerator, OscillatoryGridGenerator, PairedPatternGenerator, PlaceCellPatternGenerator
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyEnvironmentBuilder
from torch_tem.data.walks import Walk, WalkGenerator

__all__ = [
    # Environment
    "Environment",
    "EnvironmentConfig",
    "EnvLocation",
    "EnvAction",
    # Policy Generation
    "PolicyGenerator",
    # Shiny Objects
    "ShinyEnvironmentBuilder",
    # Walk Generation
    "Walk",
    "WalkGenerator",
    # DataModule
    "TEMDataModule",
    # Pattern Generators
    "OscillatoryGridGenerator",
    "PlaceCellPatternGenerator",
    "GridCellPatternGenerator",
    "PairedPatternGenerator",
    # Abstract Inference
    "TransitionPredictionGenerator",
    "MemorySignalGenerator",
    "ShinySignalGenerator",
    # Memory
    "MemoryMatrixGenerator",
]
