"""Data generation system for TEM training.

Provides synthetic graph-world environments, policy generation, walk sampling,
synthetic grid cell pattern generation, and memory training pattern generation.
Replaces the legacy world.py with a modular, Protocol-based architecture.
"""

from torch_tem.data.abstract import MemorySignalGenerator, ShinySignalGenerator, TransitionPredictionGenerator
from torch_tem.data.datamodule import InfiniteWalkDataset, TEMDataModule
from torch_tem.data.environment import Action, Environment, Location
from torch_tem.data.memory import MemoryMatrixGenerator, generate_block_diagonal_memory, generate_hebbian_memory, generate_memory_from_walk, generate_random_memory
from torch_tem.data.patterns import GridCellPatternGenerator, OscillatoryGridGenerator, PairedPatternGenerator, PlaceCellPatternGenerator
from torch_tem.data.policies import PolicyGenerator
from torch_tem.data.shiny import ShinyConfig, ShinyEnvironmentBuilder
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
    "OscillatoryGridGenerator",
    "PlaceCellPatternGenerator",
    "GridCellPatternGenerator",
    "PairedPatternGenerator",
    "TransitionPredictionGenerator",
    "MemorySignalGenerator",
    "ShinySignalGenerator",
    "MemoryMatrixGenerator",
    "generate_hebbian_memory",
    "generate_memory_from_walk",
    "generate_random_memory",
    "generate_block_diagonal_memory",
]
