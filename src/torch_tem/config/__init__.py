"""Configuration system for torch_tem package.

Provides:
- Parameters: Unified Pydantic model with all configuration (single source of truth)
- Protocol facets: Narrow interface contracts for components
"""

from torch_tem.config import facets
from torch_tem.config.parameters import Parameters

__all__ = [
    "Parameters",
    "facets",
]
