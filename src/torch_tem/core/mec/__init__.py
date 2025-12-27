from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from torch_tem.core.mec.abstract import AbstractLocConfig, AbstractLocInference
from torch_tem.core.mec.projection import Projection, ProjectionConfig
from torch_tem.core.mec.transition import Transition, TransitionConfig
from torch_tem.types import AbstractLocation, GroundedLocation, MultiScaleCode


class MECConfig(BaseModel):
    """MEC model configuration parameters."""

    model_config = ConfigDict(extra="ignore", strict=False, arbitrary_types_allowed=True)

    # Learning projection matrices
    learn_W_down: bool = Field(default=False, description="If True, downsampling matrices W_down are learnable")
    learn_W_repeat: bool = Field(default=False, description="If True, expansion matrices W_repeat are learnable")

    # Submodule configurations
    abstract: AbstractLocConfig = Field(default_factory=AbstractLocConfig, description="Abstract location inference configuration")
    transition: TransitionConfig = Field(default_factory=TransitionConfig, description="Transition model configuration")
    projection: ProjectionConfig = Field(default_factory=ProjectionConfig, description="Projection configuration")


class MECContext(Protocol):
    """Protocol for MEC model initialization parameters.

    Attributes:
        ...
    """

    W_down: List[Tensor]
    W_repeat: List[Tensor]


@dataclass
class MECState:
    """State container for MEC pathway.

    Attributes:
        transition_stats: Predicted abstract location from transition model.
        abstract_location: Fused abstract location after inference.
        projection: Projected abstract location to hippocampal input space.
    """

    transition_stats: Transition
    abstract_location: AbstractLocation
    projection: MultiScaleCode

    def detach(self) -> "MECState":
        """Detach all tensors from the computation graph."""
        return MECState(
            transition_stats=Transition(
                mean=[m.detach() for m in self.transition_stats.mean],
                uncertainty=[u.detach() for u in self.transition_stats.uncertainty],
            ),
            abstract_location=[x.detach() for x in self.abstract_location],
            projection=[x.detach() for x in self.projection],
        )


class MECModel(nn.Module):
    """Medial Entorhinal Cortex (MEC) pathway for spatial navigation.

    Implements abstract location processing through:
    - Transition prediction (path integration)
    - Abstract location inference (fusion with memory and landmarks)
    - Projection to hippocampal input space
    """

    def __init__(self, context: MECContext, config: MECConfig):
        """Initialize MEC model."""
        super().__init__()
        self._config = config

        # Register W_down ...
        p = [nn.Parameter(matrix, requires_grad=config.learn_W_down) for matrix in context.W_down]
        self._W_down = nn.ParameterList(p)

        # Register W_repeat ...
        p = [nn.Parameter(matrix, requires_grad=config.learn_W_repeat) for matrix in context.W_repeat]
        self._W_repeat = nn.ParameterList(p)

        self.projection = Projection(self.W_down, self.W_repeat, config.projection)
        n_g, n_p = self.projection.n_g, self.projection.n_p
        self.abstract = AbstractLocInference(n_g, n_p, config.abstract)
        self.transition = Transition(config.transition)

    @property
    def W_down(self) -> nn.ParameterList:
        """Downsampling matrices for projection.

        Returns:
            Parameter list of downsampling matrices, one per frequency module.
        """
        return self._W_down

    @property
    def W_repeat(self) -> nn.ParameterList:
        """Expansion matrices for projection.

        Returns:
            Parameter list of expansion matrices, one per frequency module.
        """
        return self._W_repeat

    def init_state(self, device: torch.device) -> MECState:
        """Initialize MEC state with zeros.

        Args:
            device: Device for tensor allocation.

        Returns:
            Initial MECState with zero-initialized abstract locations.
        """
        g_gen = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g = [torch.zeros((self.batch_size, self.n_g[f]), dtype=torch.float, device=device) for f in range(self.n_f)]
        g_ = self.projection(g)  # Project to hippocampal input
        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)

    def forward(self, p_x: Optional[GroundedLocation], locations: List[Dict], a: Optional[Tensor], state: MECState) -> MECState:
        """Forward pass through MEC pathway.

        Args:
            p_x: Hippocampal pattern from sensory retrieval (inference mode) or None (generative mode)
            locations: Environment descriptors for landmark cues
            a: Action taken
            state: Previous MEC state

        Returns:
            Updated MEC state with new abstract location
        """
        g_gen = self.transition(state.abstract_location, a)  # State transition: g → g (predict next abstract location)
        g = self.abstract(g_gen, p_x, locations)  # Infer entorhinal (abstract location)
        g_ = self.projection(g)  # Project to hippocampal input: g → g_
        return MECState(transition_stats=g_gen, abstract_location=g, projection=g_)


__all__ = ["MECConfig", "MECState", "MECModel"]
