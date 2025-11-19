"""Graph-world environment structure and validation."""

from typing import List, Literal, Optional, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.sparse.csgraph import shortest_path


class Action(BaseModel):
    """Action with transition probabilities.

    Represents a single action available at a location with its
    probability of selection and transition distribution to next states.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: int = Field(ge=0, description="Action identifier")
    probability: float = Field(ge=0.0, le=1.0, description="Selection probability")
    transition: List[float] = Field(description="Probability distribution over next states")

    @field_validator("transition")
    @classmethod
    def validate_probability_distribution(cls, v: List[float]) -> List[float]:
        """Ensure transition probabilities sum to approximately 1.0."""
        total = sum(v)
        if total > 0 and not (0.999 <= total <= 1.001):  # Tolerance for floating point
            raise ValueError(f"Transition probabilities must sum to 1.0, got {total}")
        return v


class Location(BaseModel):
    """Single location in the environment graph.

    Represents a discrete location with an observation, available actions,
    and optional shiny object marker.
    """

    id: int = Field(ge=0, description="Location identifier")
    observation: int = Field(ge=0, description="Observation ID at this location")
    actions: List[Action] = Field(description="Available actions from this location")
    shiny: Optional[bool] = Field(default=None, description="Whether location has shiny object")

    @field_validator("actions")
    @classmethod
    def validate_actions_nonempty(cls, v: List[Action]) -> List[Action]:
        """Ensure each location has at least one action."""
        if not v:
            raise ValueError("Location must have at least one action")
        return v


class EnvironmentParams(Protocol):
    """Protocol for objects that can parameterize an ``Environment``.

    ``EnvironmentConfig`` in ``torch_tem.config.environment`` is the primary
    implementation of this protocol in practice.
    """

    width: int
    height: int
    observation_mode: Literal["unique", "tiled", "random"]
    n_actions: int

    @property
    def n_locations(self) -> int:  # pragma: no cover - simple protocol
        ...

    @property
    def n_observations(self) -> int:  # pragma: no cover - simple protocol
        ...


class Environment:
    """Graph-world environment with locations and transitions.

    Manages environment structure loaded from JSON or generated programmatically.
    Provides graph topology queries and validation.

    Attributes:
        n_locations: Number of discrete locations
        n_observations: Number of unique sensory observations
        n_actions: Number of available actions per location
        adjacency: [n_locations, n_locations] adjacency matrix
        locations: List of Location Pydantic models
    """

    def __init__(self, params: EnvironmentParams, randomize_observations: bool = False):
        """Construct environment from validated ``EnvironmentParams``.

        Args:
            params: Environment configuration implementing ``EnvironmentParams``.
            randomize_observations: Shuffle observation assignments after loading.
        """
        # Derive basic attributes from high-level params
        width = params.width
        height = params.height
        n_locations = params.n_locations
        n_observations = params.n_observations

        # Generate observations based on mode (mirrors legacy grid helper)
        if params.observation_mode == "unique":
            observations = list(range(n_locations))
        elif params.observation_mode == "tiled":
            observations = [(i % 2) * 2 + (j % 2) for i in range(height) for j in range(width)]
        elif params.observation_mode == "random":
            observations = [int(np.random.randint(n_observations)) for _ in range(n_locations)]
        else:
            raise ValueError(f"Invalid observation_mode: {params.observation_mode}")

        # 4 directional actions: up, right, down, left
        n_actions = params.n_actions
        adjacency: List[List[float]] = [[0.0] * n_locations for _ in range(n_locations)]
        locations: List[Location] = []

        for loc_id in range(n_locations):
            i = loc_id // width  # row
            j = loc_id % width  # column

            actions: List[Action] = []

            # Up (action 0)
            if i > 0:
                next_loc = (i - 1) * width + j
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append(Action(id=0, probability=0.25, transition=transition))

            # Right (action 1)
            if j < width - 1:
                next_loc = i * width + (j + 1)
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append(Action(id=1, probability=0.25, transition=transition))

            # Down (action 2)
            if i < height - 1:
                next_loc = (i + 1) * width + j
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append(Action(id=2, probability=0.25, transition=transition))

            # Left (action 3)
            if j > 0:
                next_loc = i * width + (j - 1)
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append(Action(id=3, probability=0.25, transition=transition))

            locations.append(
                Location(
                    id=loc_id,
                    observation=observations[loc_id],
                    actions=actions,
                    shiny=None,
                )
            )

        # Store basic attributes
        self.n_locations = n_locations
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.adjacency = adjacency
        self.locations = locations

        # Randomize observations if requested
        if randomize_observations:
            self.randomize_observations()

        # Cache for shortest paths
        self._shortest_paths_cache: Optional[np.ndarray] = None

    def randomize_observations(self) -> None:
        """Randomly shuffle observation assignments across locations."""
        observations = np.random.randint(0, self.n_observations, size=self.n_locations)
        for i, location in enumerate(self.locations):
            # Create new Location with updated observation (Pydantic models are immutable)
            self.locations[i] = location.model_copy(update={"observation": int(observations[i])})

    def shortest_paths(self) -> np.ndarray:
        """Compute all-pairs shortest path distances.

        Returns:
            np.ndarray: [n_locations, n_locations] distance matrix
        """
        if self._shortest_paths_cache is None:
            adjacency_array = np.array(self.adjacency)
            self._shortest_paths_cache = shortest_path(csgraph=adjacency_array, directed=False)
        return self._shortest_paths_cache

    def validate(self) -> bool:
        """Check environment consistency.

        Validates:
        - Location IDs are sequential and complete
        - Observation IDs are within valid range
        - Adjacency matrix dimensions match n_locations
        - Action transitions reference valid locations

        Returns:
            bool: True if valid, raises ValueError otherwise
        """
        # Check location IDs
        location_ids = {loc.id for loc in self.locations}
        expected_ids = set(range(self.n_locations))
        if location_ids != expected_ids:
            raise ValueError(f"Location IDs {location_ids} don't match expected {expected_ids}")

        # Check observations in range
        for loc in self.locations:
            if not (0 <= loc.observation < self.n_observations):
                raise ValueError(f"Location {loc.id} has observation {loc.observation} " f"outside valid range [0, {self.n_observations})")

        # Check adjacency matrix shape
        if len(self.adjacency) != self.n_locations:
            raise ValueError(f"Adjacency matrix has {len(self.adjacency)} rows, " f"expected {self.n_locations}")
        for i, row in enumerate(self.adjacency):
            if len(row) != self.n_locations:
                raise ValueError(f"Adjacency matrix row {i} has {len(row)} columns, " f"expected {self.n_locations}")

        # Check action transitions
        for loc in self.locations:
            for action in loc.actions:
                if len(action.transition) != self.n_locations:
                    raise ValueError(f"Location {loc.id} action {action.id} transition has " f"{len(action.transition)} elements, expected {self.n_locations}")

        return True
