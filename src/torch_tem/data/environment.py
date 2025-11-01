"""Graph-world environment structure and validation."""

import json
from typing import Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
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

    def __init__(self, env_spec: Union[str, Dict], randomize_observations: bool = False):
        """Load environment from JSON file or dictionary.

        Args:
            env_spec: Path to JSON file or environment dictionary with keys:
                     'adjacency', 'locations', 'n_actions', 'n_locations', 'n_observations'
            randomize_observations: Shuffle observation assignments after loading

        Raises:
            ValueError: If environment specification is invalid
            FileNotFoundError: If JSON file not found
            json.JSONDecodeError: If JSON is malformed
        """
        # Load from file if string path provided
        if isinstance(env_spec, str):
            with open(env_spec, "r") as f:
                env_dict = json.load(f)
        else:
            env_dict = env_spec

        # Validate required fields
        required_fields = ["adjacency", "locations", "n_actions", "n_locations", "n_observations"]
        missing = [f for f in required_fields if f not in env_dict]
        if missing:
            raise ValueError(f"Environment specification missing required fields: {missing}")

        # Store basic attributes
        self.n_locations = env_dict["n_locations"]
        self.n_observations = env_dict["n_observations"]
        self.n_actions = env_dict["n_actions"]
        self.adjacency = env_dict["adjacency"]

        # Convert locations to Pydantic models
        self.locations: List[Location] = []
        for loc_dict in env_dict["locations"]:
            # Convert actions to Action models
            actions = [Action(**action) for action in loc_dict["actions"]]
            # Create Location model
            location = Location(id=loc_dict["id"], observation=loc_dict["observation"], actions=actions, shiny=loc_dict.get("shiny", None))
            self.locations.append(location)

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

    @classmethod
    def from_grid(cls, width: int, height: int, observation_mode: str = "unique") -> "Environment":
        """Generate grid-world environment programmatically.

        Args:
            width: Grid width
            height: Grid height
            observation_mode:
                - "unique": Each location has unique observation
                - "tiled": Observations tile in 2x2 pattern
                - "random": Random observation assignment

        Returns:
            Environment: Generated grid-world environment

        Raises:
            ValueError: If observation_mode is invalid
        """
        n_locations = width * height

        # Generate observations based on mode
        if observation_mode == "unique":
            n_observations = n_locations
            observations = list(range(n_locations))
        elif observation_mode == "tiled":
            n_observations = 4
            observations = [(i % 2) * 2 + (j % 2) for i in range(height) for j in range(width)]
        elif observation_mode == "random":
            n_observations = max(4, n_locations // 4)
            observations = [np.random.randint(n_observations) for _ in range(n_locations)]
        else:
            raise ValueError(f"Invalid observation_mode: {observation_mode}")

        # 4 actions: up, right, down, left
        n_actions = 4

        # Build adjacency matrix and locations
        adjacency = [[0.0] * n_locations for _ in range(n_locations)]
        locations = []

        for loc_id in range(n_locations):
            i = loc_id // width  # row
            j = loc_id % width  # column

            actions = []

            # Up (action 0)
            if i > 0:
                next_loc = (i - 1) * width + j
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append({"id": 0, "probability": 0.25, "transition": transition})

            # Right (action 1)
            if j < width - 1:
                next_loc = i * width + (j + 1)
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append({"id": 1, "probability": 0.25, "transition": transition})

            # Down (action 2)
            if i < height - 1:
                next_loc = (i + 1) * width + j
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append({"id": 2, "probability": 0.25, "transition": transition})

            # Left (action 3)
            if j > 0:
                next_loc = i * width + (j - 1)
                adjacency[loc_id][next_loc] = 1.0
                transition = [1.0 if k == next_loc else 0.0 for k in range(n_locations)]
            else:
                transition = [1.0 if k == loc_id else 0.0 for k in range(n_locations)]
            actions.append({"id": 3, "probability": 0.25, "transition": transition})

            locations.append({"id": loc_id, "observation": observations[loc_id], "actions": actions})

        env_dict = {"n_locations": n_locations, "n_observations": n_observations, "n_actions": n_actions, "adjacency": adjacency, "locations": locations}

        return cls(env_dict)
