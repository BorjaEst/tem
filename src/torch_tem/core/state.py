"""State container for TEM model iteration.

Represents the complete state of the model at a single timestep during
forward pass execution. Stores belief states, observations, actions,
memory structures, and optionally generated/inferred values.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor


class State(BaseModel):
    """State container for a single TEM timestep.

    Attributes:
        g: Belief states for each frequency [n_freqs tensors of shape (batch, g_dim)]
        x: Observations for each frequency [n_freqs tensors of shape (batch, x_dim)]
        a: Actions taken at this timestep [list of action values]
        L: Loss components for this timestep [8 tensors of shape (batch,)]
        M: Memory matrices [list of tensors]
        locations: Location metadata for each step in batch [list of dicts]

        g_gen: Generated belief states (optional)
        p_gen: Generated location predictions (optional)
        x_gen: Generated observations (optional)
        x_logits: Observation logits for loss computation (optional)
        x_inf: Inferred observations (optional)
        g_inf: Inferred belief states (optional)
        p_inf: Inferred location predictions (optional)
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,  # Allow PyTorch tensors
        validate_assignment=False,  # Skip validation after creation for performance
        frozen=False,  # Allow mutation (needed for correct() method)
    )

    # Core fields (always present)
    g: List[Tensor]
    x: List[Tensor]
    a: List
    L: List[Tensor]
    M: List[Tensor]
    locations: List[Dict[str, Any]]

    # Optional fields (populated during inference/generation)
    g_gen: Optional[List[Tensor]] = None
    p_gen: Optional[List[Tensor]] = None
    x_gen: Optional[List[Tensor]] = None
    x_logits: Optional[List[Tensor]] = None
    x_inf: Optional[List[Tensor]] = None
    g_inf: Optional[List[Tensor]] = None
    p_inf: Optional[List[Tensor]] = None

    @property
    def batch_size(self) -> int:
        """Batch size inferred from belief state shape."""
        return self.g[0].shape[0]

    @property
    def n_freqs(self) -> int:
        """Number of frequency levels in hierarchical representation."""
        return len(self.g)

    @model_validator(mode="after")
    def validate_list_lengths(self) -> "State":
        """Ensure all frequency-indexed lists have consistent length."""
        n_freqs = len(self.g)

        # Validate core fields (frequency-indexed)
        if len(self.x) != n_freqs:
            raise ValueError(f"x length {len(self.x)} != n_freqs {n_freqs}")
        # Note: L is loss components (8 values), not frequency-indexed
        # Note: M is memory matrices (1-2 values), not frequency-indexed

        # Validate optional fields if present
        if self.g_gen is not None and len(self.g_gen) != n_freqs:
            raise ValueError(f"g_gen length {len(self.g_gen)} != n_freqs {n_freqs}")
        if self.p_gen is not None and len(self.p_gen) != n_freqs:
            raise ValueError(f"p_gen length {len(self.p_gen)} != n_freqs {n_freqs}")
        # Note: x_gen is a tuple of 3 observations, not frequency-indexed
        # Note: x_logits is a tuple of 3 logit tensors, not frequency-indexed
        if self.x_inf is not None and len(self.x_inf) != n_freqs:
            raise ValueError(f"x_inf length {len(self.x_inf)} != n_freqs {n_freqs}")
        if self.g_inf is not None and len(self.g_inf) != n_freqs:
            raise ValueError(f"g_inf length {len(self.g_inf)} != n_freqs {n_freqs}")
        if self.p_inf is not None and len(self.p_inf) != n_freqs:
            raise ValueError(f"p_inf length {len(self.p_inf)} != n_freqs {n_freqs}")

        return self

    def correct(self, g: List[Tensor], p: List[Tensor]) -> None:
        """Correct inferred values with ground truth.

        Mutates g_inf and p_inf in place to match provided ground truth values.
        Used during training to inject correct beliefs/locations.

        Args:
            g: Ground truth belief states [n_freqs tensors]
            p: Ground truth location predictions [n_freqs tensors]
        """
        self.g_inf = g
        self.p_inf = p

    def detach(self) -> "State":
        """Create new State with all tensors detached from computation graph.

        Returns:
            New State instance with detached tensors (computation graph broken).
        """

        def detach_list(tensor_list: Optional[List[Tensor]]) -> Optional[List[Tensor]]:
            """Helper to detach all tensors in a list."""
            return [t.detach() if isinstance(t, Tensor) else t for t in tensor_list] if tensor_list else None

        return State(
            g=detach_list(self.g),
            x=detach_list(self.x),
            a=self.a,  # Actions are not tensors, no detach needed
            L=detach_list(self.L),
            M=detach_list(self.M),
            locations=self.locations,  # Metadata, no detach needed
            g_gen=detach_list(self.g_gen),
            p_gen=detach_list(self.p_gen),
            x_gen=detach_list(self.x_gen),
            x_logits=detach_list(self.x_logits),
            x_inf=detach_list(self.x_inf),
            g_inf=detach_list(self.g_inf),
            p_inf=detach_list(self.p_inf),
        )
