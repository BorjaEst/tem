"""Transition model for path integration.

Predicts next abstract location from current location and action through
learned transition dynamics.

Handles:
    - Action-based transitions (via MLP_D_a)
    - No-action transitions for shiny environments (via D_no_a)
    - Hierarchical connections between frequency modules
    - Uncertainty estimation (sigma_g)

Typical usage example:
    >>> config = TransitionConfig(d_hidden_dim=20, g_init_std=0.5)
    >>> transition = TransitionModel(
    ...     n_g=[48, 40, 32],
    ...     n_f_grid=3,
    ...     n_actions=4,
    ...     f_initial=[0.8, 0.5, 0.3],
    ...     config=config
    ... )
    >>> g_next = transition(g_prev, action, valid_mask)
"""

from typing import List, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import truncnorm
from torch import Tensor

from torch_tem import utils
from torch_tem.core.mlp import MLP
from torch_tem.types import AbstractLocation, Transition

__all__ = ["TransitionConfig", "TransitionModel"]


class TransitionConfig(BaseModel):
    """Transition model configuration (hyperparameters only).

    Attributes:
        d_hidden_dim: Hidden layer width for transition MLP.
        g_init_std: Standard deviation for initializing abstract location prior.
        do_sample: Whether to sample from transition distribution (vs using mean).
    """

    model_config = ConfigDict(extra="forbid", strict=False, arbitrary_types_allowed=True)

    d_hidden_dim: int = Field(default=20, ge=1, description="Hidden layer width for transition MLP")
    g_init_std: float = Field(default=0.5, gt=0, description="Standard deviation for initializing g prior distribution")
    do_sample: bool = Field(default=False, description="Whether to sample from transition distribution (vs using mean)")


class TransitionModel(nn.Module):
    """Predicts next abstract location from current location and action.

    Handles:
        - Action-based transitions (via MLP_D_a)
        - No-action transitions for shiny environments (via D_no_a)
        - Hierarchical connections between frequency modules
        - Uncertainty estimation (sigma_g)
    """

    def __init__(self, n_g: List[int], n_f_grid: int, n_actions: int, f_initial: List[float], config: TransitionConfig):
        """Initialize transition model.

        Args:
            n_g: Abstract location dimensions per frequency [n_f].
            n_f_grid: Number of grid cell frequency modules.
            n_actions: Number of possible actions.
            f_initial: Base frequency values for hierarchical connections [n_f_grid].
            config: Transition configuration (hyperparameters).
        """
        super().__init__()
        self._config = config

        # Architectural constants
        self.n_g = n_g
        self.n_f = n_f = len(n_g)
        self.n_f_grid = n_f_grid
        self.n_f_ovc = n_f - n_f_grid
        self.n_actions = n_actions

        # Create hierarchical connections
        # f_initial only contains grid frequencies; extend if OVC modules exist
        f_extended = f_initial if self.n_f_ovc == 0 else f_initial + [0.1] * self.n_f_ovc
        self.g_connections = utils.create_g_connections(n_f_grid, f_extended)

        # MLP for action-based transitions
        in_dim = [n_actions] * n_f
        out_dim = [sum([n_g[f_from] for f_from in range(n_f) if self.g_connections[f_to][f_from]]) * n_g[f_to] for f_to in range(n_f)]
        self.MLP_D_a = MLP(in_dim, out_dim, activation=[torch.tanh, None], hidden_dim=[config.d_hidden_dim] * n_f, bias=[True, False])
        self.MLP_D_a.set_weights(1, 0.0)  # Initialize to identity (no change initially)

        # No-action transition weights for shiny environments
        self.D_no_a = nn.ParameterList(
            [nn.Parameter(torch.zeros(sum([n_g[f_from] for f_from in range(n_f) if self.g_connections[f_to][f_from]]) * n_g[f_to])) for f_to in range(n_f)]
        )

        # Uncertainty estimation MLPs
        in_dim = n_g
        out_dim = n_g
        self.MLP_sigma_g_path = MLP(in_dim, out_dim, activation=[torch.tanh, torch.exp], hidden_dim=[2 * g for g in n_g])

        # Log of standard deviation of abstract location cells when entering a new environment
        # Standard deviation of the prior on g. Initialise with truncated normal
        self.logsig_g_init = nn.ParameterList(
            [nn.Parameter(torch.tensor(truncnorm.rvs(-2, 2, size=n_g[f], loc=0, scale=config.g_init_std), dtype=torch.float)) for f in range(n_f)]
        )

    @property
    def do_sample(self) -> bool:
        """Whether to sample from transition distribution."""
        return self._config.do_sample

    def forward(self, g_prev: AbstractLocation, a: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None) -> Transition:
        """Main forward pass.

        Args:
            g_prev: Previous abstract location
            a: Action tensor (optional). If provided, uses action-based transition.
               If None, uses no-action transition (for shiny environments).
            valid_mask: Boolean tensor [B] indicating valid steps (True) vs new walks (False)

        Returns:
            Transition: (g_gen, sigma_g) tuple
        """
        if a is not None:
            transition_result = self.transition_with_action(g_prev, a, valid_mask)
        else:
            transition_result = self.transition_no_action(g_prev)

        g_gen = self.sample(transition_result)
        return Transition(mean=g_gen, uncertainty=transition_result.uncertainty)

    def transition_with_action(self, g_prev: AbstractLocation, a: Tensor, valid_mask: Optional[Tensor] = None) -> Transition:
        """Compute transition using action: g_t+1 = g_t + D(a) * g_connections.

        Args:
            g_prev: Previous abstract location [n_f] of [B, n_g[f]]
            a: Action tensor [B, n_actions] (one-hot) or [B] (indices, gym standard)
            valid_mask: Boolean tensor [B] indicating valid steps (True) vs new walks (False)

        Returns:
            Transition: (mu_g, sigma_g) tuple
        """
        # Convert action indices to one-hot if needed
        if a.dim() == 1 or a.shape[1] == 1:
            a_indices = a.squeeze().long()
            a_onehot = torch.nn.functional.one_hot(a_indices, num_classes=self.n_actions).float()
        else:
            a_onehot = a

        mu_g = []
        # MLP expects list input if it has multiple modules
        D_a = self.MLP_D_a([a_onehot] * self.n_f)  # Replicate action for each frequency

        for f_to in range(self.n_f):
            # Gather inputs from connected frequencies
            g_inputs = torch.cat([g_prev[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1)

            # Reshape transition matrix and apply
            n_inputs = sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]])
            D_f = D_a[f_to].view(-1, n_inputs, self.n_g[f_to])

            # Apply transition: g_new = g_old + transition
            delta_g = torch.bmm(g_inputs.unsqueeze(1), D_f).squeeze(1)

            # Calculate new mean and clamp (stability)
            mu_g_f_step = g_prev[f_to] + delta_g
            mu_g_f_step = torch.clamp(mu_g_f_step, min=-1.0, max=1.0)

            if valid_mask is not None:
                # If invalid step (new walk), use g_prev (g_init) directly, UNCLAMPED
                # This matches legacy behavior where g_init is returned for new walks
                mask_f = valid_mask.unsqueeze(1).expand_as(mu_g_f_step)
                mu_g_f = torch.where(mask_f, mu_g_f_step, g_prev[f_to])
            else:
                mu_g_f = mu_g_f_step

            mu_g.append(mu_g_f)

        # Compute uncertainty
        # Legacy parity: sigma_g depends on g_prev
        from_g = self.MLP_sigma_g_path(g_prev)

        if valid_mask is not None:
            # If valid_mask is provided, switch between predicted sigma and prior sigma
            # valid_mask is [B], True if valid step, False if new walk
            from_prior = [torch.exp(logsig) for logsig in self.logsig_g_init]

            sigma_g = []
            for f in range(self.n_f):
                # Expand prior to batch size
                prior_f = from_prior[f].unsqueeze(0).expand(from_g[f].shape[0], -1)
                # Select based on mask
                # We need to handle the mask shape. valid_mask is [B].
                mask_f = valid_mask.unsqueeze(1).expand_as(from_g[f])
                sigma_f = torch.where(mask_f, from_g[f], prior_f)
                sigma_g.append(sigma_f)
        else:
            sigma_g = from_g

        return Transition(mean=mu_g, uncertainty=sigma_g)

    def transition_no_action(self, g_prev: AbstractLocation) -> Transition:
        """Compute transition without action (for shiny environments).

        Args:
            g_prev: Previous abstract location [n_f] of [B, n_g[f]]

        Returns:
            Transition: (mu_g, sigma_g) tuple
        """
        mu_g = []

        for f_to in range(self.n_f):
            g_inputs = torch.cat([g_prev[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1)

            n_inputs = sum([self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]])
            D_f = self.D_no_a[f_to].view(n_inputs, self.n_g[f_to])

            delta_g = torch.matmul(g_inputs, D_f)
            mu_g.append(g_prev[f_to] + delta_g)

        # Fixed low uncertainty for no-action
        sigma_g = [torch.ones_like(g) * 0.1 for g in mu_g]

        return Transition(mean=mu_g, uncertainty=sigma_g)

    def sample(self, g_gen: Transition) -> AbstractLocation:
        """Sample from transition distribution if do_sample=True.

        Args:
            mu_g: Mean of abstract location
            sigma_g: Standard deviation

        Returns:
            g: Sampled (or mean) abstract location
        """
        mu_g, sigma_g = g_gen.mean, g_gen.uncertainty
        if self.do_sample:
            return [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(mu_g, sigma_g)]
        return mu_g


# ======================================================================================
# USAGE EXAMPLE
# ======================================================================================

if __name__ == "__main__":
    """Transition model example: Path integration for spatial navigation.

    Demonstrates how the transition model predicts next abstract locations
    from current locations and actions, with hierarchical connections
    between frequency modules.
    """
    print("=" * 80)
    print("Transition Model Example - Path Integration")
    print("=" * 80)

    # Configuration
    n_g = [48, 40, 32]  # Grid cells per frequency
    n_f_grid = 3  # Number of grid frequency modules
    n_actions = 4  # Number of possible actions (up, down, left, right)
    f_initial = [0.8, 0.5, 0.3]  # Base frequencies
    batch_size = 4

    print(f"\nConfiguration:")
    print(f"  Grid cells: {n_g}")
    print(f"  Grid frequencies: {n_f_grid}")
    print(f"  Actions: {n_actions}")
    print(f"  Base frequencies: {f_initial}")
    print(f"  Batch size: {batch_size}")

    # Create configuration and transition model
    config = TransitionConfig(d_hidden_dim=20, g_init_std=0.5, do_sample=False)
    transition = TransitionModel(n_g=n_g, n_f_grid=n_f_grid, n_actions=n_actions, f_initial=f_initial, config=config)
    print(f"\n✓ Transition model initialized")

    # Create current location
    g_prev = [torch.randn(batch_size, n) for n in n_g]
    print(f"\n✓ Current location: {[g.shape for g in g_prev]}")

    # Example 1: Action-based transition
    print(f"\n{'=' * 80}")
    print(f"Example 1: Action-Based Transition")
    print(f"{'=' * 80}")

    actions = torch.randint(0, n_actions, (batch_size,))
    valid_mask = torch.ones(batch_size, dtype=torch.bool)
    print(f"✓ Actions: {actions}")

    g_next = transition(g_prev, actions, valid_mask)
    print(f"\n✓ Transition complete:")
    print(f"  Next location (mean): {[g.shape for g in g_next.mean]}")
    print(f"  Uncertainty: {[s.shape for s in g_next.uncertainty]}")

    # Example 2: No-action transition (for shiny environments)
    print(f"\n{'=' * 80}")
    print(f"Example 2: No-Action Transition")
    print(f"{'=' * 80}")

    g_next_no_action = transition(g_prev, None, None)
    print(f"✓ No-action transition complete:")
    print(f"  Next location (mean): {[g.shape for g in g_next_no_action.mean]}")
    print(f"  Uncertainty: {[s.shape for s in g_next_no_action.uncertainty]}")

    # Example 3: Sampling
    print(f"\n{'=' * 80}")
    print(f"Example 3: Stochastic Sampling")
    print(f"{'=' * 80}")

    g_sampled = transition.sample(g_next)
    print(f"✓ Sampled from distribution:")
    print(f"  Sampled location: {[g.shape for g in g_sampled]}")

    print(f"\n{'=' * 80}")
    print(f"✓ Transition model example complete")
    print(f"{'=' * 80}")
