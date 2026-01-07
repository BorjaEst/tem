"""MEC (Medial Entorhinal Cortex) module: grid cell path integration.

Provides action-driven abstract location updates with optional OVC support.

Design goal:
- Keep behavior equivalent to legacy.py's gen_g/f_mu_g_path/f_sigma_g_path
- Keep API compatible with core/model.py (do not modify model.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from scipy.stats import truncnorm
from torch import Tensor, nn

from torch_tem.modules import MLP
from torch_tem.types import Transition


@dataclass
class MECState:
    """State container for MEC dynamics.

    Attributes:
        g_gen: Ancestral prediction for the generative pathway.
        g_path: Path integration prior (mean, uncertainty) for inference.
        g: Abstract location used as 'previous g' at the next timestep.
    """

    g_gen: List[Tensor]
    g_path: Transition
    g: Optional[List[Tensor]] = None

    def __post_init__(self) -> None:
        """Initialize g from g_path.mean if not provided (backward compatibility)."""
        if self.g is None:
            self.g = list(self.g_path.mean)


class MECModel(nn.Module):
    """MEC path integration.

    This module implements the legacy transition model:
    - compute mu_g via action-conditioned transition
    - compute sigma_g via MLP_sigma_g_path
    - sample g if do_sample else take mu_g
    - for shiny environments, generative branch uses non-directional weights
      (legacy: if *any* shiny env exists, g_gen is recomputed for *all* envs)
    """

    def __init__(
        self,
        n_p: List[int],
        n_g: List[int],
        n_g_subsampled: List[int],
        n_f: int,
        n_f_g: int,
        n_f_ovc: int,
        n_a: int,
        d_hidden: int,
        g_conn: list[list[bool]],
        g_init_std: float,
        g_mem_std: float,
        separate_ovc: bool,
        do_sample: bool,
    ):
        super().__init__()

        self.do_sample = do_sample
        self.n_a = n_a
        self.n_g = n_g
        self.n_f = n_f
        self.g_connections = g_conn

        # Priors
        self.g_init = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(
                        truncnorm.rvs(-2, 2, size=n_g[f], loc=0, scale=g_init_std),
                        dtype=torch.float32,
                    )
                )
                for f in range(n_f)
            ]
        )
        self.logsig_g_init = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(
                        truncnorm.rvs(-2, 2, size=n_g[f], loc=0, scale=g_init_std),
                        dtype=torch.float32,
                    )
                )
                for f in range(n_f)
            ]
        )

        # Transition weights
        self.MLP_D_a = MLP(
            [n_a for _ in range(n_f)],
            [sum(n_g[f_from] for f_from in range(n_f) if g_conn[f_to][f_from]) * n_g[f_to] for f_to in range(n_f)],
            activation=[torch.tanh, None],
            hidden_dim=[d_hidden for _ in range(n_f)],
            bias=[True, False],
        )
        self.MLP_D_a.set_weights(1, 0.0)

        # Non-directional transition weights (used for shiny generative branch)
        self.D_no_a = nn.ParameterList([nn.Parameter(torch.zeros(sum(n_g[f_from] for f_from in range(n_f) if g_conn[f_to][f_from]) * n_g[f_to])) for f_to in range(n_f)])

        # Transition uncertainty model
        self.MLP_sigma_g_path = MLP(
            n_g,
            n_g,
            activation=[torch.tanh, torch.exp],
            hidden_dim=[2 * g for g in n_g],
        )

        # Legacy-exposed components (used elsewhere in model.py)
        self.MLP_sigma_p = MLP(n_p, n_p, activation=[torch.tanh, torch.exp])
        self.MLP_mu_g_mem = MLP(n_g_subsampled, n_g, hidden_dim=[2 * g for g in n_g])
        self.MLP_mu_g_mem.set_weights(
            -1,
            [
                torch.tensor(
                    truncnorm.rvs(
                        -2,
                        2,
                        size=list(self.MLP_mu_g_mem.w[f][-1].weight.shape),
                        loc=0,
                        scale=g_mem_std,
                    ),
                    dtype=torch.float32,
                )
                for f in range(n_f)
            ],
        )
        self.MLP_sigma_g_mem = MLP(
            [2 for _ in n_g_subsampled],
            n_g,
            activation=[torch.tanh, torch.exp],
            hidden_dim=[2 * g for g in n_g],
        )
        self.MLP_mu_g_shiny = MLP(
            [1 for _ in range(n_f_ovc if separate_ovc else n_f)],
            [dim for dim in n_g[(n_f_g if separate_ovc else 0) :]],
            hidden_dim=[2 * dim for dim in n_g[(n_f_g if separate_ovc else 0) :]],
        )
        self.MLP_sigma_g_shiny = MLP(
            [1 for _ in range(n_f_ovc if separate_ovc else n_f)],
            [dim for dim in n_g[(n_f_g if separate_ovc else 0) :]],
            hidden_dim=[2 * dim for dim in n_g[(n_f_g if separate_ovc else 0) :]],
            activation=[torch.tanh, torch.exp],
        )

    # ---------------------------------------------------------------------
    # Public API (compatible with model.py)
    # ---------------------------------------------------------------------

    def forward(self, a: Tensor, state: MECState, locations: list[dict]) -> MECState:
        """Compute next MEC state from action-driven transition.

        Args:
            a: One-hot encoded actions (B, n_a). With has_static_action=True,
               action 0 (stand still) is encoded as all-zeros.
            state: Current MEC state
            locations: Per-env location dicts (for shiny detection)

        Returns:
            New MECState with updated g_gen and g_path.

        Note:
            Caller is responsible for resetting state.g to g_init at episode boundaries.
            This module always applies transition dynamics from the provided state.
        """
        shiny_envs = self._shiny_envs(locations)
        trans = self.transition(a, state, no_direc=None)
        g_sample = self._sample_g(trans)

        # Legacy: if ANY shiny env exists, recompute g_gen for ALL envs with no directional drive
        g_gen = self._compute_g_gen(a, state, g_sample, shiny_envs)

        return MECState(g_gen=g_gen, g_path=trans, g=g_sample)

    def transition(self, a: Tensor, state: MECState, no_direc: list[bool] | None = None) -> Transition:
        """Return the transition distribution (mu, sigma) before sampling."""
        mu = self.estimate_next_g_mean(a, state, no_direc=no_direc)
        sigma = self.estimate_next_g_uncertainty(state)
        return Transition(mean=mu, uncertainty=sigma)

    # ---------------------------------------------------------------------
    # Mean / uncertainty (legacy f_mu_g_path / f_sigma_g_path)
    # ---------------------------------------------------------------------

    def estimate_next_g_mean(self, a: Tensor, state: MECState, no_direc: list[bool] | None = None) -> List[Tensor]:
        """Compute transition mean: g_next = g + action_delta."""
        device = a.device
        batch_size = a.shape[0]

        if no_direc is None:
            no_direc = [False] * batch_size

        mats = self._transition_matrices(a, no_direc, device=device)

        g_in = [torch.cat([state.g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from]], dim=1).unsqueeze(1) for f_to in range(self.n_f)]

        delta = [torch.bmm(g_in_f, mat_f).squeeze(1) for g_in_f, mat_f in zip(g_in, mats)]
        g_next = [g_prev_f + delta_f for g_prev_f, delta_f in zip(state.g, delta)]
        g_next = self.f_g_clamp(g_next)

        return g_next

    def estimate_next_g_uncertainty(self, state: MECState) -> List[Tensor]:
        """Compute transition uncertainty from current state."""
        return self.MLP_sigma_g_path(state.g)

    # ---------------------------------------------------------------------
    # Helper functions (small + explicit)
    # ---------------------------------------------------------------------

    def _sample_g(self, trans: Transition) -> List[Tensor]:
        """Sample g from (mu, sigma) if enabled (legacy behavior)."""
        if not self.do_sample:
            return trans.mean
        return [mu + sigma * torch.randn_like(mu) for mu, sigma in zip(trans.mean, trans.uncertainty)]

    def _shiny_envs(self, locations: list[dict]) -> list[bool]:
        """Return per-env shiny indicator (tolerates missing keys)."""
        return [loc.get("shiny") is not None for loc in locations]

    def _compute_g_gen(self, a: Tensor, state: MECState, g: List[Tensor], shiny_envs: list[bool]) -> List[Tensor]:
        """Compute g_gen for generative pathway (legacy shiny behavior).

        Legacy rule: if ANY env has shiny objects, recompute g_gen for ALL envs
        using non-directional transition weights (per-env mask applied via no_direc).
        Otherwise, g_gen = g (the sampled/mean path integration result).
        """
        if any(shiny_envs):
            return self.estimate_next_g_mean(a, state, no_direc=shiny_envs)
        return g

    def _transition_matrices(self, action_onehot: Tensor, no_direc: list[bool], device: torch.device) -> List[Tensor]:
        """Compute per-frequency transition matrices, applying no-direction rows."""
        d_flat = self.MLP_D_a([action_onehot for _ in range(self.n_f)])

        no_direc_mask = torch.tensor(no_direc, device=device, dtype=torch.bool)
        if torch.any(no_direc_mask):
            for f in range(self.n_f):
                d_no_a = self.D_no_a[f].unsqueeze(0).expand_as(d_flat[f])
                d_flat[f] = torch.where(no_direc_mask.unsqueeze(1), d_no_a, d_flat[f])

        mats: List[Tensor] = []
        for f_to in range(self.n_f):
            in_dim = sum(self.n_g[f_from] for f_from in range(self.n_f) if self.g_connections[f_to][f_from])
            mats.append(d_flat[f_to].reshape(-1, in_dim, self.n_g[f_to]))
        return mats

    def f_g_clamp(self, g: List[Tensor]) -> List[Tensor]:
        """Clamp abstract location activations to [-1, 1] (legacy stability)."""
        return [torch.clamp(g_f, min=-1.0, max=1.0) for g_f in g]
