from typing import List

import numpy as np
import torch
import torch.nn as nn

from torch_tem import utils


class MemorySystem(nn.Module):
    """Hebbian associative memory with attractor dynamics for pattern completion.

    Implements a biologically-inspired memory system based on:
    1. Hebbian Learning: "Neurons that fire together, wire together"
       - Memory updates via outer product: ΔM = η·(p_inf+p_gen)⊗(p_inf-p_gen)
    2. Attractor Dynamics: Iterative pattern completion via recurrence
       - Converges partial/noisy patterns to stored attractors
    3. Hierarchical Organization: Low→high frequency information flow
       - Allows different modules to stabilize at different rates

    The memory matrix M stores associations between place cell patterns.
    During retrieval, a query pattern is iteratively refined through
    multiplication with M until it converges to a stored pattern.

    Independent module design:
        - No config object dependencies
        - Explicit Hebbian hyperparameters with sensible defaults
        - Clear connectivity mask interface for hierarchical control

    Args:
        n_p_dims: Place cell dimensions per frequency module [n_freq]
            Example: [100, 80, 60, 40, 20] for 5 modules
        lambda_forget: Forgetting rate in [0,1], default=0.9999
            M_new = λ·M_old + η·ΔM  (λ≈1 → slow forgetting)
        eta_remember: Learning rate in [0,1], default=0.5
            Controls strength of new memories
        kappa_decay: Decay factor for attractor dynamics, default=0.8
            h_new = mask·activation(κ·h_old + M·h_old)
        n_attractor_iters: Number of attractor iterations, default=5
            More iterations → stronger pattern completion
        p_update_mask: Connectivity mask [n_p_total x n_p_total] for hierarchical updates
            None → flat (all-to-all) connectivity
        p_retrieve_mask_inf: List of retrieval masks [n_attractor_iters x n_p_total]
            Controls which modules update at each iteration (inference mode)
        p_retrieve_mask_gen: List of retrieval masks [n_attractor_iters x n_p_total]
            Controls which modules update at each iteration (generative mode)

    Attributes:
        n_p_dims: Place cell dimensions per frequency
        n_freq: Number of frequency modules
        lambda_: Forgetting rate (0=total forgetting, 1=no forgetting)
        eta: Learning rate for new memories
        kappa: Decay factor in attractor dynamics
        i_attractor: Number of attractor iterations
        p_update_mask: Hierarchical connectivity mask for updates
        p_retrieve_mask_inf: Masks for inference retrieval
        p_retrieve_mask_gen: Masks for generative retrieval
    """

    def __init__(
        self,
        n_p_dims: List[int],  # Place cell dimensions per frequency
        lambda_forget: float = 0.9999,  # Forgetting rate
        eta_remember: float = 0.5,  # Remembering rate
        kappa_decay: float = 0.8,  # Retrieval decay
        n_attractor_iters: int = 5,  # Attractor iterations
        p_update_mask: torch.Tensor = None,  # Hierarchical update mask
        p_retrieve_mask_inf: List[torch.Tensor] = None,  # Inference retrieval masks
        p_retrieve_mask_gen: List[torch.Tensor] = None,  # Generative retrieval masks
    ):
        super().__init__()
        self.n_p_dims = n_p_dims
        self.n_freq = len(n_p_dims)

        # Hebbian learning parameters
        self.lambda_ = lambda_forget  # Forgetting: higher → retain more
        self.eta = eta_remember  # Learning: higher → learn faster
        self.kappa = kappa_decay  # Decay: lower → more recurrence influence
        self.i_attractor = n_attractor_iters  # Iterations: more → stronger completion

        # Connectivity masks for hierarchical organization
        self.p_update_mask = p_update_mask  # Which connections are allowed in updates
        self.p_retrieve_mask_inf = p_retrieve_mask_inf  # Inference retrieval schedule
        self.p_retrieve_mask_gen = p_retrieve_mask_gen  # Generative retrieval schedule

    def retrieve(self, query: List[torch.Tensor], memory_matrix: torch.Tensor, mode: str = "inference") -> List[torch.Tensor]:
        """Retrieve patterns from memory via attractor dynamics.

        Performs iterative pattern completion where a partial/noisy query
        is refined through repeated application of the memory matrix:
            h[t+1] = mask[t]·activation(κ·h[t] + M·h[t])

        The mask allows hierarchical settling: low-frequency modules
        stabilize first, then high-frequency modules refine details.

        Args:
            query: Partial patterns per frequency [n_freq x [batch x n_p_f]]
                Can come from sensory input, abstract location, etc.
            memory_matrix: Associative memory [batch x n_p_total x n_p_total]
                Stores learned associations between place cell patterns
            mode: "inference" or "generative"
                Determines which retrieval mask schedule to use

        Returns:
            Completed patterns per frequency [n_freq x [batch x n_p_f]]
            Converged attractors representing stored memories
        """
        # Concatenate all frequency modules into single vector
        h_t = torch.cat(query, dim=1)  # [batch x n_p_total]
        h_t = self._activation(h_t)

        # Select appropriate mask schedule for this mode
        masks = self.p_retrieve_mask_inf if mode == "inference" else self.p_retrieve_mask_gen

        # Iterative attractor dynamics
        for tau in range(self.i_attractor):
            # Memory-based update: M·h provides associative recall
            memory_contribution = torch.squeeze(torch.matmul(torch.unsqueeze(h_t, 1), memory_matrix))

            # Combine decay term (κ·h) with memory term (M·h)
            updated = self._activation(self.kappa * h_t + memory_contribution)

            # Apply hierarchical mask: only update allowed modules at this iteration
            h_t = (1 - masks[tau]) * h_t + masks[tau] * updated

        # Split concatenated vector back into per-frequency representations
        n_p_cumsum = np.cumsum(np.concatenate(([0], self.n_p_dims)))
        return [h_t[:, n_p_cumsum[f] : n_p_cumsum[f + 1]] for f in range(self.n_freq)]

    def update(self, memory_prev: torch.Tensor, p_inferred: List[torch.Tensor], p_generated: List[torch.Tensor], hierarchical: bool = True) -> torch.Tensor:
        """Update memory via Hebbian learning rule.

        Hebbian update encodes the association between inference and generative
        pathways using an outer product:
            ΔM = η·(p_inf + p_gen)⊗(p_inf - p_gen)

        The sum (p_inf + p_gen) provides the pattern to store.
        The difference (p_inf - p_gen) provides the prediction error signal.
        This encourages the memory to align inference with generation.

        With forgetting:
            M[t] = clamp(λ·M[t-1] + η·ΔM, min=-1, max=1)

        Args:
            memory_prev: Previous memory matrix [batch x n_p_total x n_p_total]
            p_inferred: Inferred place cell patterns [n_freq x [batch x n_p_f]]
                From inference pathway (observation → memory → place cells)
            p_generated: Generated place cell patterns [n_freq x [batch x n_p_f]]
                From generative pathway (abstract location → memory → place cells)
            hierarchical: If True, apply connectivity mask for hierarchical updates
                Only allows low→high frequency connections

        Returns:
            Updated memory matrix [batch x n_p_total x n_p_total]
            Clamped to [-1, 1] for numerical stability
        """
        # Concatenate all frequencies for batch outer product
        p_inf_flat = torch.cat(p_inferred, dim=1)  # [batch x n_p_total]
        p_gen_flat = torch.cat(p_generated, dim=1)  # [batch x n_p_total]

        # Compute Hebbian update: outer product of sum and difference
        # (p_inf + p_gen)⊗(p_inf - p_gen)
        M_new = torch.squeeze(torch.matmul(torch.unsqueeze(p_inf_flat + p_gen_flat, 2), torch.unsqueeze(p_inf_flat - p_gen_flat, 1)))

        # Apply hierarchical connectivity mask if requested
        # Masks out connections that violate low→high frequency constraint
        if hierarchical and self.p_update_mask is not None:
            M_new = M_new * self.p_update_mask

        # Combine old memory (with forgetting) and new update
        # Clamp to [-1,1] to prevent unbounded growth
        M = torch.clamp(self.lambda_ * memory_prev + self.eta * M_new, min=-1, max=1)
        return M

    def _activation(self, p: torch.Tensor) -> torch.Tensor:
        """Apply leaky ReLU activation to place cell patterns.

        Leaky ReLU maintains some negative values (unlike ReLU), which
        is important for representing prediction errors in Hebbian updates.
        Clamping to [-1,1] ensures numerical stability.

        Args:
            p: Place cell activations [batch x n_p]

        Returns:
            Activated patterns [batch x n_p] in range ≈[-1, 1]
        """
        return utils.leaky_relu(torch.clamp(p, min=-1, max=1))
