from dataclasses import dataclass
from typing import Optional, Tuple

from torch import Tensor

LossWeights = Tuple[float, float, float, float, float, float, float, float]
"""Weights for the 8 loss components in canonical order.

Order:
    0. L_p_g: Grounded location consistency weight
    1. L_p_x: Sensory-grounded location consistency weight
    2. L_x_gen: Generated observation reconstruction weight
    3. L_x_g: Inferred-abstract observation reconstruction weight
    4. L_x_p: Inferred-grounded observation reconstruction weight
    5. L_g: Abstract location consistency weight
    6. L_reg_g: Abstract location regularization weight
    7. L_reg_p: Grounded location regularization weight
"""


@dataclass(frozen=True)
class Losses:
    """All loss components from a single TEM iteration.

    Attributes:
        L_p_g: Consistency loss between inferred and generated grounded locations
        L_p_x: Consistency loss between sensory-inferred and total-inferred grounded locations
        L_x_gen: Reconstruction loss for observation from generated abstract location
        L_x_g: Reconstruction loss for observation from inferred abstract location
        L_x_p: Reconstruction loss for observation from inferred grounded location
        L_g: Consistency loss between inferred and generated abstract locations
        L_reg_g: L2 regularization on abstract location codes
        L_reg_p: L1 regularization on grounded location codes

    Theory:
        The loss function balances multiple objectives:
        1. Consistency between generative and inference pathways
        2. Accurate sensory reconstruction
        3. Sparsity constraints on representations
    """

    L_p_g: Tensor  # ||p_inf - p_gen||²
    L_p_x: Tensor  # ||p_inf - p_x||²
    L_x_gen: Tensor  # CE(x, x_gen)
    L_x_g: Tensor  # CE(x, x_g)
    L_x_p: Tensor  # CE(x, x_p)
    L_g: Tensor  # ||g_inf - g_gen||²
    L_reg_g: Tensor  # ||g||²
    L_reg_p: Tensor  # ||p||₁

    def total(self, weights: Optional[LossWeights] = None) -> Tensor:
        """Compute weighted sum of all loss components.

        Parameters:
            weights: Optional tuple of 8 weights for each loss component.
                    If None, uses equal weighting (1.0 for all components).

        Returns:
            Total scalar loss tensor.
        """
        if weights is None:
            weights = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        return (
            weights[0] * self.L_p_g
            + weights[1] * self.L_p_x
            + weights[2] * self.L_x_gen
            + weights[3] * self.L_x_g
            + weights[4] * self.L_x_p
            + weights[5] * self.L_g
            + weights[6] * self.L_reg_g
            + weights[7] * self.L_reg_p
        )
