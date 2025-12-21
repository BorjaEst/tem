from typing import Dict, List, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, optim

from torch_tem.core.model import TEMModel, TEMState
from torch_tem.losses import LossOutput
from torch_tem.types import Observation

from .config.training import TrainingConfig


class TEMLightningModule(L.LightningModule):
    """PyTorch Lightning module for training the Tolman-Eichenbaum Machine (TEM).

    This module implements manual BPTT (Backpropagation Through Time) for training
    TEM with truncated sequences, following the approach from the original paper.
    Automatic optimization is disabled to enable fine-grained control over gradient
    accumulation and clipping.

    Attributes:
        model: The TEM model to train.
        config: Training configuration containing hyperparameters.
    """

    def __init__(self, model: TEMModel, config: TrainingConfig):
        """Initialize the Lightning module.

        Args:
            model: TEM model instance to train.
            config: Training configuration with learning rate, BPTT settings, etc.
        """
        super().__init__()
        self.model = model
        self.config = config

        # Save hyperparameters
        self.save_hyperparameters({**config.model_dump()})
        # self.save_hyperparameters({**model.config.model_dump()})

        # Automatic optimization is disabled to handle BPTT manually if needed,
        # Lightning's TBPTT is deprecated, so manual loop is preferred.
        self.automatic_optimization = False

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        """Configure optimizer and learning rate scheduler.

        Uses Adam optimizer with StepLR scheduler for learning rate decay.

        Returns:
            Tuple containing:
                - List with single Adam optimizer
                - List with single StepLR scheduler
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr_max)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_steps, gamma=self.config.lr_decay_rate)

        return [optimizer], [scheduler]

    def forward(self, x: Observation, locations: List[Dict], a: Optional[Tensor], state: TEMState) -> Tuple[TEMState, LossOutput]:
        """Forward pass through the model and compute loss.

        Args:
            x: Sensory observation at current timestep.
            locations: List of location metadata (e.g., shiny objects) per batch item.
            a: Actions taken (optional for first timestep).
            state: Current model state (HPC, LEC, MEC activations).

        Returns:
            Tuple containing:
                - Updated model state after processing this timestep
                - LossOutput with all loss components
        """
        state = self.model(x, locations, a, state)
        loss_output = self.model.loss(x, state)
        return state, loss_output

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """Execute one training step with manual BPTT over rollout chunks.

        Implements truncated BPTT by splitting sequences into rollout chunks of
        length n_rollout. Each chunk is processed, gradients are computed and
        clipped, then parameters are updated. State is detached between chunks
        to prevent gradient flow across chunk boundaries.

        Args:
            batch: Tuple containing:
                - observations: Sequence of observations [walk_length, batch_size, obs_dim]
                - actions: Sequence of actions [walk_length, batch_size]
                - _: Additional batch data (unused)
            batch_idx: Index of current batch (unused, required by Lightning).
        """
        observations, actions, _ = batch
        optimizer = self.optimizers()
        walk_length = len(observations)

        state = self.model.init_state(observations[0])

        # Explicit BPTT loop over rollout chunks
        for rollout_start in range(0, walk_length, self.config.n_rollout):
            rollout_end = min(rollout_start + self.config.n_rollout, walk_length)

            # Compute
            loss_output, state = self.compute_rollout(observations[rollout_start:rollout_end], actions[rollout_start:rollout_end], state)

            # Optimize
            self.manual_backward(loss_output.total)
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()
            optimizer.zero_grad()

            # Log
            self.log_loss(loss_output, "train", on_step=True, on_epoch=False)
            state = state.detach()

        # Once per batch
        self.log_learning_rate("train")
        self.lr_schedulers().step()

    def compute_rollout(self, observations: Tensor, actions: Tensor, state: TEMState) -> Tuple[LossOutput, TEMState]:
        """Compute loss for a single rollout chunk.

        Args:
            observations: Observation tensor for this rollout [rollout_length, batch_size, obs_dim]
            actions: Action tensor for this rollout [rollout_length, batch_size]
            state: Current model state

        Returns:
            Tuple of (averaged_loss, updated_state)
        """
        rollout_length, batch_size, _ = observations.shape
        step_locations = self.create_step_locations(batch_size)

        # Accumulate losses using + operator
        accumulated = LossOutput.zero()
        for t in range(rollout_length):
            state, loss_output = self.forward(observations[t], step_locations, actions[t], state)
            accumulated = accumulated + loss_output

        # Compute averages using / operator
        averaged = accumulated / rollout_length
        return averaged, state

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """Execute one validation step.

        Computes loss over entire sequence without truncation for accurate
        validation metrics.

        Args:
            batch: Tuple containing observations, actions, and metadata.
            batch_idx: Index of current batch (unused, required by Lightning).
        """
        observations, actions, _ = batch

        # Compute
        loss_output = self.compute_sequence(observations, actions)

        # Log
        self.log_loss(loss_output, "val", on_step=False, on_epoch=True)

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """Execute one test step.

        Computes loss over entire sequence without truncation for final
        model evaluation.

        Args:
            batch: Tuple containing observations, actions, and metadata.
            batch_idx: Index of current batch (unused, required by Lightning).
        """
        observations, actions, _ = batch

        # Compute
        loss_output = self.compute_sequence(observations, actions)

        # Log
        self.log_loss(loss_output, "test", on_step=False, on_epoch=True)

    def compute_sequence(self, observations: Tensor, actions: Tensor) -> LossOutput:
        """Compute loss for an entire sequence (used in validation/test).

        Args:
            observations: Full observation sequence [walk_length, batch_size, obs_dim]
            actions: Full action sequence [walk_length, batch_size]

        Returns:
            Averaged LossOutput over the sequence.
        """
        walk_length, batch_size, _ = observations.shape
        state = self.model.init_state(observations[0])
        step_locations = self.create_step_locations(batch_size)

        # Accumulate losses using + operator
        accumulated = LossOutput.zero()
        for t in range(walk_length):
            state, loss_output = self.forward(observations[t], step_locations, actions[t], state)
            accumulated = accumulated + loss_output

        # Compute average using / operator
        return accumulated / walk_length

    def create_step_locations(self, batch_size: int) -> List[Dict]:
        """Create location metadata for each batch item.

        Currently returns placeholder dictionaries. In full implementation,
        this would extract shiny object locations and other environmental
        metadata from the batch data.

        Args:
            batch_size: Number of items in the batch.

        Returns:
            List of dictionaries containing location metadata, one per batch item.

        TODO: Implement proper shiny object handling from environment data.
        """
        return [{"shiny": None} for _ in range(batch_size)]

    def log_loss(self, loss_output: LossOutput, prefix: str, on_step: bool, on_epoch: bool) -> None:
        """Log all loss components.

        Args:
            loss_output: LossOutput containing all components.
            prefix: Logging prefix ("train", "val", or "test").
            on_step: Whether to log per step.
            on_epoch: Whether to log per epoch.
        """
        prog_bar = prefix == "train"

        # Log total loss (convert Tensor to scalar for TensorBoard)
        total_scalar = loss_output.total.item() if isinstance(loss_output.total, Tensor) else loss_output.total
        self.log(f"{prefix}/loss", total_scalar, on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

        # Log all components (as_dict() already converts to float, but ensure scalars)
        components = loss_output.as_dict()
        for component_name in ["lx", "lg", "lp"]:
            value = components[component_name]
            scalar_value = value.item() if isinstance(value, Tensor) else float(value)
            self.log(f"{prefix}/{component_name}", scalar_value, on_step=on_step, on_epoch=on_epoch)

        # Log regularization terms if present and non-zero
        for reg_name in ["l_reg_g", "l_reg_p"]:
            value = components[reg_name]
            if value != 0.0:
                scalar_value = value.item() if isinstance(value, Tensor) else float(value)
                self.log(f"{prefix}/{reg_name}", scalar_value, on_step=on_step, on_epoch=on_epoch)

    def log_learning_rate(self, prefix: str) -> None:
        """Log current learning rate to tensorboard.

        Args:
            prefix: Logging prefix ("train", "val", or "test").
        """
        current_lr = self.optimizers().param_groups[0]["lr"]
        scalar_lr = float(current_lr) if not isinstance(current_lr, float) else current_lr
        self.log(f"{prefix}/lr", scalar_lr, on_step=True, on_epoch=False)
