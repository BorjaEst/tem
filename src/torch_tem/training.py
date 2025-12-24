"""TODO: Module docstring."""

from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, optim

from torch_tem.config import TrainingConfig
from torch_tem.core import TEMModel, TEMState
from torch_tem.data import Environment, TEMDataModule
from torch_tem.losses import LossOutput, TEMLoss, TEMLossConfig
from torch_tem.types import Observation, WalkBatch


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

        # Apply training-time loss weights to the model's loss aggregator.
        # TEMModel defaults to TEMLoss() with TEMLossConfig defaults; without
        # this wiring, TrainingConfig loss weights would have no effect.
        self.model.loss_total_fn = TEMLoss(config)

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

    @property
    def environment(self) -> Environment:
        """Environment instance from the attached DataModule.

        Returns:
            Environment: The environment used for data generation.
        """
        datamodule: TEMDataModule = self.trainer.datamodule  # type: ignore[attr-defined]
        assert datamodule is not None and datamodule.environment is not None, "DataModule or environment not set."
        return datamodule.environment

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
        return state, self.model.loss(x, state)

    def training_step(self, batch: WalkBatch, batch_idx: int):
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
        optimizer = self.optimizers()
        schedulers = self.lr_schedulers()

        # Explicit BPTT loop over rollout chunks
        for loss_output, state in Rollout(self.model, batch, self.environment):

            # Optimization step
            self.manual_backward(loss_output.total)  # Backpropagate loss
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()  # Optimizer step
            optimizer.zero_grad()  # Reset gradients
            schedulers.step()

            # Log
            self.log_loss(loss_output, "train", on_step=True, on_epoch=False)
            self.log_learning_rate("train")
            state = state.detach()

    def validation_step(self, batch: WalkBatch, batch_idx: int):
        """Execute one validation step.

        Computes loss over entire sequence without truncation for accurate
        validation metrics.

        Args:
            batch: Tuple containing observations, actions, and metadata.
            batch_idx: Index of current batch (unused, required by Lightning).
        """
        # Compute without gradient tracking
        with torch.no_grad():
            loss_output, _ = list(Rollout(self.model, batch, self.environment))[-1]

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
        # Compute without gradient tracking
        with torch.no_grad():
            loss_output, _ = list(Rollout(self.model, batch, self.environment))[-1]

        # Log
        self.log_loss(loss_output, "test", on_step=False, on_epoch=True)

    def log_loss(self, loss_output: LossOutput, prefix: str, on_step: bool, on_epoch: bool) -> None:
        """Log all loss components.

        Args:
            loss_output: LossOutput containing all components.
            prefix: Logging prefix ("train", "val", or "test").
            on_step: Whether to log per step.
            on_epoch: Whether to log per epoch.
        """
        prog_bar = prefix == "train"
        components = loss_output.as_dict()

        # Log total loss (convert Tensor to scalar for TensorBoard)
        self.log(f"{prefix}/loss", loss_output.total.item(), on_step=on_step, on_epoch=on_epoch, prog_bar=prog_bar)

        # Log all components (as_dict() already converts to float, but ensure scalars)
        for component_name in ["lx", "lg", "lp"]:
            value = components[component_name]
            self.log(f"{prefix}/{component_name}", value.item(), on_step=on_step, on_epoch=on_epoch)

        # Log regularization terms if present and non-zero
        for reg_name in ["l_reg_g", "l_reg_p"]:
            if (value := components[reg_name]) == 0.0:
                continue
            self.log(f"{prefix}/{reg_name}", value.item(), on_step=on_step, on_epoch=on_epoch)

    def log_learning_rate(self, prefix: str) -> None:
        """Log current learning rate to tensorboard.

        Args:
            prefix: Logging prefix ("train", "val", or "test").
        """
        current_lr = self.optimizers().param_groups[0]["lr"]
        scalar_lr = float(current_lr) if not isinstance(current_lr, float) else current_lr
        self.log(f"{prefix}/lr", scalar_lr, on_step=True, on_epoch=False)


class Rollout(Iterator[Tuple[LossOutput, TEMState]]):
    """Iterator that yields (loss_output, state) tuples for each timestep.

    Processes a walk sequence one timestep at a time, computing losses.
    Each iteration returns the loss for that timestep and the updated state.

    Example:
        >>> rollout = Rollout(model, batch)
        >>> for loss_output, state in rollout:
        ...     optimizer.zero_grad()
        ...     loss_output.total.backward()
        ...     optimizer.step()
    """

    def __init__(self, model: TEMModel, batch: WalkBatch, environment: Environment):
        """Initialize rollout iterator.

        Args:
            model: The TEM model to run.
            batch: Tuple of (observations, actions, locations) tensors.
                   observations: [walk_length, batch_size, obs_dim]
                   actions: [walk_length, batch_size]
                   locations: [walk_length, batch_size]
            environment: Environment instance for location metadata lookup.
        """
        self.model = model
        self.observations, self.actions, self.locations = batch
        self.environment = environment

        self.walk_length = self.observations.shape[0]
        self.batch_size = self.observations.shape[1]

        # Initialize state and iteration position
        self.state = model.init_state(self.observations[0])
        self.current_t = 0

    def __iter__(self) -> "Rollout":
        """Return self as iterator."""
        return self

    def __next__(self) -> Tuple[LossOutput, TEMState]:
        """Process next timestep and return (loss_output, state).

        Returns:
            Tuple of (loss for current timestep, updated state).

        Raises:
            StopIteration: When the entire sequence has been processed.
        """
        if self.current_t >= self.walk_length:
            raise StopIteration
        t = self.current_t

        # Get step locations - convert tensor to list of dicts
        step_locations = self._create_step_locations(t)
        self.state = self.model(self.observations[t], step_locations, self.actions[t], self.state)
        loss_output = self.model.loss(self.observations[t], self.state)
        self.current_t += 1

        return loss_output, self.state

    def _create_step_locations(self, t: int) -> List[Dict]:
        """Create location metadata for timestep t.

        Args:
            t: Timestep index.

        Returns:
            List of location dicts, one per batch item.
        """
        if self.locations is None:
            return [{"shiny": None} for _ in range(self.batch_size)]

        # Extract location IDs for timestep t: tensor of shape (batch_size,)
        location_ids_t = self.locations[t]
        # Convert to list of integers
        location_ids = location_ids_t.tolist()
        # Use environment to map IDs to location metadata dicts
        return self.environment.step_locations(location_ids)
