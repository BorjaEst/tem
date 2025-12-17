"""PyTorch Lightning Module for TEM training."""

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, optim

from .config.training import TrainingConfig
from .model import TEMModel, TEMState
from .types import Observation


class TEMLightningModule(L.LightningModule):
    """PyTorch Lightning module for training the Tolman-Eichenbaum Machine (TEM).

    This module implements the training loop for TEM using manual optimization
    to handle Backpropagation Through Time (BPTT) with truncation. This is necessary
    because Lightning's automatic TBPTT is deprecated.

    Attributes:
        model (TEMModel): The TEM model to train.
        config (TrainingConfig): Training configuration including learning rate,
            decay schedule, and BPTT rollout length.
    """

    def __init__(self, model: TEMModel, config: TrainingConfig):
        """Initialize the Lightning module for TEM training.

        Args:
            model (TEMModel): The TEM model instance to train.
            config (TrainingConfig): Configuration object containing training
                hyperparameters such as learning rate, decay schedule, and
                rollout length for BPTT.
        """
        super().__init__()
        self.model = model
        self.config = config

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Automatic optimization is disabled to handle BPTT manually if needed,
        # Lightning's TBPTT is deprecated, so manual loop is preferred.
        self.automatic_optimization = False

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
        """Configure optimizer and learning rate scheduler.

        Sets up an Adam optimizer with the maximum learning rate specified in
        the configuration, and a StepLR scheduler for exponential decay.

        Returns:
            Tuple[List[optim.Optimizer], List[optim.lr_scheduler._LRScheduler]]:
                A tuple containing a list with the optimizer and a list with the
                learning rate scheduler.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr_max)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_steps, gamma=self.config.lr_decay_rate)

        return [optimizer], [scheduler]

    def forward(self, x: Observation, locations: List[Dict], a: Optional[Tensor], state: TEMState) -> TEMState:
        """Forward pass through the TEM model.

        Performs a single step inference and generation in the TEM model,
        updating the internal state based on current observation and action.

        Args:
            x (Observation): Current sensory observation at time step t.
            locations (List[Dict]): List of location dictionaries containing
                environment-specific information such as shiny object presence.
            a (Optional[Tensor]): Action taken at the previous time step. None
                for the first step of a sequence.
            state (TEMState): Current state of the TEM model including grounded
                and abstract location representations.

        Returns:
            TEMState: Updated state after processing the observation and action.
        """
        return self.model(x, locations, a, state)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        """Execute a single training step with truncated BPTT.

        Processes a batch of sequential data through the TEM model using
        Backpropagation Through Time (BPTT) with truncation. The method
        manually handles optimization to allow for gradient truncation at
        specified rollout intervals, preventing gradient explosion in long
        sequences.

        The training process:
        1. Initializes model state for the batch
        2. Iterates through each time step in the sequence
        3. Accumulates loss over n_rollout steps
        4. Performs backward pass and optimizer step at truncation points
        5. Detaches state to truncate gradients
        6. Logs training metrics

        Args:
            batch (Tuple[Tensor, Tensor, Tensor]): Tuple containing:
                - observations (Tensor): Sensory observations of shape
                  (batch_size, walk_length, observation_dim)
                - actions (Tensor): Actions taken at each step of shape
                  (batch_size, walk_length)
                - _ (Tensor): Additional batch data (currently unused)
            batch_idx (int): Index of the current batch in the epoch.

        Returns:
            None: Metrics are logged internally via self.log().
        """
        observations, actions, _ = batch
        optimizer = self.optimizers()

        # Process the entire walk sequence with BPTT
        loss_output = self.walk_sequence(observations, actions, optimizer)

        # Log final metrics and update scheduler
        self.log_training_metrics(loss_output)

        # Update learning rate scheduler if present
        scheduler = self.lr_schedulers()
        if scheduler is not None:
            scheduler.step()

    def walk_sequence(self, observations: Tensor, actions: Tensor, optimizer: optim.Optimizer) -> Any:
        """Process a complete walk sequence with truncated BPTT.

        Args:
            observations (Tensor): Sequence of observations with shape
                (batch_size, walk_length, observation_dim).
            actions (Tensor): Sequence of actions with shape
                (batch_size, walk_length).
            optimizer (optim.Optimizer): The optimizer for gradient updates.

        Returns:
            Any: Loss output from the final time step.
        """
        batch_size, walk_length, _ = observations.shape

        # Initialize model state and environment locations
        state = self.model.init_state(observations)
        step_locations = self.create_step_locations(batch_size)

        # Process each time step with BPTT truncation
        optimizer.zero_grad()
        n_rollout = self.config.n_rollout
        accumulated_loss = 0.0

        for t in range(walk_length):
            # Forward pass through model for current time step
            state = self.model(observations[t], step_locations, actions[t], state)
            loss_output = self.model.loss(observations[t], state)
            accumulated_loss += loss_output.total

            # Check if we should perform BPTT truncation
            if self.should_truncate_bptt(t, walk_length, n_rollout):
                state = self.bptt_step(accumulated_loss, n_rollout, optimizer, state)
                accumulated_loss = 0.0

        return loss_output

    @staticmethod
    def should_truncate_bptt(time_step: int, walk_length: int, n_rollout: int) -> bool:
        """Determine if BPTT should be truncated at current time step.

        BPTT truncation occurs either at regular rollout intervals or at
        the end of the sequence to ensure gradients are computed for all steps.

        Args:
            time_step (int): Current time step (0-indexed).
            walk_length (int): Total length of the walk sequence.
            n_rollout (int): Number of steps between truncations.

        Returns:
            bool: True if truncation should occur, False otherwise.
        """
        is_rollout_boundary = (time_step + 1) % n_rollout == 0
        is_sequence_end = (time_step + 1) == walk_length
        return is_rollout_boundary or is_sequence_end

    def create_step_locations(self, batch_size: int) -> List[Dict]:
        """Create location dictionaries for each batch item.

        Args:
            batch_size (int): Number of sequences in the batch.

        Returns:
            List[Dict]: List of location dictionaries with shiny object info.
        """
        # TODO: Implement proper shiny object handling from environment
        return [{"shiny": None} for _ in range(batch_size)]

    def bptt_step(self, accumulated_loss: Tensor, n_rollout: int, optimizer: optim.Optimizer, state: TEMState) -> TEMState:
        """Perform backward pass and optimization step for BPTT.

        Computes gradients on the accumulated loss, clips them to prevent
        explosion, updates parameters, and detaches the state to truncate
        the computational graph.

        Args:
            accumulated_loss (Tensor): Sum of losses over the rollout period.
            n_rollout (int): Number of steps in the rollout for normalization.
            optimizer (optim.Optimizer): The optimizer for parameter updates.
            state (TEMState): Current model state to detach after update.

        Returns:
            TEMState: Detached state with gradients truncated.
        """
        # Compute average loss over rollout period
        avg_loss = accumulated_loss / n_rollout

        # Backward pass with gradient clipping
        self.manual_backward(avg_loss)
        self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")

        # Update parameters and reset gradients
        optimizer.step()
        optimizer.zero_grad()

        # Detach state to truncate backpropagation graph
        return state.detach()

    def log_training_metrics(self, loss_output: Any) -> None:
        """Log training metrics to Lightning logger.

        Args:
            loss_output (Any): Loss output containing total and component losses.
        """
        self.log("train_loss", loss_output.total, prog_bar=True)
        self.log("train_lx", loss_output.lx)
        self.log("train_lg", loss_output.lg)
        self.log("train_lp", loss_output.lp)
