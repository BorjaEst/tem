"""TODO: Module docstring."""

from collections.abc import Iterator
from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, optim

from torch_tem import losses
from torch_tem.config import TrainingConfig
from torch_tem.core import TEMModel, TEMState
from torch_tem.data import Environment, TEMDataModule
from torch_tem.losses import LossOutput
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

        # Initialize loss components
        self.loss_x_fn = losses.SensoryReconstructionLoss()
        self.loss_p_fn = losses.GroundedLocationLoss()
        self.loss_g_fn = losses.AbstractLocationLoss()
        self.loss_reg_fn = losses.RegularizationLoss()
        self.loss_total_fn = losses.TEMLoss(config.loss)

        # Save hyperparameters
        self.save_hyperparameters({**config.model_dump()})
        # self.save_hyperparameters({**model.config.model_dump()})

        # Automatic optimization is disabled to handle BPTT manually
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
        return state, self.loss(x, state)

    def loss(self, x: Observation, state: TEMState) -> losses.LossOutput:
        """Compute Evidence Lower Bound (ELBO) loss for TEM.

        The total loss comprises three components following the TEM paper:
        1. L_x: Sensory reconstruction loss (from three pathways)
        2. L_p: Grounded location consistency loss
        3. L_g: Abstract location KL divergence loss

        Args:
            x: Ground truth sensory observation.
            state: Current TEM state with all pathway outputs.

        Returns:
            LossOutput containing total loss and individual components.
        """
        # Extract grounded locations from TEM state for loss computation
        p_x, p_g, p = state.grounded
        g_gen = self.model.mec.projection(state.mec.transition_stats.mean)  # Project predicted abstract location
        p_gen = self.model.hpc.retrieve(g_gen, for_inference=False, state=state.hpc)  # Retrieve from generative memory

        # L_x: Sensory reconstruction from three pathways (teacher forcing)
        Lx = [
            # self.loss_x_fn(prediction=self.lec.decode(p_x), target=x),  # From sensory retrieval
            self.loss_x_fn(prediction=self.model.lec.decode(p_g), target=x),  # From abstract retrieval
            self.loss_x_fn(prediction=self.model.lec.decode(p), target=x),  # From inference
            self.loss_x_fn(prediction=self.model.lec.decode(p_gen), target=x),  # From generative prediction
        ]
        # L_p: Grounded location consistency (inference matches memory retrieval)
        Lp = self.loss_p_fn(p=p, p_g=p_g, p_x=p_x)
        # L_g: Abstract location KL divergence (posterior vs prior)
        Lg = self.loss_g_fn(g=state.abstract_location, g_gen=state.transition_stats)

        # Regularization losses
        L_reg_g, L_reg_p = self.loss_reg_fn(g=state.abstract_location, p=p)

        # Compute total ELBO
        return self.loss_total_fn(sum(Lx), Lp, Lg, L_reg_g, L_reg_p)

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
        for loss_output, _ in Rollout(self, batch):

            # Optimization step
            self.manual_backward(loss_output.total)  # Backpropagate loss
            self.clip_gradients(optimizer, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
            optimizer.step()  # Optimizer step
            optimizer.zero_grad()  # Reset gradients
            schedulers.step()

            # Log
            self.log_loss(loss_output, "train", on_step=True, on_epoch=False)
            self.log_learning_rate("train")

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
            # Use full sequence length for validation (no truncation)
            loss_output, _ = list(Rollout(self, batch))[-1]

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
            # Use full sequence length for test (no truncation)
            walk_length = batch[0].shape[0]
            loss_output, _ = list(Rollout(self, batch))[-1]

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

        # Log all components (as_dict() already converts to float scalars)
        for component_name in ["lx", "lg", "lp"]:
            value = components[component_name]
            self.log(f"{prefix}/{component_name}", value, on_step=on_step, on_epoch=on_epoch)

        # Log regularization terms if present and non-zero
        for reg_name in ["l_reg_g", "l_reg_p"]:
            if (value := components[reg_name]) == 0.0:
                continue
            self.log(f"{prefix}/{reg_name}", value, on_step=on_step, on_epoch=on_epoch)

    def log_learning_rate(self, prefix: str) -> None:
        """Log current learning rate to tensorboard.

        Args:
            prefix: Logging prefix ("train", "val", or "test").
        """
        current_lr = self.optimizers().param_groups[0]["lr"]
        scalar_lr = float(current_lr) if not isinstance(current_lr, float) else current_lr
        self.log(f"{prefix}/lr", scalar_lr, on_step=True, on_epoch=False)


class Rollout(Iterator[Tuple[LossOutput, TEMState]]):
    """Iterator that yields accumulated (loss_output, state) for each rollout chunk.

    Processes a walk sequence in chunks of n_rollout timesteps, computing and
    accumulating losses over each chunk. This implements truncated BPTT.

    Example:
        >>> rollout = Rollout(model, batch, environment, n_rollout=20)
        >>> for loss_output, state in rollout:
        ...     optimizer.zero_grad()
        ...     loss_output.total.backward()
        ...     optimizer.step()
    """

    def __init__(self, module: TEMLightningModule, batch: WalkBatch):
        """Initialize rollout iterator.

        Args:
            module: TEMLightningModule containing model and loss.
            batch: Tuple of (observations, actions, locations) tensors.
                   observations: [walk_length, batch_size, obs_dim]
                   actions: [walk_length, batch_size]
                   locations: [walk_length, batch_size]
        """
        self.module = module
        self.observations, self.actions, self.locations = batch
        self.walk_length = self.observations.shape[0]
        self.batch_size = self.observations.shape[1]

        # Initialize state and iteration position
        self.state = module.model.init_state(self.observations[0])
        self.current_t = 0

    def __iter__(self) -> "Rollout":
        """Return self as iterator."""
        return self

    @property
    def environment(self) -> Environment:
        """Environment instance from the attached DataModule.

        Returns:
            Environment: The environment used for data generation.
        """
        datamodule: TEMDataModule = self.module.trainer.datamodule  # type: ignore[attr-defined]
        assert datamodule is not None and datamodule.environment is not None, "DataModule or environment not set."
        return datamodule.environment

    @property
    def model(self) -> TEMModel:
        """TEM model from module."""
        return self.module.model

    @property
    def n_rollout(self) -> int:
        """Rollout length from module configuration."""
        return self.module.config.n_rollout

    def __next__(self) -> Tuple[LossOutput, TEMState]:
        """Process next n_rollout timesteps and return accumulated (loss_output, state).

        Returns:
            Tuple of (accumulated loss for this chunk, final state after chunk).

        Raises:
            StopIteration: When the entire sequence has been processed.
        """
        if self.current_t >= self.walk_length:
            raise StopIteration

        # Determine chunk boundaries and initialize accumulated loss
        chunk_start = self.current_t
        chunk_end = min(chunk_start + self.n_rollout, self.walk_length)
        accumulated_loss = LossOutput.zero()

        # Forward pass
        for t in range(chunk_start, chunk_end):
            step_locations = self._create_step_locations(t)
            self.state, loss_output = self.module(self.observations[t], step_locations, self.actions[t], self.state)
            accumulated_loss = accumulated_loss + loss_output
        self.current_t = chunk_end

        # Detach state to prevent backprop and return
        self.state = self.state.detach()
        return accumulated_loss, self.state

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
