"""PyTorch Lightning Module for TEM training."""

from typing import Any, Dict, List, Optional, Tuple

import lightning as L
import torch
from torch import Tensor, optim

from .config.training import TrainingConfig
from .model import TEMModel, TEMState
from .types import Observation


class TEMLightningModule(L.LightningModule):

    def __init__(self, model: TEMModel, config: TrainingConfig):
        super().__init__()
        self.model = model
        self.config = config

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Automatic optimization is disabled to handle BPTT manually if needed,
        # Lightning's TBPTT is deprecated, so manual loop is preferred.
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.config.lr_max)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config.lr_decay_steps, gamma=self.config.lr_decay_rate)

        return [optimizer], [scheduler]

    def forward(self, x: Observation, locations: List[Dict], a: Optional[Tensor], state: TEMState) -> TEMState:
        return self.model(x, locations, a, state)

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int):
        observations, actions, _ = batch
        batch_size, walk_length, _ = observations.shape
        optimizers = self.optimizers()
        optimizers.zero_grad()

        # Initialize state and locations
        state = self.model.init_state(observations)
        step_locations = [{"shiny": None} for _ in range(batch_size)]  # TODO: handle shiny objects

        # BPTT variables
        n_rollout = self.config.n_rollout
        accumulated_loss = 0.0

        for t in range(walk_length):
            state = self.model(observations[t], step_locations, actions[t], state)
            loss_output = self.model.loss(observations[t], state)
            accumulated_loss += loss_output.total

            # BPTT Truncation
            if (t + 1) % n_rollout == 0 or (t + 1) == walk_length:
                self.manual_backward(accumulated_loss / n_rollout)
                self.clip_gradients(optimizers, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                optimizers.step()  # Step optimizer
                optimizers.zero_grad()
                state = state.detach()  # Detach state for BPTT
                accumulated_loss = 0.0

        # Log metrics (using the last step's loss components for simplicity, or average)
        self.log("train_loss", loss_output.total, prog_bar=True)
        self.log("train_lx", loss_output.lx)
        self.log("train_lg", loss_output.lg)
        self.log("train_lp", loss_output.lp)

        # Update learning rate scheduler if needed
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            schedulers.step()
