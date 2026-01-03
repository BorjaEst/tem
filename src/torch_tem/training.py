"""Training loop and utilities for TEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Adam

from torch_tem.core.model import TEMModel, TEMState
from torch_tem.settings import ScheduleSettings, TrainerSettings

# Loss component names (order matches model.loss() output and schedule loss_weights)
LOSS_NAMES = ("p_g", "p_x", "x_gen", "x_g", "x_p", "g", "reg_g", "reg_p")


class TEMLightningModule(pl.LightningModule):
    """Lightning wrapper for TEM model."""

    def __init__(self, model: TEMModel, scheduling: ScheduleSettings, training: TrainerSettings):
        super().__init__()
        # Store settings for schedule computation
        self.schedule_settings = scheduling
        self.trainer_settings = training

        # Save hyperparameters (namespaced for clarity)
        params = {"schedule": scheduling.model_dump(), "trainer": training.model_dump()}
        self.save_hyperparameters(params)

        # Store TEM model
        self.tem: TEMModel = model
        self.prev_iter: Optional[list[TEMState]] = None

    def forward(self, chunk):
        """Forward pass through TEM."""
        return self.tem(chunk, self.prev_iter)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single training step."""
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single validation step."""
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single test step."""
        return self._shared_step(batch, stage="test")

    def _shared_step(self, batch, stage: str) -> STEP_OUTPUT:
        """Shared logic for train/val/test steps.

        Args:
            batch: Tuple of (chunk, visited).
            stage: One of "train", "val", "test".

        Returns:
            Scalar total loss.

        Note:
            Val/test set runtime hyperparameters (eta, hebbian_decay, p2g_scale_offset)
            to match the schedule at current global_step for deterministic evaluation,
            but use fixed base loss_weights and do not update prev_iter or walk curriculum.
        """
        chunk, visited = batch

        if stage == "train":
            # Compute schedule values for current iteration
            i = self.global_step
            eta_new, hebbian_decay_new, p2g_scale_offset, walk_length_center, loss_weights = self._compute_schedule(i)

            # Update model runtime hyperparameters
            self.tem.set_runtime_hyperparams(
                eta=eta_new,
                hebbian_decay=hebbian_decay_new,
                p2g_scale_offset=p2g_scale_offset,
            )

            # Update dataset's walk_length_center (via datamodule control surface)
            if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "set_walk_length_center"):
                self.trainer.datamodule.set_walk_length_center(walk_length_center)

            loss_weights = loss_weights.to(self.device)
            use_prev_iter = True
            update_prev_iter = True
        else:
            # Val/test: set runtime hyperparameters to match schedule at current step
            # (for deterministic eval), but use fixed base loss weights and no prev_iter
            i = self.global_step
            eta_new, hebbian_decay_new, p2g_scale_offset, _, _ = self._compute_schedule(i)

            self.tem.set_runtime_hyperparams(
                eta=eta_new,
                hebbian_decay=hebbian_decay_new,
                p2g_scale_offset=p2g_scale_offset,
            )

            loss_weights = self.schedule_settings.loss_weights_base.to(self.device)
            use_prev_iter = False
            update_prev_iter = False

        # Compute loss and metrics
        loss_output = self.loss(chunk, visited, loss_weights, use_prev_iter, update_prev_iter)

        # Log metrics with appropriate prefix
        prefix = "" if stage == "train" else f"{stage}/"
        self.log(f"{prefix}loss", loss_output.total, prog_bar=True)
        self.log(f"{prefix}Losses/Total", loss_output.total.detach())
        for idx, name in enumerate(LOSS_NAMES):
            self.log(f"{prefix}Losses/{name}", loss_output.plot_loss[idx])
        self.log(f"{prefix}Accuracies/p", loss_output.acc_p)
        self.log(f"{prefix}Accuracies/g", loss_output.acc_g)
        self.log(f"{prefix}Accuracies/gt", loss_output.acc_gt)

        return loss_output.total

    def loss(self, chunk, visited, loss_weights: torch.Tensor, use_prev_iter: bool = True, update_prev_iter: bool = True) -> LossOutput:
        """Compute loss and metrics for a batch chunk.

        Args:
            chunk: Batch chunk data.
            visited: Visit tracking for environments.
            loss_weights: Loss weights vector [8] (applied to raw loss components).
            use_prev_iter: Whether to condition forward pass on self.prev_iter.
            update_prev_iter: Whether to update self.prev_iter after forward pass.

        Returns:
            LossOutput with total loss, accumulated components, and accuracies.

        Note:
            Both components and plot_loss represent weighted, env-averaged losses
            summed across timesteps. components is device-resident torch tensor,
            plot_loss is CPU numpy array for logging.
        """
        # Forward pass (use prev_iter only if requested, e.g., training)
        prev_state = self.prev_iter if use_prev_iter else None
        forward = self.tem(chunk, prev_state)

        # Guard against empty chunks (shouldn't happen with current dataloader)
        if not forward:
            raise ValueError("Empty chunk received - cannot compute loss")

        # Accumulate loss across timesteps
        total_loss = torch.zeros((), device=self.device)
        components_sum = torch.zeros(8, device=self.device)
        plot_loss = np.zeros(8, dtype=np.float32)

        for step in forward:
            step_loss = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss_vec = torch.zeros(8, device=self.device) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            components_sum = components_sum + step_loss_vec
            plot_loss = plot_loss + step_loss_vec.detach().cpu().numpy()
            total_loss = total_loss + torch.sum(step_loss_vec)

        # Update prev_iter for next training step (only if requested)
        if update_prev_iter:
            self.prev_iter = [forward[-1].detach()]

        # Compute accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        return LossOutput(
            total=total_loss,
            components=components_sum,  # Accumulated weighted components across chunk
            plot_loss=plot_loss,
            acc_p=acc_p,
            acc_g=acc_g,
            acc_gt=acc_gt,
        )

    def configure_optimizers(self):
        """Configure optimizer with dynamic learning rate."""
        optimizer = Adam(self.tem.parameters(), lr=self.schedule_settings.lr_max)

        # Lightning will call optimizer_step where we can update lr dynamically
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step to update learning rate dynamically."""
        # Update learning rate based on current iteration
        lr = self._compute_lr(self.global_step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Call the default optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def _compute_lr(self, iteration: int) -> float:
        """Compute learning rate for given iteration."""
        s = self.schedule_settings
        return max(
            s.lr_min + (s.lr_max - s.lr_min) * (s.lr_decay_rate ** (iteration / s.lr_decay_steps)),
            s.lr_min,
        )

    def _compute_schedule(self, iteration: int) -> tuple[float, float, float, float, torch.Tensor]:
        """Compute all schedule values for current iteration."""
        s = self.schedule_settings
        t = self.trainer_settings

        # Hebbian memory parameters
        eta = min((iteration + 1) / s.eta_it, 1) * s.eta
        lamb = min((iteration + 1) / s.lambda_it, 1) * s.hebbian_decay

        # p->g variance offset schedule
        p2g_scale_offset = 1 / (1 + np.exp((iteration - s.p2g_sig_half_it) / s.p2g_sig_scale_it))

        # Walk length center (annealing from max to min over training)
        max_steps = max(int(t.max_steps), 1)
        walk_length_center = s.walk_it_max - s.walk_it_window * 0.5 - min((iteration + 1) / max_steps, 1) * (s.walk_it_max - s.walk_it_min - s.walk_it_window)

        # Loss weights with annealing
        L_p_g = min((iteration + 1) / s.loss_weights_p_g_it, 1) * s.loss_weights_p
        L_p_x = min((iteration + 1) / s.loss_weights_p_g_it, 1) * s.loss_weights_p * (1 - p2g_scale_offset)
        L_x_gen = s.loss_weights_x
        L_x_g = s.loss_weights_x
        L_x_p = s.loss_weights_x
        L_g = min((iteration + 1) / s.loss_weights_p_g_it, 1) * s.loss_weights_g
        L_reg_g = (1 - min((iteration + 1) / s.loss_weights_reg_g_it, 1)) * s.loss_weights_reg_g
        L_reg_p = (1 - min((iteration + 1) / s.loss_weights_reg_p_it, 1)) * s.loss_weights_reg_p

        loss_weights = torch.tensor([L_p_g, L_p_x, L_x_gen, L_x_g, L_x_p, L_g, L_reg_g, L_reg_p])

        return eta, lamb, p2g_scale_offset, walk_length_center, loss_weights


@dataclass
class LossOutput:
    """Output from loss computation containing total loss, components, and metrics.

    Attributes:
        total: Scalar total loss for backprop (sum of all weighted components).
        components: Weighted loss components tensor [8], summed across timesteps.
                   Represents the chunk's total contribution per component (device-resident).
        plot_loss: Same as components, but as CPU numpy array for logging.
                  Both are weighted by loss_weights and env-averaged.
        acc_p: Grounded location accuracy (%).
        acc_g: Abstract location accuracy (%).
        acc_gt: Ground truth location accuracy (%).
    """

    total: torch.Tensor  # Scalar total loss for backprop
    components: torch.Tensor  # Loss components tensor [8] (device-resident, summed across timesteps)
    plot_loss: np.ndarray  # Accumulated loss components for logging [8] (weighted, env-averaged)
    acc_p: float  # Grounded location accuracy (%)
    acc_g: float  # Abstract location accuracy (%)
    acc_gt: float  # Ground truth location accuracy (%)
