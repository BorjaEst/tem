"""Training loop and utilities for TEM."""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch.optim import Adam

from torch_tem.core.model import TEMModel
from torch_tem.settings import ScheduleSettings, TrainerSettings


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
        self.tem = model
        self.prev_iter = None

    def forward(self, chunk):
        """Forward pass through TEM."""
        return self.tem(chunk, self.prev_iter)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single training step."""
        chunk, visited = batch

        # Compute schedule values for current iteration
        i = self.global_step
        eta_new, hebbian_decay_new, p2g_scale_offset, walk_length_center, loss_weights = self._compute_schedule(i)

        # Update model runtime hyperparameters (explicit typed interface)
        self.tem.set_runtime_hyperparams(
            eta=eta_new,
            hebbian_decay=hebbian_decay_new,
            p2g_scale_offset=p2g_scale_offset,
        )

        # Update dataset's walk_length_center (via datamodule control surface)
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "set_walk_length_center"):
            self.trainer.datamodule.set_walk_length_center(walk_length_center)

        # Move loss_weights to device
        loss_weights = loss_weights.to(self.device)

        # Compute loss and metrics
        loss, plot_loss, acc_p, acc_g, acc_gt = self._compute_loss_and_metrics(chunk, visited, loss_weights)

        # Log metrics
        self.log("loss", loss, prog_bar=True)
        self.log("Losses/Total", loss.detach())
        for idx, name in enumerate(["p_g", "p_x", "x_gen", "x_g", "x_p", "g", "reg_g", "reg_p"]):
            self.log(f"Losses/{name}", plot_loss[idx])
        self.log("Accuracies/p", acc_p)
        self.log("Accuracies/g", acc_g)
        self.log("Accuracies/gt", acc_gt)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single validation step."""
        chunk, visited = batch

        # Use fixed schedule values (no curriculum during validation)
        loss_weights = self.schedule_settings.loss_weights_base.to(self.device)

        # Compute loss and metrics (without updating prev_iter)
        loss, plot_loss, acc_p, acc_g, acc_gt = self._compute_loss_and_metrics(chunk, visited, loss_weights, update_prev_iter=False)

        # Log validation metrics
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/Losses/Total", loss.detach())
        for idx, name in enumerate(["p_g", "p_x", "x_gen", "x_g", "x_p", "g", "reg_g", "reg_p"]):
            self.log(f"val/Losses/{name}", plot_loss[idx])
        self.log("val/Accuracies/p", acc_p)
        self.log("val/Accuracies/g", acc_g)
        self.log("val/Accuracies/gt", acc_gt)

        return loss

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single test step."""
        chunk, visited = batch

        # Use fixed schedule values (no curriculum during test)
        loss_weights = self.schedule_settings.loss_weights_base.to(self.device)

        # Compute loss and metrics (without updating prev_iter)
        loss, plot_loss, acc_p, acc_g, acc_gt = self._compute_loss_and_metrics(chunk, visited, loss_weights, update_prev_iter=False)

        # Log test metrics
        self.log("test/loss", loss)
        self.log("test/Losses/Total", loss.detach())
        for idx, name in enumerate(["p_g", "p_x", "x_gen", "x_g", "x_p", "g", "reg_g", "reg_p"]):
            self.log(f"test/Losses/{name}", plot_loss[idx])
        self.log("test/Accuracies/p", acc_p)
        self.log("test/Accuracies/g", acc_g)
        self.log("test/Accuracies/gt", acc_gt)

        return loss

    def _compute_loss_and_metrics(self, chunk, visited, loss_weights: torch.Tensor, update_prev_iter: bool = True) -> tuple[torch.Tensor, np.ndarray, float, float, float]:
        """Shared loss and metrics computation for train/val/test steps.

        Args:
            chunk: Batch chunk data.
            visited: Visit tracking for environments.
            loss_weights: Loss weights vector.
            update_prev_iter: Whether to update self.prev_iter (True for training, False for val/test).

        Returns:
            Tuple of (loss, plot_loss, acc_p, acc_g, acc_gt).
        """
        # Forward pass
        forward = self(chunk)

        # Compute loss
        loss = torch.tensor(0.0, device=self.device)
        plot_loss = 0
        for step in forward:
            step_loss = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]["id"]]:
                    step_loss.append(loss_weights * torch.stack([l[env_i] for l in step.L]))
                else:
                    env_visited[step.g[env_i]["id"]] = True
            step_loss = torch.tensor(0, device=self.device) if not step_loss else torch.mean(torch.stack(step_loss, dim=0), dim=0)
            plot_loss = plot_loss + step_loss.detach().cpu().numpy()
            loss = loss + torch.sum(step_loss)

        # Update prev_iter for next step (only during training)
        if update_prev_iter:
            self.prev_iter = [forward[-1].detach()]

        # Compute accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        return loss, plot_loss, acc_p, acc_g, acc_gt

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
        import numpy as np

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
