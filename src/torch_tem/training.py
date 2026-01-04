"""Training loop and utilities for TEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from torch_tem import losses, metrics, utils
from torch_tem.core.model import Rollout, TEMModel, TEMState
from torch_tem.losses import AccumLoss, LossG, LossOutput, LossP, LossReg, LossX, StepLoss
from torch_tem.metrics import AccuracyX
from torch_tem.settings import ScheduleSettings, TrainerSettings
from torch_tem.types import Observation


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

        # Instantiate loss modules once (reused across all steps)
        self.loss_fn = losses.TEMLoss(scheduling.loss)
        self.acc_x_fn = metrics.SensoryAccuracy(reduction="none")

    def forward(self, batch, prev_iter=None) -> tuple[LossOutput, AccuracyX, TEMState]:
        """Forward pass through TEM with streaming loss computation."""
        chunk, visited = batch

        # Accumulate loss across timesteps (streaming, no state list)
        accum = AccumLoss.zero(device=self.device)
        acc_correct = {"p": 0.0, "g": 0.0, "gt": 0.0}
        acc_total = 0

        # Single-pass rollout with on-the-fly loss computation
        for step in Rollout(self.tem, chunk, prev_iter):
            # Process this step: compute loss and update accuracies
            step_contrib, acc_increments = self.model_iteration(step, visited)

            # Accumulate loss and accuracies
            if step_contrib is not None:
                accum = accum + step_contrib

            acc_correct["p"] += acc_increments["p"]
            acc_correct["g"] += acc_increments["g"]
            acc_correct["gt"] += acc_increments["gt"]
            acc_total += acc_increments["total"]

        # Compute final averaged accuracies
        if acc_total > 0:
            final_acc = AccuracyX(
                p=torch.tensor(acc_correct["p"] / acc_total, device=self.device),
                g=torch.tensor(acc_correct["g"] / acc_total, device=self.device),
                gt=torch.tensor(acc_correct["gt"] / acc_total, device=self.device),
            )
        else:
            final_acc = AccuracyX(
                p=torch.tensor(0.0, device=self.device),
                g=torch.tensor(0.0, device=self.device),
                gt=torch.tensor(0.0, device=self.device),
            )

        last_state = step.detach()
        return accum, final_acc, last_state

    def model_iteration(self, step: TEMState, visited) -> tuple[Optional[StepLoss], dict[str, float]]:
        """Process a single rollout step: compute loss and update accuracies."""
        # Compute weighted loss components for this timestep (already weighted by TEMLoss)
        use_p_inf = self.tem.hyper["use_p_inf"]
        step_losses = self.loss_fn(step, use_p_inf)

        # Compute accuracies for this timestep (per-env, reduction="none")
        step_accuracies = self.acc_x_fn(step.x_logits, step.x)

        # Apply visited-location filtering (weights already applied by loss_fn)
        step_loss = []
        acc_increments = {"p": 0.0, "g": 0.0, "gt": 0.0, "total": 0}

        for env_i, env_visited in enumerate(visited):
            if env_visited[step.g[env_i]["id"]]:
                # Extract [8] per-env components in LOSS_NAMES order (already weighted)
                env_components = torch.stack(
                    [
                        step_losses.p.abstract[env_i],
                        step_losses.p.sensory[env_i],
                        step_losses.x.ancestral[env_i],
                        step_losses.x.retrieved[env_i],
                        step_losses.x.infer[env_i],
                        step_losses.g.transition[env_i],
                        step_losses.reg.g_l2[env_i],
                        step_losses.reg.p_l1[env_i],
                    ]
                )
                step_loss.append(env_components)

                # Accumulate accuracy for this visited env
                acc_increments["p"] += step_accuracies.p[env_i].item()
                acc_increments["g"] += step_accuracies.g[env_i].item()
                acc_increments["gt"] += step_accuracies.gt[env_i].item()
                acc_increments["total"] += 1
            else:
                env_visited[step.g[env_i]["id"]] = True

        # Average across visited environments and create LossOutput
        if step_loss:
            step_vec = torch.mean(torch.stack(step_loss, dim=0), dim=0)  # [8]
            step_contrib = StepLoss(
                x=LossX(infer=step_vec[4], retrieved=step_vec[3], ancestral=step_vec[2]),
                p=LossP(abstract=step_vec[0], sensory=step_vec[1]),
                g=LossG(transition=step_vec[5]),
                reg=LossReg(g_l2=step_vec[6], p_l1=step_vec[7]),
            )
            return step_contrib, acc_increments

        return None, acc_increments

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single training step."""
        loss_output, accuracies, state = self(batch, self.prev_iter)
        self.prev_iter = [state]  # store state for next iteration

        self._log_step_metrics(prefix="", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="", accuracies=accuracies)
        return loss_output.total

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single validation step."""
        loss_output, accuracies = self(batch)

        self._log_step_metrics(prefix="val/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="val/", accuracies=accuracies)
        return loss_output.total

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single test step."""
        loss_output, accuracies = self(batch)

        self._log_step_metrics(prefix="test/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="test/", accuracies=accuracies)
        return loss_output.total

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update schedules before training step (Lightning hook)."""
        eta, hebbian_decay, p2g_scale_offset, walk_center = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)
        self._maybe_set_walk_length_center(walk_center)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparams before validation step (Lightning hook)."""
        eta, hebbian_decay, p2g_scale_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparams before test step (Lightning hook)."""
        eta, hebbian_decay, p2g_scale_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Update learning rate before optimizer step (Lightning hook)."""
        lr = self._compute_lr(self.global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

    def _maybe_set_walk_length_center(self, walk_length_center: float) -> None:
        """Update datamodule curriculum control surface if present."""
        datamodule = getattr(self.trainer, "datamodule", None)
        setter = getattr(datamodule, "set_walk_length_center", None) if datamodule is not None else None
        if callable(setter):
            setter(walk_length_center)

    def _log_step_metrics(self, *, prefix: str, loss_output: LossOutput) -> None:
        """Log losses + accuracies with an optional prefix."""
        self.log(f"{prefix}loss", loss_output.total, prog_bar=True)
        self.log(f"{prefix}Losses/Total", loss_output.total.detach())
        # Log 8 individual components (legacy names/order)
        self.log(f"{prefix}Losses/p_g", loss_output.p.abstract)
        self.log(f"{prefix}Losses/p_x", loss_output.p.sensory)
        self.log(f"{prefix}Losses/x_gen", loss_output.x.ancestral)
        self.log(f"{prefix}Losses/x_g", loss_output.x.retrieved)
        self.log(f"{prefix}Losses/x_p", loss_output.x.infer)
        self.log(f"{prefix}Losses/g", loss_output.g.transition)
        self.log(f"{prefix}Losses/reg_g", loss_output.reg.g_l2)
        self.log(f"{prefix}Losses/reg_p", loss_output.reg.p_l1)
        # Log grouped losses
        self.log(f"{prefix}Losses/lx", loss_output.x.total)
        self.log(f"{prefix}Losses/lp", loss_output.p.total)
        self.log(f"{prefix}Losses/lg", loss_output.g.total)

    def _log_accuracy_metrics(self, *, prefix: str, accuracies: AccuracyX) -> None:
        """Log sensory prediction accuracies with an optional prefix."""
        self.log(f"{prefix}Accuracies/p", accuracies.p)
        self.log(f"{prefix}Accuracies/g", accuracies.g)
        self.log(f"{prefix}Accuracies/gt", accuracies.gt)

    def configure_optimizers(self):
        """Configure optimizer.

        Note: Learning rate is updated dynamically via on_before_optimizer_step hook.
        """
        return Adam(self.tem.parameters(), lr=self.schedule_settings.lr_max)

    def _compute_lr(self, iteration: int) -> float:
        """Compute learning rate for given iteration."""
        s = self.schedule_settings
        return max(
            s.lr_min + (s.lr_max - s.lr_min) * (s.lr_decay_rate ** (iteration / s.lr_decay_steps)),
            s.lr_min,
        )

    def _compute_schedule(self, iteration: int) -> tuple[float, float, float, float]:
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

        return eta, lamb, p2g_scale_offset, walk_length_center
