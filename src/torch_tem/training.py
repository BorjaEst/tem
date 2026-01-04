"""Training loop and utilities for TEM."""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from torch_tem import losses, metrics
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
        self.prev_state: Optional[TEMState] = None

        # Instantiate loss modules once (reused across all steps)
        self.loss_fn = losses.TEMLoss(scheduling.loss)
        self.acc_x_fn = metrics.SensoryAccuracy(reduction="none")

    def forward(self, batch, prev_state=None) -> tuple[LossOutput, AccuracyX, TEMState]:
        """Forward pass through TEM with streaming loss computation."""
        chunk, visited = batch

        # Accumulate loss across timesteps (streaming, no state list)
        accum = AccumLoss.zero(device=self.device)
        acc_correct = {"p": 0.0, "g": 0.0, "gt": 0.0}
        acc_total = 0

        # Single-pass rollout with on-the-fly loss computation
        for step in Rollout(self.tem, chunk, prev_state):
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
        final_acc = AccuracyX(
            p=torch.tensor(acc_correct["p"] / acc_total, device=self.device),
            g=torch.tensor(acc_correct["g"] / acc_total, device=self.device),
            gt=torch.tensor(acc_correct["gt"] / acc_total, device=self.device),
        )

        return accum, final_acc, step

    def init_state(self, batch: Any, memory: Optional[list[Tensor]] = None) -> TEMState:
        """Create a clean initial state for evaluation (optionally preserving memory)."""
        chunk, _visited = batch
        if len(chunk) == 0:
            raise ValueError("init_state requires a non-empty chunk")

        locations_0, x_0, _a_0 = chunk[0]
        batch_size = int(x_0.shape[0])

        return self.tem.init_iteration(locations_0, x_0, [None for _ in range(batch_size)], memory)

    def model_iteration(self, step: TEMState, visited) -> tuple[Optional[StepLoss], dict[str, float]]:
        """Process a single rollout step: compute loss and update accuracies.

        This applies a revisit-only gating policy: only environments that have
        previously visited their current location contribute to the loss.
        """
        use_p_inf = self.tem.hyper["use_p_inf"]
        step_losses = self.loss_fn(step, use_p_inf)
        step_acc = self.acc_x_fn(step.x_logits, step.x)

        # Collect per-env contributions
        losses_per_env: list[StepLoss] = []
        acc_total = {"p": 0.0, "g": 0.0, "gt": 0.0, "total": 0}

        for env_i, env_visited in enumerate(visited):
            loc_id = step.g[env_i]["id"]

            if not env_contributes_and_update(env_visited, loc_id):
                continue  # First visit: skip loss, mark visited

            # Revisit: include in loss and accuracy
            losses_per_env.append(env_step_loss(step_losses, env_i))

            acc_inc = env_acc_increments(step_acc, env_i)
            acc_total["p"] += acc_inc["p"]
            acc_total["g"] += acc_inc["g"]
            acc_total["gt"] += acc_inc["gt"]
            acc_total["total"] += acc_inc["total"]

        return mean_step_losses(losses_per_env), acc_total

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single training step."""
        loss_output, accuracies, state = self(batch, self.prev_state)
        self.prev_state = state.detach()

        self._log_step_metrics(prefix="", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="", accuracies=accuracies)
        return loss_output.total

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single validation step."""
        memory = self.prev_state.M if self.prev_state is not None else None
        init_state = self.init_state(batch, memory)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="val/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="val/", accuracies=accuracies)
        return loss_output.total

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single test step."""
        memory = self.prev_state.M if self.prev_state is not None else None
        init_state = self.init_state(batch, memory)
        loss_output, accuracies, _ = self(batch, init_state)

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


def select_env(t: Tensor, env_i: int) -> Tensor:
    """Extract value for environment `env_i` from tensor `t`.

    If `t` is scalar (reduction="sum"/"mean"), return it unchanged.
    Otherwise index into the batch dimension.
    """
    return t if t.ndim == 0 else t[env_i]


def env_contributes_and_update(visited_env: list[bool], loc_id: int) -> bool:
    """Check if env should contribute loss this step; update visited in-place.

    Returns:
        True if location was already visited (include in loss).
        False if first visit (exclude from loss, mark as visited).
    """
    if visited_env[loc_id]:
        return True
    visited_env[loc_id] = True
    return False


def env_step_loss(step_losses: LossOutput, env_i: int) -> StepLoss:
    """Build per-environment loss object from batch-level structured losses."""
    return StepLoss(
        x=LossX(
            infer=select_env(step_losses.x.infer, env_i),
            retrieved=select_env(step_losses.x.retrieved, env_i),
            ancestral=select_env(step_losses.x.ancestral, env_i),
        ),
        p=LossP(
            abstract=select_env(step_losses.p.abstract, env_i),
            sensory=select_env(step_losses.p.sensory, env_i),
        ),
        g=LossG(transition=select_env(step_losses.g.transition, env_i)),
        reg=LossReg(
            g_l2=select_env(step_losses.reg.g_l2, env_i),
            p_l1=select_env(step_losses.reg.p_l1, env_i),
        ),
    )


def env_acc_increments(step_acc: AccuracyX, env_i: int) -> dict[str, float]:
    """Extract scalar accuracy increments for one environment."""
    return {
        "p": select_env(step_acc.p, env_i).item(),
        "g": select_env(step_acc.g, env_i).item(),
        "gt": select_env(step_acc.gt, env_i).item(),
        "total": 1,
    }


def mean_step_losses(losses_per_env: list[StepLoss]) -> Optional[StepLoss]:
    """Compute mean of per-env losses; return None if list is empty."""
    if not losses_per_env:
        return None
    total = losses_per_env[0]
    for loss in losses_per_env[1:]:
        total = total + loss
    return total / len(losses_per_env)
