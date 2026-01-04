"""Training loop and utilities for TEM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from torch_tem import losses, utils
from torch_tem.core.model import TEMModel, TEMState
from torch_tem.losses import AccumLoss, LossG, LossOutput, LossP, LossReg, LossX, StepLoss
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
        self.loss_x_fn = losses.SensoryReconstructionLoss(reduction="none")
        self.loss_p_fn = losses.GroundedLocationLoss(reduction="none")
        self.loss_g_fn = losses.AbstractLocationLoss(mode="mse", reduction="none")
        self.loss_reg_fn = losses.RegularizationLoss(reduction="none")

    def forward(self, chunk):
        """Forward pass through TEM."""
        return self.tem(chunk, self.prev_iter)

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single training step."""
        chunk, visited = batch
        i = self.global_step

        # Update model runtime hyperparameters and datamodule curriculum
        eta_new, hebbian_decay_new, p2g_scale_offset, walk_length_center, loss_weights = self._compute_schedule(i)
        self.tem.set_runtime_hyperparams(eta_new, hebbian_decay_new, p2g_scale_offset)
        self._maybe_set_walk_length_center(walk_length_center)

        loss_output = self.loss(chunk, visited, loss_weights.to(self.device), use_prev_iter=True, update_prev_iter=True)
        self._log_step_metrics(prefix="", loss_output=loss_output)
        return loss_output.total

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single validation step."""
        chunk, visited = batch
        i = self.global_step

        # Keep eval deterministic by matching runtime hparams to current global_step.
        eta_new, hebbian_decay_new, p2g_scale_offset, _, _ = self._compute_schedule(i)
        self.tem.set_runtime_hyperparams(eta_new, hebbian_decay_new, p2g_scale_offset)

        loss_weights = self.schedule_settings.loss_weights_base
        loss_output = self.loss(chunk, visited, loss_weights.to(self.device), use_prev_iter=False, update_prev_iter=False)
        self._log_step_metrics(prefix="val/", loss_output=loss_output)
        return loss_output.total

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Single test step."""
        chunk, visited = batch
        i = self.global_step

        # Keep eval deterministic by matching runtime hparams to current global_step.
        eta_new, hebbian_decay_new, p2g_scale_offset, _, _ = self._compute_schedule(i)
        self.tem.set_runtime_hyperparams(eta_new, hebbian_decay_new, p2g_scale_offset)

        loss_weights = self.schedule_settings.loss_weights_base
        loss_output = self.loss(chunk, visited, loss_weights.to(self.device), use_prev_iter=False, update_prev_iter=False)
        self._log_step_metrics(prefix="test/", loss_output=loss_output)
        return loss_output.total

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

    def loss(self, chunk, visited, loss_weights: torch.Tensor, use_prev_iter: bool = True, update_prev_iter: bool = True) -> AccumLoss:
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

        # Get use_p_inf flag from model (controls L_p_x computation)
        use_p_inf = self.tem.hyper["use_p_inf"]

        # Accumulate loss across timesteps
        accum = AccumLoss.zero(device=self.device)

        for step in forward:
            # Compute raw loss components for this timestep
            step_losses = self._step_loss(step, use_p_inf)

            # Apply visited-location filtering and weighting
            step_loss = []
            for env_i, env_visited in enumerate(visited):
                if env_visited[step.g[env_i]["id"]]:
                    # Extract [8] per-env components in LOSS_NAMES order and apply weights
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
                    step_loss.append(loss_weights * env_components)
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
                accum = accum + step_contrib

        # Update prev_iter for next training step (only if requested)
        if update_prev_iter:
            self.prev_iter = [forward[-1].detach()]

        return accum

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

    def _step_loss(self, step: TEMState, use_p_inf: bool) -> StepLoss:
        """Compute per-timestep loss using modular loss components.

        Args:
            step: TEMState containing model outputs for this timestep.
            use_p_inf: Whether to compute L_p_x (sensory consistency term).

        Returns:
            StepLoss with all components computed via loss modules.
        """
        # Use instantiated loss modules (with reduction="none" for per-env outputs)
        L_x = self.loss_x_fn(step.x_logits, step.x)
        L_p = self.loss_p_fn(step.p_inf, step.p_gen, step.p_inf_x, use_p_inf)
        L_g = self.loss_g_fn(step.g_inf, step.g_gen)
        L_reg = self.loss_reg_fn(step.g_inf, step.p_inf)

        # Return as compositional LossOutput
        return StepLoss(x=L_x, p=L_p, g=L_g, reg=L_reg)
