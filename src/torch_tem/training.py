"""Training loop and utilities for TEM."""

from __future__ import annotations

import time
from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from pydantic import BaseModel, Field
from torch.optim import Adam
from torch.utils.data import IterableDataset
from torch.utils.tensorboard import SummaryWriter

from torch_tem import data, utils
from torch_tem.core import model
from torch_tem.core.model import Parameters
from torch_tem.settings import DataSettings, ScheduleSettings, TrainerSettings


class TEMDataModule(pl.LightningDataModule):
    """Lightning DataModule for TEM training."""

    def __init__(self, data_settings: DataSettings, schedule_settings: ScheduleSettings):
        super().__init__()
        self.data_settings = data_settings
        self.schedule_settings = schedule_settings
        self.dataset: Optional[TEMDataset] = None

    def setup(self, stage: str = None):
        """Setup is called on every process."""
        pass

    def train_dataloader(self):
        """Return training dataloader (iterable dataset)."""
        self.dataset = TEMDataset(
            self.data_settings,
            walk_it_min=self.schedule_settings.walk_it_min,
            walk_it_max=self.schedule_settings.walk_it_max,
            walk_it_window=self.schedule_settings.walk_it_window,
        )
        return self.dataset

    def set_walk_length_center(self, value: float):
        """Control surface: set walk length center (called by trainer during training)."""
        if self.dataset is not None:
            self.dataset.walk_length_center = value


class TEMDataset(IterableDataset):
    """Iterable dataset that generates TEM batches on-the-fly."""

    def __init__(self, data_settings: DataSettings, walk_it_min: int, walk_it_max: int, walk_it_window: float):
        super().__init__()
        self.data_settings = data_settings
        self.env_paths = [str(p) for p in data_settings.envs]

        # Walk curriculum bounds (owned by schedule, injected here)
        self.walk_it_min = walk_it_min
        self.walk_it_max = walk_it_max
        self.walk_it_window = walk_it_window
        self._walk_length_center = walk_it_max  # Default to max

        # Initialize environments and walks
        self.environments, self.walks, self.visited = self._setup_environments()

    def _setup_environments(self):
        """Initialize training environments, walks, and visit tracking."""
        environments = [
            data.World(
                graph,
                randomise_observations=self.data_settings.randomise_observations,
                shiny=(self.data_settings.shiny if np.random.rand() < self.data_settings.shiny_rate else None),
            )
            for graph in np.random.choice(self.env_paths, self.data_settings.batch_size)
        ]

        visited = [[False for _ in range(env.n_locations)] for env in environments]

        walks = [
            env.generate_walks(
                self.data_settings.n_rollout * np.random.randint(self.walk_it_min, self.walk_it_max),
                1,
            )[0]
            for env in environments
        ]

        return environments, walks, visited

    @property
    def walk_length_center(self) -> float:
        """Current center of walk length sampling window (updated by trainer)."""
        return self._walk_length_center

    @walk_length_center.setter
    def walk_length_center(self, value: float):
        """Update walk length center (called by LightningModule during training)."""
        self._walk_length_center = value

    def __iter__(self):
        """Generate batches indefinitely."""
        while True:
            yield self._generate_batch()

    def _generate_batch(self):
        """Generate a single batch (chunk) of data."""
        walk_length_center = int(self._walk_length_center)
        low = max(1, int(walk_length_center - self.walk_it_window * 0.5))
        high = max(low + 1, int(walk_length_center + self.walk_it_window * 0.5))

        # Build batch chunk
        chunk: list[list[list[Any]]] = []
        for env_i, walk in enumerate(self.walks):
            if len(walk) < self.data_settings.n_rollout:
                # Generate new environment and walk
                self.environments[env_i] = data.World(
                    self.env_paths[np.random.randint(len(self.env_paths))],
                    randomise_observations=self.data_settings.randomise_observations,
                    shiny=(self.data_settings.shiny if np.random.rand() < self.data_settings.shiny_rate else None),
                )
                self.visited[env_i] = [False for _ in range(self.environments[env_i].n_locations)]
                walk = self.environments[env_i].generate_walks(
                    self.data_settings.n_rollout * np.random.randint(low, high),
                    1,
                )[0]
                self.walks[env_i] = walk

            for step in range(self.data_settings.n_rollout):
                if len(chunk) < self.data_settings.n_rollout:
                    chunk.append([[comp] for comp in walk.pop(0)])
                else:
                    for comp_i, comp in enumerate(walk.pop(0)):
                        chunk[step][comp_i].append(comp)

        # Stack observations
        for i_step, step in enumerate(chunk):
            chunk[i_step][1] = torch.stack(step[1], dim=0)

        return chunk, self.visited


class TEMLightningModule(pl.LightningModule):
    """Lightning wrapper for TEM model."""

    def __init__(
        self,
        tem_model: model.Model,
        schedule_settings: ScheduleSettings,
        trainer_settings: TrainerSettings,
    ):
        super().__init__()
        # Store settings for schedule computation
        self.schedule_settings = schedule_settings
        self.trainer_settings = trainer_settings

        # Save hyperparameters (namespaced for clarity)
        self.save_hyperparameters(
            {
                "schedule": schedule_settings.model_dump(),
                "trainer": trainer_settings.model_dump(),
            }
        )

        # Store TEM model
        self.tem = tem_model
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

        # Update model hyperparameters
        self.tem.hyper["eta"] = eta_new
        self.tem.hyper["hebbian_decay"] = hebbian_decay_new
        self.tem.hyper["p2g_scale_offset"] = p2g_scale_offset

        # Update dataset's walk_length_center (via datamodule control surface)
        if hasattr(self.trainer, "datamodule") and hasattr(self.trainer.datamodule, "set_walk_length_center"):
            self.trainer.datamodule.set_walk_length_center(walk_length_center)

        # Move loss_weights to device
        loss_weights = loss_weights.to(self.device)

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

        # Update prev_iter for next step
        self.prev_iter = [forward[-1].detach()]

        # Compute accuracies
        acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
        acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

        # Log metrics
        self.log("loss", loss, prog_bar=True)
        self.log("Losses/Total", loss.detach())
        for idx, name in enumerate(["p_g", "p_x", "x_gen", "x_g", "x_p", "g", "reg_g", "reg_p"]):
            self.log(f"Losses/{name}", plot_loss[idx])
        self.log("Accuracies/p", acc_p)
        self.log("Accuracies/g", acc_g)
        self.log("Accuracies/gt", acc_gt)

        return loss

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
