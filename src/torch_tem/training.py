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


class TEMDataModule(pl.LightningDataModule):
    """Lightning DataModule for TEM training."""

    def __init__(self, env_paths: list, params: dict[str, Any]):
        super().__init__()
        self.env_paths = [str(p) for p in env_paths]
        self.params = params

    def setup(self, stage: str = None):
        """Setup is called on every process."""
        pass

    def train_dataloader(self):
        """Return training dataloader (iterable dataset)."""
        return TEMDataset(self.env_paths, self.params)


class TEMDataset(IterableDataset):
    """Iterable dataset that generates TEM batches on-the-fly."""

    def __init__(self, env_paths: list[str], params: dict[str, Any]):
        super().__init__()
        self.env_paths = env_paths
        self.params = params

        # Initialize environments and walks
        self.environments, self.walks, self.visited = self._setup_environments()

    def _setup_environments(self):
        """Initialize training environments, walks, and visit tracking."""
        environments = [
            data.World(
                graph,
                randomise_observations=self.params["randomise_observations"],
                shiny=(self.params["shiny"] if np.random.rand() < self.params["shiny_rate"] else None),
            )
            for graph in np.random.choice(self.env_paths, self.params["batch_size"])
        ]

        visited = [[False for _ in range(env.n_locations)] for env in environments]

        walks = [
            env.generate_walks(
                self.params["n_rollout"] * np.random.randint(self.params["walk_it_min"], self.params["walk_it_max"]),
                1,
            )[0]
            for env in environments
        ]

        return environments, walks, visited

    def __iter__(self):
        """Generate batches indefinitely."""
        while True:
            yield self._generate_batch()

    def _generate_batch(self):
        """Generate a single batch (chunk) of data."""
        walk_length_center = int(self.params.get("walk_length_center", self.params["walk_it_max"]))
        low = max(1, int(walk_length_center - self.params["walk_it_window"] * 0.5))
        high = max(low + 1, int(walk_length_center + self.params["walk_it_window"] * 0.5))

        # Build batch chunk
        chunk: list[list[list[Any]]] = []
        for env_i, walk in enumerate(self.walks):
            if len(walk) < self.params["n_rollout"]:
                # Generate new environment and walk
                self.environments[env_i] = data.World(
                    self.env_paths[np.random.randint(len(self.env_paths))],
                    randomise_observations=self.params["randomise_observations"],
                    shiny=(self.params["shiny"] if np.random.rand() < self.params["shiny_rate"] else None),
                )
                self.visited[env_i] = [False for _ in range(self.environments[env_i].n_locations)]
                walk = self.environments[env_i].generate_walks(
                    self.params["n_rollout"] * np.random.randint(low, high),
                    1,
                )[0]
                self.walks[env_i] = walk

            for step in range(self.params["n_rollout"]):
                if len(chunk) < self.params["n_rollout"]:
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

    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self.save_hyperparameters(params)
        self.params = params

        # Create TEM model
        self.tem = model.Model(params)
        self.prev_iter = None

    def forward(self, chunk):
        """Forward pass through TEM."""
        return self.tem(chunk, self.prev_iter)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        """Single training step."""
        chunk, visited = batch

        # Update hyperparameters for this iteration
        i = self.global_step
        eta_new, hebbian_decay_new, p2g_scale_offset, lr, walk_length_center, loss_weights = model.parameter_iteration(i, self.hparams)
        self.tem.hyper["eta"] = eta_new
        self.tem.hyper["hebbian_decay"] = hebbian_decay_new
        self.tem.hyper["p2g_scale_offset"] = p2g_scale_offset

        # Propagate walk_length_center to params for dataset
        self.params["walk_length_center"] = float(walk_length_center)

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
        optimizer = Adam(self.tem.parameters(), lr=self.hparams["lr_max"])

        # Lightning will call optimizer_step where we can update lr dynamically
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Custom optimizer step to update learning rate dynamically."""
        # Update learning rate based on current iteration
        i = self.global_step
        _, _, _, lr, _, _ = model.parameter_iteration(i, self.hparams)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Call the default optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
