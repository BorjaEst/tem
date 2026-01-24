"""PyTorch Lightning training infrastructure for TEM.

Provides :class:`TrainingLoop`, a Lightning wrapper that implements the
complete TEM training loop with visit-masked loss accumulation and curriculum
scheduling.

Key Features:
    Streaming rollouts:
        Losses are accumulated incrementally during iteration (memory-efficient,
        no intermediate state storage).

    Visit masking:
        Only revisits contribute to optimization. First visits update the visited
        mask but are excluded from loss/accuracy computation.

    Stateful batches:
        Final state from each training batch is detached and reused as the
        initial state for the next batch (maintains RNN continuity).

    Curriculum scheduling:
        Hebbian parameters and walk length are scheduled dynamically during training.
        Learning rate scheduling is handled via a Lightning lr_scheduler.

Architecture Note
-----------------
Settings composition:
    - TrainerConfig composes low-level '*Settings' from settings.py
    - Prevents duplication of parameters like walk curriculum bounds
    - Instantiated in run.py from individual settings components

See Also:
    :class:`torch_tem.losses.TEMLoss`: Loss computation
    :class:`torch_tem.model.Model`: Core TEM model
    :class:`torch_tem.model.RolloutStream`: Streaming rollout iterator
"""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torch_tem import losses, metrics, settings
from torch_tem.losses import AccumLoss, LossG, LossOutput, LossP, LossReg, LossX, StepLoss
from torch_tem.metrics import AccuracyO
from torch_tem.model import Model, RolloutStream, TEMLabel, TEMOutput, TEMState


class TrainerConfig(BaseModel):
    """Composite configuration for TEM training (Lightning trainer + schedules).

    This Config class composes low-level '*Settings' from settings.py to provide
    complete configuration for TrainingLoop and Lightning Trainer. It aggregates
    Lightning infrastructure settings with runtime schedules and PyTorch/Lightning
    optimization components (optimizer + lr_scheduler).

    Architecture:
        - Composes settings.LossSettings, settings.OptimizerSettings, and settings.SchedulerSettings.
        - Used by TrainingLoop
        - Instantiated from RunArguments in run.py (prevents parameter duplication)

    Note:
        Walk curriculum settings (walk) are shared with DataConfig as a single
        runtime curriculum: the trainer schedules the current walk length, and the
        data pipeline consumes it when generating walks.
    """

    model_config = ConfigDict(extra="allow")  # Allow extra Lightning kwargs

    # Core Lightning Trainer kwargs
    max_steps: int = Field(default=20000, description="Maximum training steps.")
    log_every_n_steps: int = Field(default=10, description="Log metrics every N steps.")
    enable_progress_bar: bool = Field(default=True, description="Show progress bar during training.")

    # Loss settings (leaf settings)
    loss: settings.LossSettings = Field(
        default_factory=settings.LossSettings,
        description="Loss settings including weights for each component.",
    )

    # Walk curriculum bounds (referenced from DataConfig for annealing schedule)
    walk: settings.CurriculumSettings = Field(
        default_factory=settings.CurriculumSettings,
        description="Walk length curriculum settings (shared with DataConfig).",
    )

    # PyTorch/Lightning optimization components
    optimizer: settings.OptimizerSettings = Field(
        default_factory=settings.OptimizerSettings,
        description="Optimizer settings (e.g., Adam hyperparameters).",
    )

    # Schedulers (LR scheduler + runtime hyperparameter schedules)
    scheduler: settings.SchedulerSettings = Field(
        default_factory=settings.SchedulerSettings,
        description="Schedulers grouped by what they control (lr/memory/uncertainty).",
    )


class TrainingLoop(pl.LightningModule):
    """PyTorch Lightning module for TEM training.

    Integrates TEM model training with PyTorch Lightning, handling:
        - Visit-masked loss accumulation during streaming rollouts
        - Dynamic hyperparameter scheduling (learning rate, Hebbian parameters)
        - Stateful batch processing for recurrent continuity
        - Curriculum control (walk length annealing)
        - Metric logging and validation

    The module maintains an internal state (``prev_state``) that carries recurrent
    memory across training batches. This state is detached after each step to
    prevent gradient accumulation across batches while preserving memory content.

    Attributes:
        tem: The wrapped TEM model.
        loss_fn: Loss computation module.
        acc_o_fn: Sensory accuracy metric.
        prev_state: Previous batch's final state (detached).
        trainer_settings: Combined trainer and schedule configuration.
    """

    def __init__(self, model: Model, training: TrainerConfig):
        """Initialize the Lightning module.

        Args:
            model: TEM model to wrap.
            training: Trainer settings including schedules for runtime hyperparameters
                and learning rate, plus walk curriculum bounds for annealing.
        """
        super().__init__()
        self.trainer_settings = training

        params = {"trainer": training.model_dump()}
        self.save_hyperparameters(params)

        self.tem: Model = model
        self.prev_state: Optional[TEMState] = None

        self.loss_fn = losses.TEMLoss(training.loss)
        self.acc_o_fn = metrics.SensoryAccuracy(reduction="none")

    def forward(self, batch: Any, prev_state: Optional[TEMState] = None) -> tuple[LossOutput, AccuracyO, TEMState]:
        """Execute streaming rollout with visit-masked loss accumulation.

        Iterates through environment steps, computing and accumulating losses only
        for locations that have been previously visited (revisits). First visits
        update the visited mask but do not contribute to the loss.

        Args:
            batch: Tuple ``(chunk, visited)`` where:
                chunk: Iterable of ``(locations, observations, actions)`` tuples
                    for one rollout segment.
                visited: List of per-environment boolean masks ``[env_i][loc_id]``
                    tracking which locations have been visited. Updated in-place.
            prev_state: Initial state for the rollout. If None, a fresh state is
                initialized.

        Returns:
            Tuple of:
                loss_output: Accumulated losses over all revisits.
                accuracies: Mean sensory prediction accuracies.
                last_state: Final TEM state from the rollout.

        Raises:
            ValueError: If chunk is empty or rollout produces no states.
        """
        chunk, visited = batch
        if len(chunk) == 0:
            raise ValueError("forward requires a non-empty chunk")

        accum = AccumLoss.zero(device=self.device)
        acc_counts = AccuracyO.zero(device=self.device)

        for output, labels, state in RolloutStream(self.tem, chunk, prev_state):
            step_contrib, acc_increments = self.model_iteration(output, labels, state, visited)

            # Accumulate loss and accuracies
            if step_contrib is not None:
                accum = accum + step_contrib
            acc_counts = acc_counts + acc_increments

        final_acc = acc_counts

        return accum, final_acc, state  # last_state

    def model_iteration(self, output: TEMOutput, label: TEMLabel, state: TEMState, visited: list[list[bool]]) -> tuple[Optional[StepLoss], AccuracyO]:
        """Compute visit-masked loss and accuracy for a single timestep.

        Implements the revisit gating policy: losses and accuracies are only
        accumulated for environments visiting previously-seen locations. First
        visits are excluded from optimization but update the visited mask.

        Args:
            output: TEM predictions for the current timestep.
            label: Ground truth observations for the current timestep.
            state: Current TEM state containing predictions and ground truth.
            visited: Per-environment visited masks ``visited[env_i][loc_id]``.
                Updated in-place when environments visit new locations.

        Returns:
            Tuple of:
                step_loss: Mean loss over contributing environments, or None if
                    all environments are on first visits.
                accuracy_counts: :class:`AccuracyO` with weighted accuracies.
        """
        step_losses = self.loss_fn(output, label, state)
        step_acc = self.acc_o_fn(output.reconstruction.o_logits, label.observation)

        losses_per_env: list[StepLoss] = []
        acc_total = AccuracyO.zero(device=self.device)

        for env_i, env_visited in enumerate(visited):
            loc_id = label.locations[env_i]["id"]
            if not env_contributes_and_update(env_visited, loc_id):
                continue

            losses_per_env.append(env_step_loss(step_losses, env_i))
            acc_total = acc_total + env_acc_increments(step_acc, env_i)

        return mean_step_losses(losses_per_env), acc_total

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one training step.

        Args:
            batch: Training batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for optimization.
        """
        loss_output, accuracies, state = self(batch, self.prev_state)
        self.prev_state = state.detach()

        self._log_step_metrics(prefix="", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="", accuracies=accuracies)
        return loss_output.total

    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one validation step.

        Resets recurrent state (except memory) at batch boundaries.

        Args:
            batch: Validation batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for logging.
        """
        batch_size, device = batch[0][0][1].shape[0], batch[0][0][1].device  # Chunk, step, observations
        init_state = self.prev_state.new() if self.prev_state else self.tem.init_state(batch_size, device)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="val/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="val/", accuracies=accuracies)
        return loss_output.total

    def test_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Execute one test step.

        Resets recurrent state (except memory) at batch boundaries.

        Args:
            batch: Test batch from dataloader.
            batch_idx: Batch index (unused).

        Returns:
            Total loss for logging.
        """
        batch_size, device = batch[0][0][1].shape[0], batch[0][0][1].device  # Chunk, step, observations
        init_state = self.prev_state.new() if self.prev_state else self.tem.init_state(batch_size, device)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="test/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="test/", accuracies=accuracies)
        return loss_output.total

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update runtime hyperparameters before training step.

        Applies curriculum scheduling to Hebbian parameters and walk length based on
        global step count.

        Args:
            batch: Training batch (unused).
            batch_idx: Batch index (unused).
        """
        eta, hebbian_decay, p2g_uncertainty_offset, walk_center = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)
        self._maybe_set_walk_length_center(walk_center)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before validation step.

        Args:
            batch: Validation batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple validation sets.
        """
        eta, hebbian_decay, p2g_uncertainty_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before test step.

        Args:
            batch: Test batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple test sets.
        """
        eta, hebbian_decay, p2g_uncertainty_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime(eta, hebbian_decay, p2g_uncertainty_offset)

    def _maybe_set_walk_length_center(self, walk_length_center: float) -> None:
        """Update datamodule walk length curriculum if available.

        Attempts to call ``datamodule.set_walk_length_center()`` if the method
        exists. Safe to call even if datamodule lacks curriculum support.

        Args:
            walk_length_center: Target mean walk length for curriculum annealing.
        """
        datamodule = getattr(self.trainer, "datamodule", None)
        setter = getattr(datamodule, "set_walk_length_center", None) if datamodule is not None else None
        if callable(setter):
            setter(walk_length_center)

    def _log_step_metrics(self, *, prefix: str, loss_output: LossOutput) -> None:
        """Log hierarchical loss components to tensorboard.

        Args:
            prefix: Metric namespace prefix (e.g., "val/", "test/", "").
            loss_output: Complete loss output to log.
        """
        self.log(f"{prefix}loss", loss_output.total, prog_bar=True)

        # Sensory reconstruction losses
        self.log(f"{prefix}Losses/lx_p_inf", loss_output.x.infer)
        self.log(f"{prefix}Losses/lx_p_gen_gi", loss_output.x.retrieved)
        self.log(f"{prefix}Losses/lx_p_gen_gg", loss_output.x.ancestral)
        self.log(f"{prefix}Losses/lx", loss_output.x.total)

        # Abstract location consistency losses
        self.log(f"{prefix}Losses/lg", loss_output.g.total)

        # Grounded location consistency losses
        self.log(f"{prefix}Losses/lp_g", loss_output.p.abstract)
        self.log(f"{prefix}Losses/lp_x", loss_output.p.sensory)
        self.log(f"{prefix}Losses/lp", loss_output.p.total)

        # Regularization losses
        self.log(f"{prefix}Losses/reg_g", loss_output.reg.g_l2)
        self.log(f"{prefix}Losses/reg_p", loss_output.reg.p_l1)

    def _log_accuracy_metrics(self, *, prefix: str, accuracies: AccuracyO) -> None:
        """Log sensory prediction accuracies to tensorboard.

        Args:
            prefix: Metric namespace prefix (e.g., "val/", "test/", "").
            accuracies: Accuracy metrics for the three prediction pathways.
        """
        self.log(f"{prefix}Accuracies/o_p_inf", accuracies.o_p_inf)
        self.log(f"{prefix}Accuracies/o_gen_gi", accuracies.o_gen_gi)
        self.log(f"{prefix}Accuracies/o_gen_gg", accuracies.o_gen_gg)

    def configure_optimizers(self):
        """Configure optimizer and LR scheduler for training."""
        config = self.trainer_settings.optimizer
        optimizer = Adam(self.tem.parameters(), config.lr, config.betas, config.eps, config.weight_decay)
        scheduler = ExponentialLR(optimizer, self.trainer_settings.scheduler.lr.gamma)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def _compute_schedule(self, iteration: int) -> tuple[float, float, float, float]:
        """Compute all scheduled hyperparameters for current iteration.

        Args:
            iteration: Current global step.

        Returns:
            Tuple of (eta, hebbian_decay, p2g_uncertainty_offset, walk_length_center):
                eta: Hebbian learning rate.
                hebbian_decay: Hebbian memory decay factor.
                p2g_uncertainty_offset: Additive uncertainty offset for place-to-grid inference.
                walk_length_center: Target mean walk length for curriculum.
        """
        walk = self.trainer_settings.walk
        hebbian = self.trainer_settings.scheduler.memory
        p2g = self.trainer_settings.scheduler.uncertainty

        # Hebbian memory parameters
        eta = min((iteration + 1) / hebbian.eta_it, 1) * hebbian.eta
        lamb = min((iteration + 1) / hebbian.lambda_it, 1) * hebbian.hebbian_decay

        # p->g uncertainty offset schedule (eta-style: schedule outputs the final runtime value)
        p2g_scale = 1 / (1 + np.exp((iteration - p2g.p2g_sig_half_it) / p2g.p2g_sig_scale_it))
        p2g_uncertainty_offset = p2g.offset_min + (p2g.offset_max - p2g.offset_min) * p2g_scale

        # Walk length center (annealing from max to min over training)
        max_steps = max(int(self.trainer_settings.max_steps), 1)
        walk_length_center = walk.walk_it_max - walk.walk_it_window * 0.5 - min((iteration + 1) / max_steps, 1) * (walk.walk_it_max - walk.walk_it_min - walk.walk_it_window)

        return eta, lamb, p2g_uncertainty_offset, walk_length_center


def select_env(t: Tensor, env_i: int) -> Tensor:
    """Extract single environment from batch tensor.

    Handles both reduced (scalar) and unreduced (batched) tensors gracefully.

    Args:
        t: Input tensor. Either scalar (reduced) or ``(B, ...)`` (unreduced).
        env_i: Environment index in ``[0, B)``.

    Returns:
        Scalar for reduced input, or ``t[env_i]`` for batched input.
    """
    return t if t.ndim == 0 else t[env_i]


def env_contributes_and_update(visited_env: list[bool], loc_id: int) -> bool:
    """Check and update visit status for one environment.

    Implements the revisit gating policy: only revisited locations contribute
    to training. First visits update the mask in-place but are excluded.

    Args:
        visited_env: Boolean mask ``visited_env[location_id]`` for one environment.
            Modified in-place on first visit.
        loc_id: Current location identifier.

    Returns:
        True if location was previously visited (contribute to loss/accuracy).
        False if first visit (exclude from optimization, mask updated).
    """
    if visited_env[loc_id]:
        return True
    visited_env[loc_id] = True
    return False


def env_step_loss(step_losses: LossOutput, env_i: int) -> StepLoss:
    """Extract single-environment loss from batch loss output.

    Args:
        step_losses: Batch-level loss output (possibly unreduced).
        env_i: Environment index to extract.

    Returns:
        StepLoss with components for environment ``env_i``.
    """
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


def env_acc_increments(step_acc: AccuracyO, env_i: int) -> AccuracyO:
    """Extract accuracy for one environment.

    Args:
        step_acc: Batch-level accuracy (possibly unreduced).
        env_i: Environment index to extract.

    Returns:
        :class:`AccuracyO` with per-pathway correctness and internal weight of 1.
    """
    o_p_inf = select_env(step_acc.o_p_inf, env_i)
    o_gen_gi = select_env(step_acc.o_gen_gi, env_i)
    o_gen_gg = select_env(step_acc.o_gen_gg, env_i)
    _total = torch.ones((), device=o_p_inf.device, dtype=o_p_inf.dtype)
    return AccuracyO(o_p_inf=o_p_inf, o_gen_gi=o_gen_gi, o_gen_gg=o_gen_gg, _total=_total)


def mean_step_losses(losses_per_env: list[StepLoss]) -> Optional[StepLoss]:
    """Average losses across contributing environments.

    Args:
        losses_per_env: Losses for environments that passed revisit gating.

    Returns:
        Mean loss over contributing environments, or None if list is empty.
    """
    if not losses_per_env:
        return None
    total = losses_per_env[0]
    for loss in losses_per_env[1:]:
        total = total + loss
    return total / len(losses_per_env)
