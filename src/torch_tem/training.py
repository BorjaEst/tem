"""PyTorch Lightning training infrastructure for TEM.

Provides :class:`TEMLightningModule`, a Lightning wrapper that implements the
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
        Learning rate, Hebbian parameters, and walk length are scheduled
        dynamically during training.

Architecture Note
-----------------
Settings composition:
    - TrainerConfig composes low-level '*Settings' from settings.py
    - Prevents duplication of parameters like walk curriculum bounds
    - Instantiated in run.py from individual settings components

See Also:
    :class:`torch_tem.losses.TEMLoss`: Loss computation
    :class:`torch_tem.core.model.TEMModel`: Core TEM model
    :class:`torch_tem.core.model.Rollout`: Streaming rollout iterator
"""

from __future__ import annotations

from typing import Any, Optional

import lightning.pytorch as pl
import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.optim import Adam

from torch_tem import losses, metrics, settings
from torch_tem.core.model import Rollout, TEMModel, TEMState
from torch_tem.losses import AccumLoss, LossG, LossOutput, LossP, LossReg, LossX, StepLoss
from torch_tem.metrics import AccuracyCounts, AccuracyX


class TrainerConfig(BaseModel):
    """Composite configuration for TEM training (Lightning trainer + schedules).

    This Config class composes low-level '*Settings' from settings.py to provide
    complete configuration for TEMLightningModule and Lightning Trainer. It aggregates
    Lightning infrastructure settings with schedule configuration for loss weights,
    learning rate, Hebbian plasticity, and p2g variance offset.

    Architecture:
        - Composes settings.LossSettings, settings.LRScheduleSettings, etc.
        - Used by TEMLightningModule
        - Instantiated from RunArguments in run.py (prevents parameter duplication)

    Note:
        Walk curriculum settings (walk) are shared with DataConfig to coordinate
        walk length annealing between data generation and training loop.
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
    walk: settings.WalkCurriculumSettings = Field(
        default_factory=settings.WalkCurriculumSettings,
        description="Walk length curriculum settings (shared with DataConfig).",
    )

    # Training schedules (leaf settings)
    lr: settings.LRScheduleSettings = Field(
        default_factory=settings.LRScheduleSettings,
        description="Learning rate schedule settings.",
    )
    hebbian: settings.HebbianScheduleSettings = Field(
        default_factory=settings.HebbianScheduleSettings,
        description="Hebbian memory plasticity schedule settings.",
    )
    p2g_offset: settings.P2GOffsetScheduleSettings = Field(
        default_factory=settings.P2GOffsetScheduleSettings,
        description="Place-to-grid variance offset schedule settings.",
    )


class TEMLightningModule(pl.LightningModule):
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
        acc_x_fn: Sensory accuracy metric.
        prev_state: Previous batch's final state (detached).
        trainer_settings: Combined trainer and schedule configuration.
    """

    def __init__(self, model: TEMModel, training: TrainerConfig):
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

        self.tem: TEMModel = model
        self.prev_state: Optional[TEMState] = None

        self.loss_fn = losses.TEMLoss(training.loss)
        self.acc_x_fn = metrics.SensoryAccuracy(reduction="none")

    def forward(self, batch: Any, prev_state: Optional[TEMState] = None) -> tuple[LossOutput, AccuracyX, TEMState]:
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
        acc_counts = AccuracyCounts.zero(device=self.device)

        last_state: Optional[TEMState] = None

        for step in Rollout(self.tem, chunk, prev_state):
            last_state = step
            step_contrib, acc_increments = self.model_iteration(step, visited)

            # Accumulate loss and accuracies
            if step_contrib is not None:
                accum = accum + step_contrib
            acc_counts = acc_counts + acc_increments

        if last_state is None:
            raise ValueError("Rollout produced no states; check chunk formatting")

        final_acc = acc_counts.to_accuracy()

        return accum, final_acc, last_state

    def init_state(self, batch: Any, memory: Optional[list[Tensor]] = None) -> TEMState:
        """Initialize clean state for validation/test rollouts.

        Creates a fresh initial state while optionally preserving Hebbian memory.
        Used to reset recurrent state at evaluation boundaries while maintaining
        learned associations.

        Args:
            batch: Tuple ``(chunk, visited)`` where chunk[0] provides initial
                location and observation.
            memory: Optional Hebbian memory matrices to preserve across episodes.
                If None, memory is freshly initialized.

        Returns:
            Fresh :class:`~torch_tem.core.model.TEMState` with given memory.

        Raises:
            ValueError: If chunk is empty.
        """
        chunk, _visited = batch
        if len(chunk) == 0:
            raise ValueError("init_state requires a non-empty chunk")

        locations_0, x_0, _a_0 = chunk[0]
        batch_size = int(x_0.shape[0])

        return self.tem.init_iteration(locations_0, x_0, [None for _ in range(batch_size)], memory)

    def model_iteration(self, step: TEMState, visited: list[list[bool]]) -> tuple[Optional[StepLoss], AccuracyCounts]:
        """Compute visit-masked loss and accuracy for a single timestep.

        Implements the revisit gating policy: losses and accuracies are only
        accumulated for environments visiting previously-seen locations. First
        visits are excluded from optimization but update the visited mask.

        Args:
            step: Current TEM state containing predictions and ground truth.
            visited: Per-environment visited masks ``visited[env_i][loc_id]``.
                Updated in-place when environments visit new locations.

        Returns:
            Tuple of:
                step_loss: Mean loss over contributing environments, or None if
                    all environments are on first visits.
                accuracy_counts: :class:`AccuracyCounts` with summed correct
                    predictions and total count.
        """
        use_p_inf = self.tem.hyper["use_p_inf"]
        step_losses = self.loss_fn(step, use_p_inf)
        step_acc = self.acc_x_fn(step.x_logits, step.x)

        losses_per_env: list[StepLoss] = []
        acc_total = AccuracyCounts.zero(device=self.device)

        for env_i, env_visited in enumerate(visited):
            loc_id = step.g[env_i]["id"]
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
        memory = self.prev_state.M if self.prev_state is not None else None
        init_state = self.init_state(batch, memory)
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
        memory = self.prev_state.M if self.prev_state is not None else None
        init_state = self.init_state(batch, memory)
        loss_output, accuracies, _ = self(batch, init_state)

        self._log_step_metrics(prefix="test/", loss_output=loss_output)
        self._log_accuracy_metrics(prefix="test/", accuracies=accuracies)
        return loss_output.total

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        """Update runtime hyperparameters before training step.

        Applies curriculum scheduling to learning rate, Hebbian parameters,
        and walk length based on global step count.

        Args:
            batch: Training batch (unused).
            batch_idx: Batch index (unused).
        """
        eta, hebbian_decay, p2g_scale_offset, walk_center = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)
        self._maybe_set_walk_length_center(walk_center)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before validation step.

        Args:
            batch: Validation batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple validation sets.
        """
        eta, hebbian_decay, p2g_scale_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Update runtime hyperparameters before test step.

        Args:
            batch: Test batch (unused).
            batch_idx: Batch index (unused).
            dataloader_idx: Dataloader index for multiple test sets.
        """
        eta, hebbian_decay, p2g_scale_offset, _ = self._compute_schedule(self.global_step)
        self.tem.set_runtime_hyperparams(eta, hebbian_decay, p2g_scale_offset)

    def on_before_optimizer_step(self, optimizer) -> None:
        """Apply learning rate schedule before optimizer step.

        Args:
            optimizer: The optimizer being used (Adam).
        """
        lr = self._compute_lr(self.global_step)
        for group in optimizer.param_groups:
            group["lr"] = lr

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
        self.log(f"{prefix}Losses/Total", loss_output.total.detach())
        # Individual components follow legacy naming for comparability.
        self.log(f"{prefix}Losses/p_g", loss_output.p.abstract)
        self.log(f"{prefix}Losses/p_x", loss_output.p.sensory)
        self.log(f"{prefix}Losses/x_gen", loss_output.x.ancestral)
        self.log(f"{prefix}Losses/x_g", loss_output.x.retrieved)
        self.log(f"{prefix}Losses/x_p", loss_output.x.infer)
        self.log(f"{prefix}Losses/g", loss_output.g.transition)
        self.log(f"{prefix}Losses/reg_g", loss_output.reg.g_l2)
        self.log(f"{prefix}Losses/reg_p", loss_output.reg.p_l1)
        self.log(f"{prefix}Losses/lx", loss_output.x.total)
        self.log(f"{prefix}Losses/lp", loss_output.p.total)
        self.log(f"{prefix}Losses/lg", loss_output.g.total)

    def _log_accuracy_metrics(self, *, prefix: str, accuracies: AccuracyX) -> None:
        """Log sensory prediction accuracies to tensorboard.

        Args:
            prefix: Metric namespace prefix (e.g., "val/", "test/", "").
            accuracies: Accuracy metrics for the three prediction pathways.
        """
        self.log(f"{prefix}Accuracies/p", accuracies.p)
        self.log(f"{prefix}Accuracies/g", accuracies.g)
        self.log(f"{prefix}Accuracies/gt", accuracies.gt)

    def configure_optimizers(self):
        """Configure optimizer for training.

        Returns:
            Adam optimizer with initial learning rate from schedule settings.

        Note:
            Learning rate is updated dynamically in :meth:`on_before_optimizer_step`
            according to the exponential decay schedule.
        """
        return Adam(self.tem.parameters(), lr=self.trainer_settings.lr.lr_max)

    def _compute_lr(self, iteration: int) -> float:
        """Compute learning rate using exponential decay schedule.

        Args:
            iteration: Current global step.

        Returns:
            Learning rate (clamped to lr_min).
        """
        lr = self.trainer_settings.lr
        return max(
            lr.lr_min + (lr.lr_max - lr.lr_min) * (lr.lr_decay_rate ** (iteration / lr.lr_decay_steps)),
            lr.lr_min,
        )

    def _compute_schedule(self, iteration: int) -> tuple[float, float, float, float]:
        """Compute all scheduled hyperparameters for current iteration.

        Args:
            iteration: Current global step.

        Returns:
            Tuple of (eta, hebbian_decay, p2g_scale_offset, walk_length_center):
                eta: Hebbian learning rate.
                hebbian_decay: Hebbian memory decay factor.
                p2g_scale_offset: Place-to-grid transition variance offset.
                walk_length_center: Target mean walk length for curriculum.
        """
        walk = self.trainer_settings.walk
        hebbian = self.trainer_settings.hebbian
        p2g = self.trainer_settings.p2g_offset

        # Hebbian memory parameters
        eta = min((iteration + 1) / hebbian.eta_it, 1) * hebbian.eta
        lamb = min((iteration + 1) / hebbian.lambda_it, 1) * hebbian.hebbian_decay

        # p->g variance offset schedule
        p2g_scale_offset = 1 / (1 + np.exp((iteration - p2g.p2g_sig_half_it) / p2g.p2g_sig_scale_it))

        # Walk length center (annealing from max to min over training)
        max_steps = max(int(self.trainer_settings.max_steps), 1)
        walk_length_center = walk.walk_it_max - walk.walk_it_window * 0.5 - min((iteration + 1) / max_steps, 1) * (walk.walk_it_max - walk.walk_it_min - walk.walk_it_window)

        return eta, lamb, p2g_scale_offset, walk_length_center


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


def env_acc_increments(step_acc: AccuracyX, env_i: int) -> AccuracyCounts:
    """Extract accuracy counts for one environment.

    Args:
        step_acc: Batch-level accuracy (possibly unreduced).
        env_i: Environment index to extract.

    Returns:
        :class:`AccuracyCounts` with per-pathway correctness and count of 1.
    """
    p = select_env(step_acc.p, env_i)
    g = select_env(step_acc.g, env_i)
    gt = select_env(step_acc.gt, env_i)
    total = torch.ones((), device=p.device, dtype=p.dtype)
    return AccuracyCounts(p=p, g=g, gt=gt, total=total)


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
