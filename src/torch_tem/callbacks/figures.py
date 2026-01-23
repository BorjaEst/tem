"""Lightning callback for periodic figure generation and logging.

Integrates the figures subsystem into PyTorch Lightning training:
    - Generates figures at configurable intervals (every N steps)
    - Saves canonical PDF artifacts to disk
    - Logs rasterized previews to TensorBoard
    - Handles rank-zero-only behavior for distributed training
    - Automatically closes figures to prevent memory leaks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.diagnostics import extract_rollout_trace
from torch_tem.figures import tem_overview
from torch_tem.figures.sinks import log_tensorboard_figure, make_figure_path, save_pdf
from torch_tem.model import Rollout, TEMModel


class FigureCallbackSettings(BaseModel):
    """Configuration for the FiguresCallback.

    Attributes:
        enabled: Whether to generate figures during training.
        every_n_steps: Generate figures every N global steps (None = disabled).
        on_validation_end: Whether to generate figures after validation epochs.
        max_trace_steps: Maximum rollout steps to extract for visualization.
        downsample_stride: Temporal downsampling stride (1 = keep all steps).
        save_pdf: Whether to save PDF files to disk.
        log_tensorboard: Whether to log preview images to TensorBoard.
        figures: List of figure names to generate (e.g., ["tem_overview"]).
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Enable figure generation during training.",
    )
    every_n_steps: Optional[int] = Field(
        default=100,
        description="Generate figures every N global steps (None = disabled).",
    )
    on_validation_end: bool = Field(
        default=False,
        description="Generate figures after each validation epoch.",
    )
    max_trace_steps: int = Field(
        default=100,
        description="Maximum rollout steps to extract for visualization.",
    )
    downsample_stride: int = Field(
        default=1,
        description="Temporal downsampling stride for rollout extraction.",
    )
    save_pdf: bool = Field(
        default=True,
        description="Save figures as PDF files.",
    )
    log_tensorboard: bool = Field(
        default=True,
        description="Log figure previews to TensorBoard.",
    )
    figures: list[str] = Field(
        default_factory=lambda: ["tem_overview"],
        description="List of figure modules to generate.",
    )


class FiguresCallback(Callback):
    """PyTorch Lightning callback for periodic figure generation.

    This callback integrates with the TEM training loop to periodically:
        1. Extract a small rollout trace from the current batch
        2. Generate diagnostic figures using the figures subsystem
        3. Save PDFs to the logger directory
        4. Log preview images to TensorBoard

    The callback respects rank-zero-only behavior in distributed training
    and ensures figures are closed after save/log to prevent memory leaks.

    Example:
        >>> settings = FigureCallbackSettings(every_n_steps=500)
        >>> callback = FiguresCallback(settings)
        >>> trainer = Trainer(callbacks=[callback], ...)
    """

    def __init__(self, settings: FigureCallbackSettings):
        """Initialize the callback.

        Args:
            settings: Configuration for figure generation.
        """
        super().__init__()
        self.settings = settings
        self._last_batch_cache: Optional[Any] = None

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: Any, batch: Any, batch_idx: int) -> None:
        """Cache the last training batch for figure generation.

        Called after every training batch. Stores the batch for later use
        in figure generation (triggered by on_train_batch_start).
        """
        # Cache batch for next step's figure generation
        self._last_batch_cache = batch

    def on_train_batch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, batch_idx: int) -> None:
        """Generate figures at configured intervals.

        Called before each training batch. Checks if the current global step
        matches the configured frequency and generates figures if so.
        """
        if not self.settings.enabled:
            return

        # Check if we should generate figures this step
        if self.settings.every_n_steps is None:
            return

        global_step = trainer.global_step
        if global_step % self.settings.every_n_steps != 0:
            return

        # Only run on rank zero (main process in distributed training)
        if trainer.is_global_zero:
            self._generate_figures(trainer, pl_module, batch, global_step)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Generate figures after validation epoch (if enabled).

        Called after each validation loop completes.
        """
        if not self.settings.enabled or not self.settings.on_validation_end:
            return

        # Use cached training batch (validation batches may have different structure)
        if self._last_batch_cache is not None and trainer.is_global_zero:
            self._generate_figures(trainer, pl_module, self._last_batch_cache, trainer.global_step)

    def _generate_figures(self, trainer: pl.Trainer, pl_module: pl.LightningModule, batch: Any, global_step: int) -> None:
        """Generate and save/log all configured figures.

        Args:
            trainer: Lightning Trainer instance.
            pl_module: TEMLightningModule instance.
            batch: Training batch (chunk, visited) for rollout extraction.
            global_step: Current global training step.
        """
        chunk, _visited = batch
        trace = self._try_extract_trace(pl_module.tem, chunk, global_step)
        if trace is None:
            return

        base_dir = self._get_figure_base_dir(trainer)
        self._generate_and_persist_all(trainer, trace, base_dir, global_step)

    def _try_extract_trace(self, tem_model: TEMModel, chunk: Any, global_step: int):
        """Create a fresh rollout and extract a plot-ready trace.

        Returns:
            TEMRolloutTrace or None if extraction fails.
        """
        rollout = Rollout(tem_model, chunk, initial=None)
        try:
            return extract_rollout_trace(
                rollout,
                max_steps=self.settings.max_trace_steps,
                downsample_stride=self.settings.downsample_stride,
            )
        except (ValueError, RuntimeError) as e:
            print(f"[FiguresCallback] Skipping figure generation at step {global_step}: {e}")
            return None

    def _get_figure_base_dir(self, trainer: pl.Trainer) -> Path:
        """Determine the directory where figure artifacts should be written."""
        if trainer.logger is not None and hasattr(trainer.logger, "log_dir"):
            return Path(trainer.logger.log_dir)
        return Path("./figures_output")

    def _generate_and_persist_all(
        self,
        trainer: pl.Trainer,
        trace: Any,
        base_dir: Path,
        global_step: int,
    ) -> None:
        """Generate each configured figure and persist it to sinks."""
        for figure_name in self.settings.figures:
            fig = self._try_make_figure(figure_name, trace)
            if fig is None:
                continue

            self._maybe_save_pdf(fig, base_dir, figure_name, global_step)
            self._maybe_log_tensorboard(trainer, fig, figure_name, global_step)

    def _try_make_figure(self, figure_name: str, trace: Any):
        """Safely build a figure; returns None on failure."""
        try:
            return self._make_figure(figure_name, trace)
        except Exception as e:
            print(f"[FiguresCallback] Failed to generate figure '{figure_name}': {e}")
            return None

    def _maybe_save_pdf(
        self,
        fig: Any,
        base_dir: Path,
        figure_name: str,
        global_step: int,
    ) -> None:
        """Save the figure to disk (PDF) if enabled."""
        if not self.settings.save_pdf:
            return

        pdf_path = make_figure_path(base_dir, figure_name, step=global_step, extension="pdf")
        try:
            save_pdf(fig, pdf_path)
        except OSError as e:
            print(f"[FiguresCallback] Failed to save PDF {pdf_path}: {e}")

    def _maybe_log_tensorboard(
        self,
        trainer: pl.Trainer,
        fig: Any,
        figure_name: str,
        global_step: int,
    ) -> None:
        """Log the figure preview to TensorBoard if enabled."""
        if not self.settings.log_tensorboard or trainer.logger is None:
            return

        try:
            log_tensorboard_figure(
                trainer.logger,
                tag=f"figures/{figure_name}",
                fig=fig,
                global_step=global_step,
                close=True,
            )
        except (AttributeError, RuntimeError) as e:
            print(f"[FiguresCallback] Failed to log figure to TensorBoard: {e}")

    def _make_figure(self, figure_name: str, trace):
        """Dispatch to the appropriate figure module.

        Args:
            figure_name: Name of the figure module (e.g., "tem_overview").
            trace: TEMRolloutTrace with plot-ready data.

        Returns:
            Matplotlib Figure object.

        Raises:
            ValueError: If figure_name is not recognized.
        """
        if figure_name == "tem_overview":
            return tem_overview.make_figure(trace)
        else:
            raise ValueError(f"Unknown figure name: {figure_name}")
