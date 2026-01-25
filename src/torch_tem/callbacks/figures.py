"""FiguresCallback: Lightning callback for periodic figure generation.

Generates figures from model rollouts during training and logs them to
TensorBoard and/or saves them as PDF artifacts.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, List, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.diagnostics import DataTrace, RolloutTrace
from torch_tem.figures import register, sinks
from torch_tem.figures.registry import REGISTRY, FigureContext
from torch_tem.model import RolloutStream


class FigureCallbackSettings(BaseModel):
    """Settings for FiguresCallback.

    Controls when figures are generated, which figures to create, and where
    to save them.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="Whether to enable figure generation callback.",
    )
    figures: List[str] = Field(
        default_factory=lambda: ["overview", "walk.trajectories"],
        description="List of figure names to generate (from registry).",
    )
    every_n_steps: int = Field(
        default=10,
        description="Generate figures every N training steps.",
    )
    max_rollout_steps: int = Field(
        default=100,
        description="Maximum rollout steps to extract for figures.",
    )
    downsample_stride: int = Field(
        default=1,
        description="Downsample stride for trace extraction (1 = keep all).",
    )
    save_pdf: bool = Field(
        default=True,
        description="Save figures as PDF artifacts.",
    )
    log_tensorboard: bool = Field(
        default=True,
        description="Log figures to TensorBoard.",
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Base directory for PDF outputs (default: logger.log_dir). PDFs go under <base_dir>/figures/.",
    )
    env_idx: int = Field(
        default=0,
        description="Environment index to visualize.",
    )
    freq_idx: int = Field(
        default=0,
        description="Frequency module index to visualize.",
    )


class FiguresCallback(pl.Callback):
    """Lightning callback for periodic figure generation.

    Generates figures from model rollouts at regular intervals during training,
    saving PDFs and/or logging to TensorBoard.

    This callback:
    - Runs on rank 0 only (distributed training safe)
    - Extracts traces from model rollouts
    - Looks up figure specs from the registry
    - Generates figures and persists via sinks
    - Catches exceptions to avoid interrupting training
    """

    def __init__(self, settings: FigureCallbackSettings):
        """Initialize callback.

        Args:
            settings: Configuration for figure generation.

        Raises:
            ValueError: If any requested figure names are not registered.
        """
        super().__init__()
        self.settings = settings
        register.register_builtin_figures()  # Ensure built-in figures are registered
        REGISTRY.validate(settings.figures)  # Validate figure names at initialization

    def on_train_batch_start(self, trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int) -> None:
        """Generate figures periodically during training.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Training module (contains model and datamodule).
            batch: Current batch data.
            batch_idx: Batch index.
        """
        # Only run on rank 0 and when enabled
        if not self.settings.enabled or trainer.global_rank != 0:
            return
        # Check if we should generate figures this step
        if trainer.global_step % self.settings.every_n_steps != 0:
            return
        # Skip on first step (no meaningful data yet)
        if trainer.global_step == 0:
            return

        try:  # Catch all exceptions to avoid interrupting training
            self._generate_figures(trainer, pl_module, batch)
        except Exception as e:
            print(f"FiguresCallback: Error generating figures at step {trainer.global_step}: {e}")
            traceback.print_exc()

    def _generate_figures(self, trainer: Trainer, pl_module: LightningModule, batch: Any) -> None:
        """Generate and persist all configured figures.

        Builds ModelTrace (for model diagnostics), DataTrace (for data/walk figures),
        and RolloutTrace (for combined spatial figures), then dispatches each requested
        figure to the appropriate trace using isinstance-based type matching.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Training module.
            batch: Current batch data.
        """
        # The training dataloader yields (walk, visited).
        if type(batch) not in {tuple, list}:
            raise ValueError(f"FiguresCallback expected batch=(walk, visited); got type={type(batch).__name__}")
        if len(batch) != 2:
            raise ValueError(f"FiguresCallback expected batch of length 2; got length={len(batch)}")

        chunk, visited = batch
        if not isinstance(chunk, list) or len(chunk) == 0:
            raise ValueError("FiguresCallback requires a non-empty walk chunk")

        # Extract model and validate
        if not (model := getattr(pl_module, "tem", None)):
            raise AttributeError("LightningModule missing 'tem' attribute")

        # Validate datamodule
        if not (datamodule := trainer.datamodule) or not datamodule.dataset:
            raise ValueError("DataModule or dataset not available")

        # Limit chunk to max_rollout_steps and metadata
        chunk_limied = chunk[: self.settings.max_rollout_steps]
        meta = {"global_step": trainer.global_step, "split": "train"}

        # Create RolloutTrace via canonical constructor (single source of truth
        # for max-steps + downsampling across walk/location/model components)
        rollout_trace = RolloutTrace.from_rollout(
            worlds=datamodule.dataset.environments,
            rollout=RolloutStream(model, chunk_limied, initial=None),
            chunk=chunk_limied,
            max_steps=self.settings.max_rollout_steps,
            downsample_stride=self.settings.downsample_stride,
            meta=meta,
        )

        # Derive ModelTrace and DataTrace from the same aligned rollout
        model_trace = rollout_trace.model
        data_trace = DataTrace(
            worlds=rollout_trace.worlds,
            walks=rollout_trace.walks,
            visited=visited if visited is not None else getattr(datamodule.dataset, "visited", None),
            meta=meta,
        )
        data_trace.validate()

        # Generate and persist each figure
        context = self.figure_context(trainer, split_name="train")
        traces = [rollout_trace, model_trace, data_trace]

        for figure_name in self.settings.figures:
            spec = REGISTRY.get(figure_name)

            # Find first compatible trace
            if trace := next((t for t in traces if isinstance(t, spec.accepts)), None):
                self.generate_figure(trainer, trace, context, spec)
            else:
                print(f"Warning: Skipping {figure_name} (no compatible trace for {spec.accepts.__name__})")

    def figure_context(self, trainer: Trainer, split_name: Optional[str]) -> FigureContext:
        """Build FigureContext from settings and trainer state."""
        return FigureContext(
            env_idx=self.settings.env_idx,
            freq_idx=self.settings.freq_idx,
            global_step=trainer.global_step,
            split_name=split_name,
        )

    def generate_figure(self, trainer: Trainer, trace: Any, ctx: FigureContext, spec: Any) -> None:
        fig = spec.plot(trace, ctx)  # Generate figure

        # Save PDF if requested
        if self.settings.save_pdf:
            base_dir = self.settings.output_dir
            if base_dir is None and isinstance(trainer.logger, TensorBoardLogger):
                base_dir = Path(trainer.logger.log_dir)

            if base_dir is not None:
                base_dir = Path(base_dir)
                pdf_path = sinks.make_figure_path(base_dir, spec.default_filename, step=trainer.global_step)
                sinks.save_pdf(fig, pdf_path)

        # Log to TensorBoard if requested
        if self.settings.log_tensorboard and isinstance(trainer.logger, TensorBoardLogger):
            tag = f"figures/{spec.default_filename}"
            sinks.log_tensorboard_figure(trainer.logger, tag, fig, global_step=trainer.global_step)

        # Close figure to prevent memory leaks
        plt.close(fig)
