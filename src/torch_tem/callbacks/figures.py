"""FiguresCallback: Lightning callback for periodic figure generation.

Generates figures from model rollouts during training and logs them to
TensorBoard and/or saves them as PDF artifacts.
"""

from __future__ import annotations

import itertools
import traceback
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from pydantic import BaseModel, ConfigDict, Field

from torch_tem.diagnostics.trace_collectors import collect_rollout_trace_tree, concat_traces, downsample_trace
from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures import register, sinks
from torch_tem.figures.registry import REGISTRY, FigureContext


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
    split: Literal["validate", "test"] = Field(
        default="validate",
        description="Split to sample for figure generation.",
    )
    figures: List[str] = Field(
        default_factory=lambda: ["overview", "walk.trajectories"],
        description="Figure names to generate (from registry).",
    )
    aggregate_for_tags: List[str] = Field(
        default_factory=lambda: ["coverage"],
        description="Registry tags that should use aggregated sampling.",
    )
    aggregate_batches: Optional[int] = Field(
        default=None,
        description="Number of batches to aggregate (None = all batches).",
    )
    reset_each_val_epoch: bool = Field(
        default=False,
        description="Reset validation dataset before and after figure sampling.",
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

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate figures at validation time."""
        if not self.settings.enabled or trainer.global_rank != 0 or self.settings.split != "validate":
            return
        try:
            self._generate_figures(trainer, pl_module, split_name="validate")
        except Exception as e:
            print("FiguresCallback: Error generating figures at step " f"{trainer.global_step}: {e}")
            traceback.print_exc()

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Generate figures at test time."""
        if not self.settings.enabled or trainer.global_rank != 0 or self.settings.split != "test":
            return
        try:
            self._generate_figures(trainer, pl_module, split_name="test")
        except Exception as e:
            print("FiguresCallback: Error generating figures at step " f"{trainer.global_step}: {e}")
            traceback.print_exc()

    def _generate_figures(self, trainer: Trainer, pl_module: LightningModule, split_name: str) -> None:
        """Generate and persist all configured figures.

        Builds a TraceTree for episode sampling and (optionally) an aggregated
        TraceTree, then dispatches each requested figure.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: Training module.
        """
        # Extract model and validate
        if not (model := getattr(pl_module, "tem", None)):
            raise AttributeError("LightningModule missing 'tem' attribute")

        # Validate datamodule
        datamodule = trainer.datamodule
        if datamodule is None or datamodule.get_dataset(split_name) is None:
            raise ValueError("DataModule or dataset not available")

        reset_for_figures = self.settings.reset_each_val_epoch and split_name == "validate"
        if reset_for_figures:
            datamodule.reset_split(split_name)

        try:
            dataset = datamodule.get_dataset(split_name)
            if dataset is None:
                raise ValueError(f"Dataset for split '{split_name}' not available")

            batches = iter(dataset)
            episode_batch = next(batches)
            episode_trace = self._build_rollout_trace(trainer, datamodule, model, episode_batch, split_name)

            aggregate_names = self._aggregate_figure_names(self.settings.figures)
            aggregate_trace = self._build_aggregate_trace(trainer, datamodule, model, split_name, episode_batch, batches) if aggregate_names else episode_trace
        finally:
            if reset_for_figures:
                datamodule.reset_split(split_name)

        # Generate and persist each figure
        context = self.figure_context(trainer, split_name)

        aggregate_names = self._aggregate_figure_names(self.settings.figures)
        episode_names = [name for name in self.settings.figures if name not in aggregate_names]

        self._dispatch_figures(trainer, episode_names, context, episode_trace)
        self._dispatch_figures(trainer, aggregate_names, context, aggregate_trace)

    def _sample_batch(self, datamodule: Any, split_name: str) -> Any:
        """Sample a single batch from the requested split."""
        return datamodule.sample_batch(split=split_name)

    def _build_rollout_trace(self, trainer: Trainer, datamodule: Any, model: Any, batch: Any, split_name: str) -> TraceTree:
        """Create a TraceTree from a batch."""
        if type(batch) not in {tuple, list}:
            raise ValueError("FiguresCallback expected batch=(walk, visited); " f"got type={type(batch).__name__}")
        if len(batch) != 2:
            raise ValueError("FiguresCallback expected batch of length 2; " f"got length={len(batch)}")

        dataset = datamodule.get_dataset(split_name)
        if dataset is None:
            raise ValueError(f"Dataset for split '{split_name}' not available")

        trace = collect_rollout_trace_tree(
            batch=self._move_to_device(batch, self._infer_model_device(model)),
            environments=dataset.environments,
            model=model,
            stop=self.settings.max_rollout_steps,
            meta={"global_step": trainer.global_step, "split": split_name},
        )
        return downsample_trace(trace, self.settings.downsample_stride)

    def _infer_model_device(self, model: Any) -> torch.device:
        """Infer the device used by the model parameters.

        Args:
            model: Model instance used for rollout inference.

        Returns:
            Torch device for model parameters, defaulting to CPU when unknown.
        """
        if hasattr(model, "parameters"):
            params = list(model.parameters())
            if params:
                return params[0].device
        return torch.device("cpu")

    def _move_to_device(self, value: Any, device: torch.device) -> Any:
        """Move tensors within a nested structure to a target device.

        Args:
            value: Nested structure containing tensors.
            device: Target torch device.

        Returns:
            Structure with tensors moved to the requested device.
        """
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, tuple):
            return tuple(self._move_to_device(item, device) for item in value)
        if isinstance(value, list):
            return [self._move_to_device(item, device) for item in value]
        if isinstance(value, dict):
            return {key: self._move_to_device(val, device) for key, val in value.items()}
        return value

    def _build_aggregate_trace(self, trainer: Trainer, datamodule: Any, model: Any, split_name: str, episode_batch: Any, batch_iter: Iterable[Any]) -> TraceTree:
        """Create an aggregated trace by concatenating all batches."""
        traces = [self._build_rollout_trace(trainer, datamodule, model, episode_batch, split_name)]

        remaining = self.settings.aggregate_batches
        if remaining is not None:
            batch_iter = itertools.islice(batch_iter, max(remaining - 1, 0))

        for batch in batch_iter:
            traces.append(self._build_rollout_trace(trainer, datamodule, model, batch, split_name))

        return concat_traces(traces)

    def _aggregate_figure_names(self, figure_names: Iterable[str]) -> list[str]:
        """Resolve figure names that should use aggregated sampling."""
        aggregate_tags = set(self.settings.aggregate_for_tags)
        if not aggregate_tags:
            return []

        return [name for name in figure_names if REGISTRY.get(name).tags & aggregate_tags]

    def _dispatch_figures(self, trainer: Trainer, figure_names: Iterable[str], context: FigureContext, trace: TraceTree) -> None:
        for figure_name in figure_names:
            spec = REGISTRY.get(figure_name)
            if isinstance(trace, spec.accepts):
                self.generate_figure(trainer, trace, context, spec)
            else:
                print("Warning: Skipping " f"{figure_name} (no compatible trace)")

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
