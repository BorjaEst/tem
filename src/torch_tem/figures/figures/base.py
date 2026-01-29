from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.registry import FigureContext


@dataclass(frozen=True)
class LayoutSpec:
    """Lightweight panel placement descriptor (legacy template support)."""

    type: str
    position: tuple[int, int]
    rowspan: int = 1
    colspan: int = 1


class BaseFigureTemplate(ABC):
    LAYOUT: dict[str, LayoutSpec] = {}  # subclasses override
    COLORBAR_GROUPS: dict[str, dict[str, Any]] = {}  # subclasses override

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        self.trace = trace
        self.ctx = ctx

    def plot(self) -> Figure:
        """Public entry point."""
        style = getattr(self.ctx, "style", None) or "default"
        with plt.style.context(style):
            fig, axes = self._create_layout()
            self.fig = fig
            self.axes = axes

            self._apply_context_styles()
            self._fill_panels()
            self._apply_colorbars()
            self.post_process()

        return self.fig

    def _create_layout(self) -> tuple[Figure, dict[str, plt.Axes]]:
        """Create figure and axes based on the layout specification.

        Returns:
            A tuple of (figure, axes_by_name).
        """
        panel_specs = self._validate_panels(list(self.LAYOUT.items()))
        n_rows, n_cols = self._grid_shape(panel_specs)

        fig = plt.figure(figsize=self.ctx.figsize, dpi=self.ctx.dpi)
        grid = fig.add_gridspec(n_rows, n_cols)
        axes: dict[str, plt.Axes] = {}

        for name, spec in panel_specs:
            row, col = spec.position
            ax = fig.add_subplot(grid[row : row + spec.rowspan, col : col + spec.colspan])
            axes[name] = ax

        return fig, axes

    def _fill_panels(self) -> None:
        for name, _spec in self._validate_panels(list(self.LAYOUT.items())):
            fn = getattr(self, f"fill_{name}", None) or getattr(self, name, None)
            if fn is None:
                raise NotImplementedError(f"{self.__class__.__name__} missing fill_{name}()")
            fn(self.axes[name])

    def _apply_context_styles(self) -> None:
        color_cycle = getattr(self.ctx, "color_cycle", None)
        if color_cycle:
            plt.rcParams["axes.prop_cycle"] = plt.cycler(color=self.ctx.color_cycle)

        tick_fontsize = getattr(self.ctx, "tick_fontsize", None)
        for ax in self.axes.values():
            if tick_fontsize is not None:
                ax.tick_params(labelsize=tick_fontsize)

    def post_process(self) -> None:
        """Subclasses may override."""
        self.fig.tight_layout()

    def _validate_panels(self, panel_specs: list[tuple[str, LayoutSpec]]) -> list[tuple[str, LayoutSpec]]:
        """Validate layout specs for collisions and invalid spans."""
        seen = set()
        occupied: set[tuple[int, int]] = set()
        for name, spec in panel_specs:
            if name in seen:
                raise ValueError(f"Duplicate panel name: {name}")
            seen.add(name)
            row, col = spec.position
            if row < 0 or col < 0:
                raise ValueError(f"Invalid layout position for {name}: {spec.position}")
            if spec.rowspan <= 0 or spec.colspan <= 0:
                raise ValueError(f"Invalid layout span for {name}: {spec.rowspan}x{spec.colspan}")
            for row_idx in range(row, row + spec.rowspan):
                for col_idx in range(col, col + spec.colspan):
                    cell = (row_idx, col_idx)
                    if cell in occupied:
                        raise ValueError(f"Layout panels overlap at {cell} (panel {name})")
                    occupied.add(cell)
        return panel_specs

    def _grid_shape(self, panel_specs: list[tuple[str, LayoutSpec]]) -> tuple[int, int]:
        """Compute the grid size for the layout specification."""
        max_row = 0
        max_col = 0
        for _name, spec in panel_specs:
            row, col = spec.position
            max_row = max(max_row, row + spec.rowspan)
            max_col = max(max_col, col + spec.colspan)
        if max_row <= 0 or max_col <= 0:
            raise ValueError("Layout must contain at least one panel")
        return max_row, max_col

    def _apply_colorbars(self) -> None:
        """Apply shared colorbars for configured panel groups."""
        for group_name, group in self.COLORBAR_GROUPS.items():
            panels = group.get("panels", [])
            if not panels:
                warnings.warn(f"Colorbar group '{group_name}' has no panels", stacklevel=2)
                continue
            axes = [self.axes[name] for name in panels if name in self.axes]
            if not axes:
                warnings.warn(f"Colorbar group '{group_name}' references unknown panels", stacklevel=2)
                continue
            mappable = self._find_mappable(axes)
            if mappable is None:
                warnings.warn(f"Colorbar group '{group_name}' has no mappable artists", stacklevel=2)
                continue
            colorbar_kwargs = {key: value for key, value in group.items() if key != "panels"}
            self.fig.colorbar(mappable, ax=axes, **colorbar_kwargs)

    def _find_mappable(self, axes: list[plt.Axes]) -> Any:
        """Find the most recent mappable from a list of axes."""
        for ax in axes:
            mappable = getattr(ax, "_tem_colorbar_mappable", None)
            if mappable is not None:
                return mappable
            if ax.images:
                return ax.images[-1]
            if ax.collections:
                return ax.collections[-1]
        return None
