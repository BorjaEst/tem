from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import pub_ready_plots as prp
import scienceplots  # noqa: F401 (registers "science", "nature", ...)
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from torch_tem.figures.registry import FigureContext


class BaseFigureTemplate(ABC):
    """Base class for multi-panel figure templates.

    Subclasses implement `_create_layout()` and panel methods named in PANEL_NAMES.
    """

    PANEL_NAMES: Sequence[str] = ()
    BASE_FIGSIZE: float = 1.2  # Base figure size multiplier
    HEIGHT_FRAC: float = 0.2  # Fraction of textheight for figure height

    def __init__(self, trace: Any, ctx: FigureContext) -> None:
        self.trace = trace
        self.ctx = ctx
        self.fig: Optional[Figure] = None
        self.axdict: dict[str, Axes] = {}

    @abstractmethod
    def _create_layout(self, fig: Figure) -> Sequence[Axes]:
        """Create the figure layout and return axes in the order of PANEL_NAMES."""
        raise NotImplementedError

    def plot(self) -> Figure:
        """Create, render, and return the final Matplotlib figure."""
        colorbar_groups: dict[str, dict[str, Any]] = {}
        rc_params, prp_w, prp_h = prp.get_mpl_rcParams(self.ctx.layout, height_frac=self.HEIGHT_FRAC)
        rc_params = dict(rc_params)

        with ExitStack() as stack:
            # scienceplots aesthetics and prp + registry overrides
            stack.enter_context(plt.style.context(list(self.ctx.styles)))
            stack.enter_context(plt.rc_context(rc_params))

            # figure size from context or prp defaults
            self.fig = plt.figure(
                figsize=(prp_w * self.BASE_FIGSIZE, prp_h * self.BASE_FIGSIZE),
                dpi=self.ctx.dpi,
                constrained_layout=True,
            )
            axs = list(self._create_layout(self.fig))
            self.axdict = {name: ax for name, ax in zip(self.PANEL_NAMES, axs)}

            # render each panel
            for name, ax in self.axdict.items():
                self._render_panel(name, ax, colorbar_groups)

            self._apply_colorbars(colorbar_groups)
            return self.fig

    def _render_panel(self, name: str, ax: Axes, colorbar_groups: dict[str, dict[str, Any]]) -> None:
        panel_fn = getattr(self, name)
        panel_fn(ax)  # Rendering function is expected to modify ax in-place
        meta = getattr(panel_fn, "_tem_colorbar", None)
        if not meta:
            return

        group = meta["group"]
        label = meta.get("label", None)
        group_state = colorbar_groups.setdefault(group, {"axes": [], "mappable": None, "label": label})
        group_state["axes"].append(ax)

        # Keep first non-empty label (avoid overwriting)
        if group_state.get("label") is None and label is not None:
            group_state["label"] = label

        # Discover a mappable (host axis first, then inset axes, then fallback)
        mappable = self._find_panel_mappable(ax)

        if group_state["mappable"] is None and mappable is not None:
            group_state["mappable"] = mappable

    def _apply_colorbars(self, colorbar_groups: dict[str, dict[str, Any]]) -> None:
        """Attach grouped colorbars to the figure."""
        for group_state in colorbar_groups.values():
            mappable = group_state.get("mappable")
            axes = group_state.get("axes", [])
            if mappable is None or not axes:
                continue
            label = group_state.get("label", None)
            self.fig.colorbar(mappable, ax=axes, label=label)

    def _find_panel_mappable(self, ax: Axes) -> ScalarMappable | None:
        # A) Preferred: explicit convention used by your plot functions
        mappable = getattr(ax, "_tem_colorbar_mappable", None)
        if mappable is not None:
            return mappable

        # B) Inset axes created via ax.inset_axes(...) end up here
        for child in getattr(ax, "child_axes", []):
            mappable = getattr(child, "_tem_colorbar_mappable", None)
            if mappable is not None:
                return mappable

        # C) Optional fallback: common Matplotlib artists
        # matrix/matshow creates an AxesImage stored in ax.images
        if getattr(ax, "images", None):
            return ax.images[-1]
        if getattr(ax, "collections", None):
            return ax.collections[-1]

        return None
