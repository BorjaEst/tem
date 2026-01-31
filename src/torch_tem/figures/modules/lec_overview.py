from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
import torch
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.colorbars import colorbar
from torch_tem.figures.figures.templates import ParamRasterTemplate
from torch_tem.figures.plots.rasterplot import plot_rasterplot
from torch_tem.figures.registry import FigureContext
from torch_tem.modules.lec import LECModel


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a per-frequency LEC overview with observations and parameters."""
    return LECOverview(trace, ctx).plot()


class LECOverview(ParamRasterTemplate):
    """Encapsulate state and rendering logic for the LEC overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.n_freq = trace.n_freq("state/lec/cells")
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idxs = [self.trace.validate_freq_idx("state/lec/cells", f) for f in range(self.n_freq)]

        self.observations = self.trace.get("world_step/observation")[:, self.env_idx]
        self.cells = [self.trace.get(f"state/lec/cells/{f}")[:, self.env_idx, :] for f in range(self.n_freq)]

        lec: LECModel = self.ctx.extras.get("lec")
        self.alpha = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.filter.alpha]
        self.w_f = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.w_f]

    def params(self, ax: Axes) -> None:
        """Plot per-frequency LEC parameters if available."""
        n_freq = min(len(self.alpha), len(self.w_f))
        freq_ids = np.arange(n_freq)
        ax.plot(freq_ids, self.alpha[:n_freq], marker="o", label="sigmoid(alpha)")
        ax.plot(freq_ids, self.w_f[:n_freq], marker="s", label="sigmoid(w_f)")
        ax.set_title("LEC parameters by frequency")
        ax.set_xlabel("Frequency index")
        ax.set_ylabel("Value")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize="small")

    @colorbar(group="lec_activity", label="Activation")
    def raster(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        activation_names = [f"Cells f{f_idx}" for f_idx in self.freq_idxs]
        options = {"vmin": 0.0, "vmax": 1.0, "obs_height": 0.2, "activation_names": activation_names}
        plot_rasterplot(ax, self.observations, self.cells, **options)
        ax.set_title("LEC activations (after ponderation)")
