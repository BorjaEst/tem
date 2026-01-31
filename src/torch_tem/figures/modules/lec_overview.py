from __future__ import annotations

import matplotlib.figure as mpl_figure
import numpy as np
import torch
from matplotlib.axes import Axes

from torch_tem.diagnostics.traces import TraceTree
from torch_tem.figures.figures.colorbars import colorbar
from torch_tem.figures.figures.templates import LECOverviewTemplate
from torch_tem.figures.plots.rasterplot import plot_rasterplot
from torch_tem.figures.plots.trajectory import plot_time_colored_trajectory
from torch_tem.figures.registry import FigureContext
from torch_tem.modules.lec import LECModel


def plot(trace: TraceTree, ctx: FigureContext) -> mpl_figure.Figure:
    """Plot a per-frequency LEC overview with observations and parameters."""
    return LECOverview(trace, ctx).plot()


class LECOverview(LECOverviewTemplate):
    """Encapsulate state and rendering logic for the LEC overview."""

    def __init__(self, trace: TraceTree, ctx: FigureContext) -> None:
        super().__init__(trace, ctx)
        self.env_idx = self.trace.validate_env_idx(self.ctx.env_idx)
        self.freq_idx = self.trace.validate_freq_idx("state/lec/cells", self.ctx.freq_idx)

        self.world = self.trace.get_world(self.env_idx)
        self.location_ids = self.trace.get("world_step/location_ids")[:, self.env_idx]
        self.observations = self.trace.get("world_step/observation")[:, self.env_idx]
        self.feature_series = self.trace.get("output/features")[:, self.env_idx, :]
        self.filtered_series = self.trace.get(f"state/lec/filtered/{self.freq_idx}")[:, self.env_idx, :]
        self.cell_series = self.trace.get(f"state/lec/cells/{self.freq_idx}")[:, self.env_idx, :]

        lec: LECModel = self.ctx.extras.get("lec")
        self.alpha = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.filter.alpha]
        self.w_f = [torch.sigmoid(p).detach().cpu().numpy() for p in lec.w_f]

        self.features_vmax = self._resolve_vmax(self.feature_series, min_value=1.0)
        self.lec_vmax = self._resolve_vmax(self.filtered_series, self.cell_series, min_value=1.0)

    @staticmethod
    def _resolve_vmax(*arrays: np.ndarray, min_value: float) -> float:
        vmax = 0.0
        for arr in arrays:
            if arr is None:
                continue
            if np.isfinite(arr).any():
                vmax = max(vmax, float(np.nanmax(arr)))
        return max(vmax, min_value)

    def trajectory(self, ax: Axes) -> None:
        """Plot the trajectory colored by time."""
        plot_time_colored_trajectory(ax, self.world, self.location_ids.tolist())
        ax.set_title("Trajectory colored by time")

    @colorbar(group="features", label="Feature value")
    def features(self, ax: Axes) -> None:
        """Plot the autoencoder features over time."""
        ax.imshow(self.feature_series.T, aspect="auto", cmap="GnBu", vmin=0.0, vmax=self.features_vmax)
        ax.set_title("LEC input features")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Feature")

    @colorbar(group="lec_activity", label="Activation")
    def filtered(self, ax: Axes) -> None:
        """Plot filtered features for the selected frequency."""
        ax.imshow(self.filtered_series.T, aspect="auto", cmap="GnBu", vmin=0.0, vmax=self.lec_vmax)
        ax.set_title(f"Filtered features f{self.freq_idx}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Feature")

    @colorbar(group="lec_activity", label="Activation")
    def cells(self, ax: Axes) -> None:
        """Plot LEC cell activations for the selected frequency."""
        ax.imshow(self.cell_series.T, aspect="auto", cmap="GnBu", vmin=0.0, vmax=self.lec_vmax)
        ax.set_title(f"LEC cells f{self.freq_idx}")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Cell")

    def raster(self, ax: Axes) -> None:
        """Plot observations and LEC activations over time."""
        options = {
            "vmin": 0.0,
            "vmax": self.lec_vmax,
            "activation_names": [f"LEC cells f{self.freq_idx}"],
        }
        plot_rasterplot(ax, observations=self.observations, activations=[self.cell_series], **options)
        ax.set_title("Observations and LEC activations")

    def params(self, ax: Axes) -> None:
        """Plot per-frequency LEC parameters if available."""
        if not self.alpha or not self.w_f:
            ax.text(0.5, 0.5, "LEC params unavailable", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title("LEC parameters")
            return

        n_freq = min(len(self.alpha), len(self.w_f))
        freq_ids = np.arange(n_freq)
        ax.plot(freq_ids, self.alpha[:n_freq], marker="o", label="sigmoid(alpha)")
        ax.plot(freq_ids, self.w_f[:n_freq], marker="s", label="sigmoid(w_f)")
        ax.axvline(self.freq_idx, linestyle="--", color="gray", linewidth=1.0)
        ax.set_title("LEC parameters by frequency")
        ax.set_xlabel("Frequency index")
        ax.set_ylabel("Value")
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="best", fontsize="small")
