"""TEM Overview Figure Module.

Generates a comprehensive multi-panel diagnostic figure showing:
    - Sensory reconstruction accuracy over time
    - Location belief uncertainty (place cells, grid cells)
    - Temporal dynamics of inference vs generation

This is the canonical "first figure" demonstrating end-to-end TEM behavior.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from torch_tem.diagnostics import TEMRolloutTrace, compute_location_uncertainty, compute_sensory_accuracy
from torch_tem.figures.style import DEFAULT_STYLE, StyleConfig


def make_figure(
    trace: TEMRolloutTrace,
    *,
    env_idx: int = 0,
    freq_idx: int = 0,
    style: Optional[StyleConfig] = None,
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Generate TEM overview diagnostic figure.
    
    Creates a 2x2 multi-panel figure with:
        - Top-left: Sensory prediction accuracy over time
        - Top-right: Location belief uncertainty (entropy) over time
        - Bottom-left: Place cell belief dynamics (p_inf vs p_gen)
        - Bottom-right: Grid cell belief dynamics (g_inf vs g_gen)
    
    Args:
        trace: TEMRolloutTrace with model outputs and labels.
        env_idx: Environment index to visualize (batch dimension).
        freq_idx: Frequency module index to visualize.
        style: Optional StyleConfig. If None, uses DEFAULT_STYLE.
        figsize: Figure size in inches (width, height).
    
    Returns:
        Matplotlib Figure object. Does NOT call plt.show().
    
    Raises:
        ValueError: If trace is empty or env_idx/freq_idx out of bounds.
    
    Example:
        >>> trace = extract_rollout_trace(rollout, max_steps=100)
        >>> fig = make_figure(trace, env_idx=0)
        >>> save_pdf(fig, Path("outputs/tem_overview.pdf"))
    """
    if env_idx >= trace.batch_size:
        raise ValueError(f"env_idx {env_idx} out of bounds for batch_size {trace.batch_size}")
    if freq_idx >= trace.n_frequencies:
        raise ValueError(f"freq_idx {freq_idx} out of bounds for n_frequencies {trace.n_frequencies}")
    
    if style is None:
        style = DEFAULT_STYLE
    
    # Extract single environment for visualization
    trace_single = trace.select_env(env_idx)
    
    # Compute diagnostic signals
    accuracy = compute_sensory_accuracy(trace_single).squeeze(1)  # (T,)
    uncertainty = compute_location_uncertainty(trace_single, freq_idx=freq_idx).squeeze(1)  # (T,)
    
    # Create figure with 2x2 subplots
    with style.apply_context():
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f"TEM Overview (Env {env_idx}, Frequency {freq_idx})", fontsize=14, fontweight="bold")
        
        time_steps = np.arange(trace_single.n_steps)
        
        # Top-left: Sensory prediction accuracy
        ax = axes[0, 0]
        ax.plot(time_steps, accuracy, color="steelblue", linewidth=1.5)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Accuracy (0-1)")
        ax.set_title("Sensory Reconstruction Accuracy")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Top-right: Location uncertainty
        ax = axes[0, 1]
        ax.plot(time_steps, uncertainty, color="darkorange", linewidth=1.5)
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title("Location Belief Uncertainty (Place Cells)")
        ax.grid(True, alpha=0.3)
        
        # Bottom-left: Place cell dynamics (p_inf vs p_gen)
        ax = axes[1, 0]
        # Sample a few place cells for visualization
        n_place = trace_single.p_inf[freq_idx].shape[2]
        cell_sample = min(5, n_place)  # Show up to 5 cells
        cell_indices = np.linspace(0, n_place - 1, cell_sample, dtype=int)
        
        for i, cell_idx in enumerate(cell_indices):
            p_inf_cell = trace_single.p_inf[freq_idx][:, 0, cell_idx]
            ax.plot(time_steps, p_inf_cell, label=f"Cell {cell_idx}", linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Place Cell Activity")
        ax.set_title("Place Cell Dynamics (p_inf)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Bottom-right: Grid cell dynamics (g_inf vs g_gen)
        ax = axes[1, 1]
        # Sample a few grid cells for visualization
        n_grid = trace_single.g_inf[freq_idx].shape[2]
        grid_sample = min(5, n_grid)
        grid_indices = np.linspace(0, n_grid - 1, grid_sample, dtype=int)
        
        for i, grid_idx in enumerate(grid_indices):
            g_inf_cell = trace_single.g_inf[freq_idx][:, 0, grid_idx]
            ax.plot(time_steps, g_inf_cell, label=f"Grid {grid_idx}", linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Grid Cell Activity")
        ax.set_title("Grid Cell Dynamics (g_inf)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    return fig
