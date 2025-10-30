"""
Visualization Functions for Sensory Processing

This module provides comprehensive visualization tools for analyzing and
understanding the SensoryProcessor's multi-scale temporal filtering behavior.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from torch_tem.modules.sensory import SensoryProcessor, SensoryState


def plot_sensory_analysis(
    model: torch.nn.Module,
    dataset,
    history: Dict[str, List],
    output_dir: Path,
    dpi: int = 200,
) -> Path:
    """Generate comprehensive visualization showing multi-scale temporal filtering.

    Creates a single publication-quality figure with 4 panels demonstrating:
    - A. Training convergence over epochs
    - B. Multi-scale filtered representations vs input observations
    - C. Evolution of learned filter parameters during training
    - D. Frequency response characteristics of the learned filters

    Args:
        model: Trained model containing a SensoryProcessor (e.g., LightningModule)
        dataset: Dataset containing sequences for visualization
        history: Training history dict with 'loss' and 'alpha_values' keys
        output_dir: Directory to save the figure
        dpi: Resolution for saved figure (default: 200)

    Returns:
        Path to the saved figure

    Example:
        >>> from torch_tem.figures.sensory import plot_sensory_analysis
        >>> output_path = plot_sensory_analysis(
        ...     model=trained_model,
        ...     dataset=test_dataset,
        ...     history=training_history,
        ...     output_dir=Path("outputs/figures")
        ... )
        >>> print(f"Saved to {output_path}")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Extract processor (handle both raw SensoryProcessor and wrapped models)
    if hasattr(model, "processor"):
        processor = model.processor
    elif isinstance(model, SensoryProcessor):
        processor = model
    else:
        raise ValueError("Model must be a SensoryProcessor or have a 'processor' attribute")

    n_freq = processor.n_freq

    # Get an example sequence
    sequence = dataset[0]

    with torch.no_grad():
        # Handle different model types
        if hasattr(model, "processor"):
            # Wrapped model (e.g., LightningModule)
            states, _ = model(sequence)
        else:
            # Raw SensoryProcessor
            seq_len = sequence.shape[0]
            n_compressed = processor.two_hot_table.shape[1]
            x_prev = [torch.zeros(1, n_compressed) for _ in range(n_freq)]
            states = []
            for t in range(seq_len):
                x_raw = sequence[t : t + 1]
                state = processor(x_raw, x_prev)
                states.append(state)
                x_prev = state.filtered

    # Extract filtered values over time (use L2 norm across dimensions for clarity)
    seq_len = len(states)
    filtered_norms = np.zeros((n_freq, seq_len))

    for t, state in enumerate(states):
        for f in range(n_freq):
            # Use L2 norm to represent multi-dimensional filtered state as single value
            filtered_norms[f, t] = torch.norm(state.filtered[f][0]).item()

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ========== Panel 1: Training Loss ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history["loss"], linewidth=2.5, color="#2563eb", marker="o", markersize=4)
    ax1.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Loss", fontsize=12, fontweight="bold")
    ax1.set_title("A. Training Convergence", fontsize=13, fontweight="bold", loc="left")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # ========== Panel 2: Multi-Scale Temporal Filtering ==========
    ax2 = fig.add_subplot(gs[0, 1])

    # Plot input observations as scatter
    obs_indices = sequence.argmax(dim=-1).numpy()
    time_steps = np.arange(seq_len)
    ax2.scatter(time_steps, obs_indices, c="black", s=30, alpha=0.6, label="Input Observations", zorder=3)

    # Plot filtered representations at each frequency (normalized)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, n_freq))

    for f in range(n_freq):
        with torch.no_grad():
            alpha = torch.sigmoid(processor.alpha[f]).item()

        # Normalize filtered values to observation scale for visualization
        norm_filtered = filtered_norms[f] / filtered_norms[f].max() * obs_indices.max()
        ax2.plot(time_steps, norm_filtered, linewidth=2.5, color=colors[f], alpha=0.8, label=f"Filter {f} (α={alpha:.3f})")

    ax2.set_xlabel("Time Step", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Signal Magnitude", fontsize=12, fontweight="bold")
    ax2.set_title("B. Multi-Scale Temporal Filtering", fontsize=13, fontweight="bold", loc="left")
    ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # ========== Panel 3: Filter Parameter Evolution ==========
    ax3 = fig.add_subplot(gs[1, 0])

    colors_evol = plt.cm.viridis(np.linspace(0, 1, n_freq))

    for f in range(n_freq):
        ax3.plot(history["alpha_values"][f], linewidth=2.5, color=colors_evol[f], marker="o", markersize=3, label=f"Filter {f}")

    ax3.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax3.set_ylabel("α (Filter Rate)", fontsize=12, fontweight="bold")
    ax3.set_title("C. Learned Filter Parameters", fontsize=13, fontweight="bold", loc="left")
    ax3.legend(loc="best", fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.set_ylim([0, 1])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Add horizontal lines for initial values
    initial_alphas = [history["alpha_values"][f][0] for f in range(n_freq)]
    for f, init_alpha in enumerate(initial_alphas):
        ax3.axhline(y=init_alpha, color=colors_evol[f], linestyle=":", alpha=0.4, linewidth=1)

    # ========== Panel 4: Frequency Response ==========
    ax4 = fig.add_subplot(gs[1, 1])

    omega = np.linspace(0, np.pi, 1000)

    for f in range(n_freq):
        with torch.no_grad():
            alpha = torch.sigmoid(processor.alpha[f]).item()

        # Frequency response magnitude
        magnitude = alpha / np.sqrt(1 + (1 - alpha) ** 2 - 2 * (1 - alpha) * np.cos(omega))

        ax4.plot(omega / np.pi, magnitude, linewidth=2.5, color=colors_evol[f], label=f"Filter {f} (α={alpha:.3f})")

    ax4.set_xlabel("Normalized Frequency (×π rad/sample)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Magnitude Response", fontsize=12, fontweight="bold")
    ax4.set_title("D. Frequency Response Characteristics", fontsize=13, fontweight="bold", loc="left")
    ax4.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, linestyle="--")
    ax4.set_ylim([0, 1.2])
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Add shaded regions to indicate frequency bands
    ax4.axvspan(0, 0.2, alpha=0.1, color="green", label="Low Freq")
    ax4.axvspan(0.2, 0.5, alpha=0.1, color="yellow")
    ax4.axvspan(0.5, 1.0, alpha=0.1, color="red")

    # Overall title
    fig.suptitle("SensoryProcessor: Multi-Scale Temporal Filtering Analysis", fontsize=16, fontweight="bold", y=0.995)

    # Save figure
    output_path = output_dir / "sensory_analysis.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def print_sensory_summary(history: Dict[str, List], n_freq: int) -> None:
    """Print a summary of sensory processing training results.

    Args:
        history: Training history dict with 'loss' and 'alpha_values' keys
        n_freq: Number of frequency modules

    Example:
        >>> print_sensory_summary(history, n_freq=5)
        📊 Summary:
          • Final loss: 3.8831
          • Loss improvement: 2.0%
          • Filter parameters converged: True
    """
    print(f"\n📊 Summary:")
    print(f"  • Final loss: {history['loss'][-1]:.4f}")
    print(f"  • Loss improvement: {(1 - history['loss'][-1]/history['loss'][0])*100:.1f}%")

    # Check convergence (less than 0.1% change in last epoch)
    converged = all(abs(history["alpha_values"][f][-1] - history["alpha_values"][f][-2]) < 0.001 for f in range(n_freq) if len(history["alpha_values"][f]) > 1)

    print(f"  • Filter parameters converged: {converged}")
