"""
Visualization Functions for Sensory Processing

This module provides comprehensive visualization tools for analyzing and
understanding the SensoryProcessor's multi-scale temporal filtering behavior,
compression mechanisms, and frequency response characteristics.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from torch_tem.modules.sensory import SensoryProcessor, SensoryState


def plot_sensory_analysis(
    processor: SensoryProcessor,
    sequence: Tensor,
    output_dir: Path,
    dpi: int = 200,
) -> Path:
    """Generate comprehensive visualization of SensoryProcessor behavior.

    Creates a single publication-quality figure with 4 panels demonstrating:
    - A. Compression mechanism: One-hot → Two-hot encoding
    - B. Multi-scale temporal filtering across time steps
    - C. Filter parameter values and frequency hierarchy
    - D. Frequency response characteristics of each filter

    This visualization helps understand:
    - How observations are compressed while preserving distinctiveness
    - How different filters track observations at different time scales
    - The relationship between filter parameters (α) and frequency response
    - Which frequency bands each filter captures

    Args:
        processor: SensoryProcessor instance (trained or initialized)
        sequence: [seq_len, n_obs] one-hot encoded observation sequence
        output_dir: Directory to save the figure
        dpi: Resolution for saved figure (default: 200)

    Returns:
        Path to the saved figure

    Example:
        >>> from torch_tem.figures.sensory import plot_sensory_analysis
        >>> output_path = plot_sensory_analysis(
        ...     processor=sensory_processor,
        ...     sequence=test_sequence,
        ...     output_dir=Path("outputs/figures")
        ... )
        >>> print(f"Saved to {output_path}")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_freq = processor.n_freq
    n_obs = processor.two_hot_table.shape[0]
    n_compressed = processor.two_hot_table.shape[1]
    seq_len = sequence.shape[0]

    # Process sequence through sensory processor
    with torch.no_grad():
        x_prev = [torch.zeros(1, n_compressed) for _ in range(n_freq)]
        states = []
        for t in range(seq_len):
            x_raw = sequence[t : t + 1]
            state = processor(x_raw, x_prev)
            states.append(state)
            x_prev = state.filtered

    # Extract data for visualization
    obs_indices = sequence.argmax(dim=-1).numpy()
    time_steps = np.arange(seq_len)

    # Filtered values over time (L2 norm for visualization)
    filtered_norms = np.zeros((n_freq, seq_len))
    for t, state in enumerate(states):
        for f in range(n_freq):
            filtered_norms[f, t] = torch.norm(state.filtered[f][0]).item()

    # Get filter parameters
    with torch.no_grad():
        alphas = [torch.sigmoid(processor.alpha[f]).item() for f in range(n_freq)]

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # ========== Panel 1: Compression Mechanism ==========
    ax1 = fig.add_subplot(gs[0, 0])

    # Show a few example compressions
    n_examples = min(8, n_obs)
    example_indices = np.linspace(0, n_obs - 1, n_examples, dtype=int)

    for i, obs_idx in enumerate(example_indices):
        one_hot = torch.zeros(1, n_obs)
        one_hot[0, obs_idx] = 1.0
        two_hot = processor.compress(one_hot)[0].numpy()

        # Plot as bars
        x_pos = np.arange(n_compressed) + i * (n_compressed + 1)
        ax1.bar(x_pos, two_hot, width=0.8, label=f"Obs {obs_idx}" if i < 4 else None)

    ax1.set_xlabel("Compressed Dimension Index", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Activation", fontsize=12, fontweight="bold")
    ax1.set_title(f"A. Compression: One-Hot ({n_obs}D) → Two-Hot ({n_compressed}D)", fontsize=13, fontweight="bold", loc="left")
    ax1.legend(loc="upper right", fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")
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

    # ========== Panel 3: Filter Parameters ==========
    ax3 = fig.add_subplot(gs[1, 0])

    colors_param = plt.cm.viridis(np.linspace(0, 1, n_freq))

    # Plot filter rates as bars
    x_pos = np.arange(n_freq)
    bars = ax3.bar(x_pos, alphas, color=colors_param, width=0.6, edgecolor="black", linewidth=1.5)

    # Add value labels on bars
    for i, (bar, alpha) in enumerate(zip(bars, alphas)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{alpha:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax3.set_xlabel("Filter Index (High → Low Frequency)", fontsize=12, fontweight="bold")
    ax3.set_ylabel("α (Filter Rate)", fontsize=12, fontweight="bold")
    ax3.set_title("C. Filter Parameter Hierarchy", fontsize=13, fontweight="bold", loc="left")
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f"F{f}" for f in range(n_freq)])
    ax3.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax3.set_ylim([0, 1.05])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Add reference lines
    ax3.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=1, label="α = 0.5 (balanced)")
    ax3.legend(loc="upper right", fontsize=9)

    # ========== Panel 4: Frequency Response ==========
    ax4 = fig.add_subplot(gs[1, 1])

    omega = np.linspace(0, np.pi, 1000)
    colors_freq = plt.cm.viridis(np.linspace(0, 1, n_freq))

    for f in range(n_freq):
        alpha = alphas[f]

        # Frequency response magnitude: |H(ω)| = α / sqrt(1 + (1-α)² - 2(1-α)cos(ω))
        magnitude = alpha / np.sqrt(1 + (1 - alpha) ** 2 - 2 * (1 - alpha) * np.cos(omega))

        ax4.plot(omega / np.pi, magnitude, linewidth=2.5, color=colors_freq[f], label=f"Filter {f} (α={alpha:.3f})")

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
    fig.suptitle("SensoryProcessor: Multi-Scale Temporal Filtering Mechanism", fontsize=16, fontweight="bold", y=0.995)

    # Save figure
    output_path = output_dir / "sensory_processor_analysis.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()

    return output_path


def print_processor_summary(processor: SensoryProcessor) -> None:
    """Print a summary of sensory processor configuration and parameters.

    Args:
        processor: SensoryProcessor instance to summarize

    Example:
        >>> print_processor_summary(processor)
        📊 SensoryProcessor Configuration:
          • Observations: 45 → Compressed: 10 (22.2% size)
          • Frequency modules: 5
          • Filter rates (α): [0.900, 0.300, 0.090, 0.030, 0.010]
          • Memory projection: 10 → 300 dimensions
    """
    n_obs = processor.two_hot_table.shape[0]
    n_compressed = processor.two_hot_table.shape[1]
    n_freq = processor.n_freq
    n_memory = processor.W_tile[0].shape[1]

    with torch.no_grad():
        alphas = [torch.sigmoid(processor.alpha[f]).item() for f in range(n_freq)]

    print(f"\n📊 SensoryProcessor Configuration:")
    print(f"  • Observations: {n_obs} → Compressed: {n_compressed} ({n_compressed/n_obs*100:.1f}% size)")
    print(f"  • Frequency modules: {n_freq}")
    print(f"  • Filter rates (α): {[f'{a:.3f}' for a in alphas]}")
    print(f"  • Memory projection: {n_compressed} → {n_memory} dimensions")

    # Classify filters
    fast = sum(1 for a in alphas if a > 0.5)
    slow = sum(1 for a in alphas if a <= 0.5)
    print(f"  • Filter classification: {fast} fast (α>0.5), {slow} slow (α≤0.5)")
