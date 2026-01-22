"""Diagnostics extraction from TEM rollouts.

Converts live Rollout iterators into plot-ready TEMRolloutTrace dataclasses.
All GPU tensors are detached and moved to CPU. Supports downsampling to
reduce memory footprint for large rollouts.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from torch_tem.core.model import Rollout, TEMLabel, TEMOutput, TEMState
from torch_tem.diagnostics.rollout_trace import TEMRolloutTrace


def extract_rollout_trace(rollout: Rollout, *, max_steps: Optional[int] = None, downsample_stride: int = 1) -> TEMRolloutTrace:
    """Extract a plot-ready trace from a TEM Rollout.

    Iterates through the rollout, collecting outputs, labels, and states.
    All tensors are detached and moved to CPU to prevent GPU memory pressure.

    Args:
        rollout: Active Rollout iterator over a walk.
        max_steps: Maximum number of steps to extract (None = extract all).
        downsample_stride: Keep every stride-th step (1 = keep all).

    Returns:
        TEMRolloutTrace with CPU-based NumPy arrays.

    Raises:
        ValueError: If rollout is empty or produces no steps.

    Example:
        >>> from torch_tem.core.model import Rollout
        >>> rollout = Rollout(model, chunk)
        >>> trace = extract_rollout_trace(rollout, max_steps=100)
        >>> # trace.o_predicted is a NumPy array on CPU
    """
    outputs: List[TEMOutput] = []
    labels: List[TEMLabel] = []
    states: List[TEMState] = []

    step_count = 0
    for output, label, state in rollout:
        # Collect every stride-th step
        if step_count % downsample_stride == 0:
            outputs.append(output)
            labels.append(label)
            states.append(state)

        step_count += 1
        if max_steps is not None and step_count >= max_steps:
            break

    if len(outputs) == 0:
        raise ValueError("extract_rollout_trace: rollout produced no outputs")

    # Extract batch size from first output
    batch_size = outputs[0].reconstruction.o_hat[0].shape[0]
    n_steps = len(outputs)
    n_frequencies = len(outputs[0].inference.p_inf)

    # Preallocate arrays
    n_obs = outputs[0].reconstruction.o_hat[0].shape[1]
    o_predicted = np.zeros((n_steps, batch_size, n_obs), dtype=np.float32)
    o_true = np.zeros((n_steps, batch_size, n_obs), dtype=np.float32)

    # Infer dimensions from first output
    n_place = [outputs[0].inference.p_inf[f].shape[1] for f in range(n_frequencies)]
    n_grid = [outputs[0].inference.g_inf[f].shape[1] for f in range(n_frequencies)]

    p_inf = [np.zeros((n_steps, batch_size, n_place[f]), dtype=np.float32) for f in range(n_frequencies)]
    p_gen = [np.zeros((n_steps, batch_size, n_place[f]), dtype=np.float32) for f in range(n_frequencies)]
    g_inf = [np.zeros((n_steps, batch_size, n_grid[f]), dtype=np.float32) for f in range(n_frequencies)]
    g_gen = [np.zeros((n_steps, batch_size, n_grid[f]), dtype=np.float32) for f in range(n_frequencies)]

    # Check if p_xi is available (depends on use_x_cued_recall setting)
    has_p_xi = outputs[0].inference.p_xi is not None
    if has_p_xi:
        p_xi = [np.zeros((n_steps, batch_size, n_place[f]), dtype=np.float32) for f in range(n_frequencies)]
    else:
        p_xi = None

    location_ids = np.zeros((n_steps, batch_size), dtype=np.int32)
    actions = np.zeros((n_steps, batch_size), dtype=np.int32)

    # Fill arrays
    for t, (output, label, state) in enumerate(zip(outputs, labels, states)):
        # Sensory reconstructions (use first output: o_p_inf)
        o_predicted[t] = output.reconstruction.o_hat[0].detach().cpu().numpy()
        o_true[t] = label.o.detach().cpu().numpy()

        # Location representations
        for f in range(n_frequencies):
            p_inf[f][t] = output.inference.p_inf[f].detach().cpu().numpy()
            p_gen[f][t] = output.generative.p_gen_gi[f].detach().cpu().numpy()
            g_inf[f][t] = output.inference.g_inf[f].detach().cpu().numpy()
            g_gen[f][t] = output.generative.g_gen[f].detach().cpu().numpy()

            if has_p_xi:
                p_xi[f][t] = output.inference.p_xi[f].detach().cpu().numpy()

        # Labels
        location_ids[t] = np.array([loc["id"] for loc in label.locations], dtype=np.int32)
        # Actions: extract from label if available, otherwise use -1 as placeholder
        # (First step in a walk may not have an action)
        actions[t] = -1  # Placeholder; action extraction needs walk context

    return TEMRolloutTrace(
        n_steps=n_steps,
        batch_size=batch_size,
        o_predicted=o_predicted,
        o_true=o_true,
        p_inf=p_inf,
        p_gen=p_gen,
        g_inf=g_inf,
        g_gen=g_gen,
        p_xi=p_xi,
        location_ids=location_ids,
        actions=actions,
        env_names=None,
    )


def compute_sensory_accuracy(trace: TEMRolloutTrace) -> np.ndarray:
    """Compute per-step sensory prediction accuracy from a trace.

    Args:
        trace: TEMRolloutTrace with o_predicted and o_true.

    Returns:
        NumPy array of shape (n_steps, batch_size) with binary accuracy
        (1.0 if argmax matches, 0.0 otherwise).
    """
    pred_labels = np.argmax(trace.o_predicted, axis=-1)  # (T, B)
    true_labels = np.argmax(trace.o_true, axis=-1)  # (T, B)
    return (pred_labels == true_labels).astype(np.float32)


def compute_location_uncertainty(trace: TEMRolloutTrace, freq_idx: int = 0) -> np.ndarray:
    """Compute location belief uncertainty (entropy) from place cell distributions.

    Args:
        trace: TEMRolloutTrace with p_inf.
        freq_idx: Frequency module index to analyze.

    Returns:
        NumPy array of shape (n_steps, batch_size) with entropy values.
    """
    p = trace.p_inf[freq_idx]  # (T, B, n_place)
    # Add small epsilon to prevent log(0)
    eps = 1e-10
    p_safe = np.clip(p, eps, 1.0)
    entropy = -np.sum(p_safe * np.log(p_safe), axis=-1)  # (T, B)
    return entropy
