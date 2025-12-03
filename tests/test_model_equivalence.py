"""Integration tests comparing legacy Model and refactored TEMModel end-to-end.

This module tests the full forward pass to verify both implementations produce
equivalent results given identical inputs and initial conditions.

Test levels:
1. Single-step forward pass (rtol=1e-5)
2. Multi-step rollout (rtol=1e-4)
3. Loss computation equivalence
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

import parameters
import world
from adapters.legacy_adapter import get_memory_parameters, legacy_to_typed
from model import Iteration
from model import Model as LegacyModel
from torch_tem.model import TEMModel, TEMState


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_test_walk(env_path: str = "./envs/5x5.json", n_steps: int = 5, batch_size: int = 3) -> List:
    """Create identical test walks for both models.

    Args:
        env_path: Path to environment JSON
        n_steps: Number of steps in walk
        batch_size: Number of parallel walks

    Returns:
        List of walk steps in format expected by models:
        [(locations, observations, actions), ...]
    """
    set_seed(42)

    # Create environments
    environments = [world.World(env_path, randomise_observations=True) for _ in range(batch_size)]

    # Generate walks
    walks = [env.generate_walks(n_steps, 1)[0] for env in environments]

    # Format for model input: group by timestep
    formatted_walk = []
    for step_idx in range(n_steps):
        # Collect data from all environments at this timestep
        locations = [walks[env_idx][step_idx][0] for env_idx in range(batch_size)]
        observations = [walks[env_idx][step_idx][1] for env_idx in range(batch_size)]
        actions = [walks[env_idx][step_idx][2] for env_idx in range(batch_size)]

        # Stack observations into batch tensor
        observations = torch.stack(observations, dim=0)

        formatted_walk.append([locations, observations, actions])

    return formatted_walk


def compare_state_fields(legacy_iter: Iteration, refactored_state: State, rtol: float, field_name: str) -> Dict[str, float]:
    """Compare a specific field between legacy Iteration and refactored State.

    Returns dict with comparison metrics.
    """
    legacy_val = getattr(legacy_iter, field_name)
    refactored_val = getattr(refactored_state, field_name)

    # Handle different field types
    if legacy_val is None or refactored_val is None:
        if legacy_val is None and refactored_val is None:
            return {"max_diff": 0.0, "mean_diff": 0.0, "match": True}
        return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}

    # List of tensors (per-frequency fields like g, x, p)
    if isinstance(legacy_val, list) and isinstance(refactored_val, list):
        if len(legacy_val) != len(refactored_val):
            return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}

        # Check if list contains tensors
        if not all(isinstance(item, Tensor) for item in legacy_val + refactored_val):
            # Non-tensor list, skip comparison
            return {"max_diff": 0.0, "mean_diff": 0.0, "match": True, "skipped": True}

        max_diffs = []
        mean_diffs = []
        for l, r in zip(legacy_val, refactored_val):
            if l.shape != r.shape:
                return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}
            diff = (l - r).abs()
            max_diffs.append(diff.max().item())
            mean_diffs.append(diff.mean().item())

        max_diff = max(max_diffs)
        mean_diff = np.mean(mean_diffs)
        match = all(torch.allclose(l, r, rtol=rtol, atol=1e-8) for l, r in zip(legacy_val, refactored_val))

        return {"max_diff": max_diff, "mean_diff": mean_diff, "match": match}

    # Single tensor
    if isinstance(legacy_val, Tensor) and isinstance(refactored_val, Tensor):
        if legacy_val.shape != refactored_val.shape:
            return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}

        diff = (legacy_val - refactored_val).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        match = torch.allclose(legacy_val, refactored_val, rtol=rtol, atol=1e-8)

        return {"max_diff": max_diff, "mean_diff": mean_diff, "match": match}

    # List (e.g., M - list of memory matrices)
    if isinstance(legacy_val, list):
        if len(legacy_val) != len(refactored_val):
            return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}

        # Compare each matrix in list
        max_diffs = []
        mean_diffs = []
        matches = []
        for l, r in zip(legacy_val, refactored_val):
            if l.shape != r.shape:
                return {"max_diff": float("inf"), "mean_diff": float("inf"), "match": False}
            diff = (l - r).abs()
            max_diffs.append(diff.max().item())
            mean_diffs.append(diff.mean().item())
            matches.append(torch.allclose(l, r, rtol=rtol, atol=1e-8))

        return {"max_diff": max(max_diffs), "mean_diff": np.mean(mean_diffs), "match": all(matches)}

    # Other types (locations, actions) - skip numerical comparison
    return {"max_diff": 0.0, "mean_diff": 0.0, "match": True, "skipped": True}


def test_single_step_equivalence():
    """Test that a single forward step produces identical results."""
    print("Testing single-step forward pass equivalence...")
    print("-" * 70)

    # Create models with identical configuration
    legacy_params = parameters.parameters()
    legacy_params["batch_size"] = 3

    set_seed(42)
    legacy_model = LegacyModel(legacy_params)
    legacy_model.eval()

    typed_config = legacy_to_typed(legacy_params)
    set_seed(42)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(3)
    refactored_model.eval()

    # Create test walk (1 step)
    walk = create_test_walk(n_steps=1, batch_size=3)

    # Run both models
    with torch.no_grad():
        legacy_steps = legacy_model(walk, prev_iter=None, prev_M=None)
        refactored_steps = refactored_model(walk, prev_M=None)

    # Compare results
    print(f"\nLegacy model returned {len(legacy_steps)} steps")
    print(f"Refactored model returned {len(refactored_steps)} steps")

    if len(legacy_steps) != len(refactored_steps):
        print(f"✗ FAILED: Different number of steps")
        return False

    # Compare each step
    rtol = 1e-5
    step = legacy_steps[0]
    ref_step = refactored_steps[0]

    print(f"\nComparing step 0 (tolerance rtol={rtol}):")
    print(f"{'Field':<15} {'Max Diff':<12} {'Mean Diff':<12} {'Match':<8}")
    print("-" * 50)

    fields_to_compare = ["g", "x", "a", "L", "M", "g_gen", "p_gen", "x_gen", "x_logits", "x_inf", "g_inf", "p_inf"]

    all_match = True
    for field in fields_to_compare:
        result = compare_state_fields(step, ref_step, rtol, field)

        if result.get("skipped"):
            status = "SKIP"
        elif result["match"]:
            status = "✓"
        else:
            status = "✗"
            all_match = False

        print(f"{field:<15} {result['max_diff']:<12.2e} {result['mean_diff']:<12.2e} {status:<8}")

    print("-" * 50)
    if all_match:
        print("✓ All fields match within tolerance!")
        return True
    else:
        print("✗ Some fields do not match")
        return False


def test_multi_step_equivalence():
    """Test that a multi-step rollout produces equivalent results."""
    print("\n\nTesting multi-step rollout equivalence (20 steps)...")
    print("-" * 70)

    # Create models
    legacy_params = parameters.parameters()
    legacy_params["batch_size"] = 3

    set_seed(42)
    legacy_model = LegacyModel(legacy_params)
    legacy_model.eval()

    typed_config = legacy_to_typed(legacy_params)
    set_seed(42)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(3)
    refactored_model.eval()

    # Create longer test walk
    walk = create_test_walk(n_steps=20, batch_size=3)

    # Run both models
    with torch.no_grad():
        legacy_steps = legacy_model(walk, prev_iter=None, prev_M=None)
        refactored_steps = refactored_model(walk, prev_M=None)

    print(f"\nProcessed {len(legacy_steps)} steps")

    # Relaxed tolerance for accumulated errors
    rtol = 1e-4

    # Compare final step
    step = legacy_steps[-1]
    ref_step = refactored_steps[-1]

    print(f"\nComparing final step (tolerance rtol={rtol}):")
    print(f"{'Field':<15} {'Max Diff':<12} {'Mean Diff':<12} {'Match':<8}")
    print("-" * 50)

    fields_to_compare = ["g_inf", "p_inf", "x_gen", "L", "M"]

    all_match = True
    for field in fields_to_compare:
        result = compare_state_fields(step, ref_step, rtol, field)

        if result.get("skipped"):
            status = "SKIP"
        elif result["match"]:
            status = "✓"
        else:
            status = "✗"
            all_match = False

        print(f"{field:<15} {result['max_diff']:<12.2e} {result['mean_diff']:<12.2e} {status:<8}")

    # Track divergence over time
    print(f"\nDivergence over time (g_inf max diff):")
    divergence = []
    for i, (legacy_step, ref_step) in enumerate(zip(legacy_steps, refactored_steps)):
        result = compare_state_fields(legacy_step, ref_step, rtol, "g_inf")
        divergence.append(result["max_diff"])
        if i % 5 == 0:
            print(f"  Step {i:2d}: {result['max_diff']:.2e}")

    print("-" * 50)
    if all_match:
        print("✓ Multi-step rollout matches within tolerance!")
        return True
    else:
        print("✗ Multi-step rollout diverges beyond tolerance")
        return False


def test_loss_computation():
    """Test that loss computation produces equivalent results."""
    print("\n\nTesting loss computation equivalence...")
    print("-" * 70)

    # Create models
    legacy_params = parameters.parameters()
    legacy_params["batch_size"] = 3

    set_seed(42)
    legacy_model = LegacyModel(legacy_params)
    legacy_model.eval()

    typed_config = legacy_to_typed(legacy_params)
    set_seed(42)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(3)
    refactored_model.eval()

    # Create test walk
    walk = create_test_walk(n_steps=5, batch_size=3)

    # Run both models
    with torch.no_grad():
        legacy_steps = legacy_model(walk, prev_iter=None, prev_M=None)
        refactored_steps = refactored_model(walk, prev_M=None)

    # Compare losses for each step
    print(f"\nLoss components (8 components per step):")
    print(f"{'Step':<6} {'Component':<4} {'Legacy':<12} {'Refactored':<12} {'Diff':<12}")
    print("-" * 60)

    rtol = 1e-5
    max_loss_diff = 0.0

    for step_idx in range(min(3, len(legacy_steps))):  # Check first 3 steps
        legacy_L = legacy_steps[step_idx].L
        refactored_L = refactored_steps[step_idx].L

        for comp_idx in range(8):
            legacy_val = legacy_L[comp_idx].mean().item()
            refactored_val = refactored_L[comp_idx].mean().item()
            diff = abs(legacy_val - refactored_val)
            max_loss_diff = max(max_loss_diff, diff)

            print(f"{step_idx:<6} {comp_idx:<4} {legacy_val:<12.4f} {refactored_val:<12.4f} {diff:<12.2e}")

    print("-" * 60)
    print(f"Maximum loss difference: {max_loss_diff:.2e}")

    # Check if within tolerance
    if max_loss_diff < 1e-3:  # Relaxed tolerance for losses
        print("✓ Loss computation matches within tolerance!")
        return True
    else:
        print("✗ Loss computation differs significantly")
        return False


# ======================================================================================
# MAIN TEST RUNNER
# ======================================================================================


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("INTEGRATION TESTS: Legacy Model vs Refactored TEMModel")
    print("=" * 70)
    print()

    results = {
        "Single-step equivalence": test_single_step_equivalence(),
        "Multi-step equivalence": test_multi_step_equivalence(),
        "Loss computation": test_loss_computation(),
    }

    print("\n\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")

    passed_count = sum(results.values())
    total_count = len(results)

    print("=" * 70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print("=" * 70)

    return all(results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
