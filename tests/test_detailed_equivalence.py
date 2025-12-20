"""Detailed component integration tests to isolate M and p_gen discrepancies.

This module provides granular tests for:
1. Memory update (Hebbian plasticity) equivalence
2. Generative pathway (p_gen) equivalence
3. Step-by-step state comparison
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from adapters.legacy_adapter import legacy_to_typed

import parameters
from model import Model as LegacyModel
from torch_tem import utils as tem_utils
from torch_tem.core.model import TEMModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_test_models(batch_size=3):
    """Create matched legacy and refactored models."""
    legacy_params = parameters.parameters()
    legacy_params["batch_size"] = batch_size

    set_seed(42)
    legacy_model = LegacyModel(legacy_params)
    legacy_model.eval()

    typed_config = legacy_to_typed(legacy_params)
    set_seed(42)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(batch_size)
    refactored_model.eval()

    # Copy weights
    _copy_weights(legacy_model, refactored_model, legacy_params)

    return legacy_model, refactored_model, legacy_params


def _copy_weights(legacy, refactored, params):
    """Copy all weights from legacy to refactored model."""
    from tests.test_component_parity import _copy_weights as copy_weights_impl

    copy_weights_impl(legacy, refactored, params)


def test_memory_update_equivalence():
    """Test that Hebbian memory update produces identical results."""
    print("\n" + "=" * 70)
    print("TEST 1: Memory Update (Hebbian Plasticity) Equivalence")
    print("=" * 70)

    legacy, refactored, params = create_test_models(batch_size=3)

    batch_size = 3
    n_p = params["n_p"]
    n_p_total = sum(n_p)

    # Create test inputs
    p_inf = [torch.randn(batch_size, n_p[f]) for f in range(len(n_p))]
    p_gen = [torch.randn(batch_size, n_p[f]) for f in range(len(n_p))]
    p_inf_x = [torch.randn(batch_size, n_p[f]) for f in range(len(n_p))]

    # Initial memory (all zeros)
    M_gen = torch.zeros(batch_size, n_p_total, n_p_total)
    M_inf = torch.zeros(batch_size, n_p_total, n_p_total)

    eta = params["eta"]
    kappa = params["kappa"]
    lamb = 1.0 - kappa

    # Legacy update
    p_inf_flat = torch.cat(p_inf, dim=1)
    p_gen_flat = torch.cat(p_gen, dim=1)
    p_inf_x_flat = torch.cat(p_inf_x, dim=1)

    M_gen_legacy = legacy.hebbian(M_gen, p_inf_flat, p_gen_flat)
    M_inf_legacy = legacy.hebbian(M_inf, p_inf_flat, p_inf_x_flat, do_hierarchical_connections=False)

    # Refactored update
    refactored.storage.M_gen = M_gen.clone()
    refactored.storage.M_inf = M_inf.clone()
    refactored.storage.update(p_inf_flat, p_gen_flat, eta, lamb, p_retrieved=p_inf_x_flat)
    M_gen_refactored = refactored.storage.M_gen
    M_inf_refactored = refactored.storage.M_inf

    # Compare
    M_gen_diff = (M_gen_legacy - M_gen_refactored).abs()
    M_inf_diff = (M_inf_legacy - M_inf_refactored).abs()

    print(f"\nM_gen comparison:")
    print(f"  Max diff: {M_gen_diff.max().item():.6e}")
    print(f"  Mean diff: {M_gen_diff.mean().item():.6e}")
    print(f"  Match (rtol=1e-5): {torch.allclose(M_gen_legacy, M_gen_refactored, rtol=1e-5, atol=1e-8)}")

    print(f"\nM_inf comparison:")
    print(f"  Max diff: {M_inf_diff.max().item():.6e}")
    print(f"  Mean diff: {M_inf_diff.mean().item():.6e}")
    print(f"  Match (rtol=1e-5): {torch.allclose(M_inf_legacy, M_inf_refactored, rtol=1e-5, atol=1e-8)}")

    # Check if close enough
    m_gen_pass = M_gen_diff.max().item() < 1e-6
    m_inf_pass = M_inf_diff.max().item() < 1e-6

    if m_gen_pass and m_inf_pass:
        print("\n✓ PASSED: Memory updates match within tolerance")
        return True
    else:
        print("\n✗ FAILED: Memory updates differ")
        return False


def test_generative_pathway_equivalence():
    """Test that generative pathway (g_inf -> p_gen) produces identical results."""
    print("\n" + "=" * 70)
    print("TEST 2: Generative Pathway (g_inf -> p_gen) Equivalence")
    print("=" * 70)

    legacy, refactored, params = create_test_models(batch_size=3)

    batch_size = 3
    n_f = params["n_f"]
    n_g = params["n_g"]
    n_p = params["n_p"]
    n_p_total = sum(n_p)

    # Create test inputs
    g_inf = [torch.randn(batch_size, n_g[f]) for f in range(n_f)]
    M_gen = torch.randn(batch_size, n_p_total, n_p_total) * 0.1  # Small random memory

    # Legacy generative
    p_gen_legacy = legacy.gen_p(g_inf, M_gen)

    # Refactored generative
    # gen_p expects g: AbstractLocation (list[Tensor]) and M_prev: Matrix
    p_gen_refactored = refactored.generative.gen_p(g_inf, M_gen)

    # Compare
    print(f"\np_gen comparison:")
    for f in range(n_f):
        diff = (p_gen_legacy[f] - p_gen_refactored[f]).abs()
        print(f"  Frequency {f}:")
        print(f"    Max diff: {diff.max().item():.6e}")
        print(f"    Mean diff: {diff.mean().item():.6e}")
        print(f"    Match: {torch.allclose(p_gen_legacy[f], p_gen_refactored[f], rtol=1e-5, atol=1e-8)}")

    # Overall match
    all_match = all(torch.allclose(p_gen_legacy[f], p_gen_refactored[f], rtol=1e-5, atol=1e-8) for f in range(n_f))

    if all_match:
        print("\n✓ PASSED: Generative pathway matches")
        return True
    else:
        print("\n✗ FAILED: Generative pathway differs")
        return False


def test_attractor_retrieval_equivalence():
    """Test that attractor memory retrieval produces identical results."""
    print("\n" + "=" * 70)
    print("TEST 3: Attractor Memory Retrieval Equivalence")
    print("=" * 70)

    legacy, refactored, params = create_test_models(batch_size=3)

    batch_size = 3
    n_p = params["n_p"]
    n_p_total = sum(n_p)

    # Create test inputs
    query = [torch.randn(batch_size, n_p[f]) for f in range(len(n_p))]
    M = torch.randn(batch_size, n_p_total, n_p_total) * 0.1

    # Legacy retrieval
    p_retrieved_legacy = legacy.attractor(query, M, retrieve_it_mask=params["p_retrieve_mask_inf"])

    # Refactored retrieval
    query_flat = torch.cat(query, dim=1)
    p_retrieved_flat = refactored.attractor.retrieve(query_flat, M, for_inference=True)

    # Split back to frequencies
    p_retrieved_refactored = tem_utils.split_to_frequencies(p_retrieved_flat, n_p)

    # Compare
    print(f"\nAttractor retrieval comparison:")
    for f in range(len(n_p)):
        diff = (p_retrieved_legacy[f] - p_retrieved_refactored[f]).abs()
        print(f"  Frequency {f}:")
        print(f"    Max diff: {diff.max().item():.6e}")
        print(f"    Mean diff: {diff.mean().item():.6e}")
        print(f"    Match: {torch.allclose(p_retrieved_legacy[f], p_retrieved_refactored[f], rtol=1e-5, atol=1e-8)}")

    all_match = all(torch.allclose(p_retrieved_legacy[f], p_retrieved_refactored[f], rtol=1e-5, atol=1e-8) for f in range(len(n_p)))

    if all_match:
        print("\n✓ PASSED: Attractor retrieval matches")
        return True
    else:
        print("\n✗ FAILED: Attractor retrieval differs")
        return False


def test_full_iteration_step_by_step():
    """Test a full iteration with intermediate state comparisons."""
    print("\n" + "=" * 70)
    print("TEST 4: Full Iteration with Step-by-Step Validation")
    print("=" * 70)

    legacy, refactored, params = create_test_models(batch_size=3)

    # Disable sampling
    legacy.hyper["do_sample"] = False
    refactored.config.do_sample = False

    # Use the same walk data format as the integration test
    from test_model_equivalence import _copy_weights, create_test_walk

    # Copy weights using the integration test's method (which we know works)
    _copy_weights(legacy, refactored, params)

    # Create a single-step walk with real environment data
    walk = create_test_walk(n_steps=1, batch_size=3)
    first_step = walk[0]
    locations, x, actions = first_step

    # actions are actual action IDs, we need None for first step
    prev_actions = [None] * 3

    # Initialize
    n_p_total = sum(params["n_p"])
    M_init = torch.zeros(3, n_p_total, n_p_total)

    print("\n--- Running Legacy Iteration ---")
    with torch.no_grad():
        legacy_iter = legacy.init_iteration(None, x, None, [M_init, M_init])
        L_legacy, M_legacy, g_gen_legacy, p_gen_legacy, x_gen_legacy, x_logits_legacy, x_inf_legacy, g_inf_legacy, p_inf_legacy = legacy.iteration(
            x, locations, prev_actions, legacy_iter.M, legacy_iter.x_inf, legacy_iter.g_inf
        )

    print("✓ Legacy iteration complete")
    print(f"  g_inf norm: {[t.norm().item() for t in g_inf_legacy]}")
    print(f"  p_inf norm: {[t.norm().item() for t in p_inf_legacy]}")
    print(f"  p_gen norm: {[t.norm().item() for t in p_gen_legacy]}")
    print(f"  M[0] norm: {M_legacy[0].norm().item():.6f}")

    print("\n--- Running Refactored Iteration ---")
    with torch.no_grad():
        state_init = refactored.init_state(x, [M_init, M_init])
        losses, state = refactored(x, locations, prev_actions, state_init)

    print("✓ Refactored iteration complete")
    print(f"  g_inf norm: {[t.norm().item() for t in state.inference_state.latent_prediction.abstract]}")
    print(f"  p_inf norm: {[t.norm().item() for t in state.inference_state.latent_prediction.grounded]}")
    print(f"  p_gen norm: {[t.norm().item() for t in state.generative_state.p_g]}")
    print(f"  M_gen norm: {state.generative_state.memory_gen.norm().item():.6f}")

    # Compare key outputs
    print("\n--- Comparison ---")

    # g_inf
    g_inf_diff = max((g_inf_legacy[f] - state.inference_state.latent_prediction.abstract[f]).abs().max().item() for f in range(len(g_inf_legacy)))
    print(f"g_inf max diff: {g_inf_diff:.6e}")

    # p_inf
    p_inf_diff = max((p_inf_legacy[f] - state.inference_state.latent_prediction.grounded[f]).abs().max().item() for f in range(len(p_inf_legacy)))
    print(f"p_inf max diff: {p_inf_diff:.6e}")

    # p_gen
    p_gen_diff = max((p_gen_legacy[f] - state.generative_state.p_g[f]).abs().max().item() for f in range(len(p_gen_legacy)))
    print(f"p_gen max diff: {p_gen_diff:.6e}")

    # M
    M_gen_diff = (M_legacy[0] - state.generative_state.memory_gen).abs().max().item()
    print(f"M_gen max diff: {M_gen_diff:.6e}")

    # Check thresholds
    thresholds = {
        "g_inf": 1e-6,
        "p_inf": 1e-6,
        "p_gen": 2e-5,  # Slightly higher tolerance
        "M_gen": 1e-4,  # Higher tolerance for memory
    }

    passed = g_inf_diff < thresholds["g_inf"] and p_inf_diff < thresholds["p_inf"] and p_gen_diff < thresholds["p_gen"] and M_gen_diff < thresholds["M_gen"]

    if passed:
        print("\n✓ PASSED: All components within tolerance")
        return True
    else:
        print("\n✗ FAILED: Some components exceed tolerance")
        print(f"  Thresholds: {thresholds}")
        return False


def main():
    """Run all detailed tests."""
    print("=" * 70)
    print("DETAILED EQUIVALENCE TESTS")
    print("=" * 70)

    results = []

    # Test 1: Memory update
    results.append(("Memory Update", test_memory_update_equivalence()))

    # Test 2: Generative pathway
    results.append(("Generative Pathway", test_generative_pathway_equivalence()))

    # Test 3: Attractor retrieval
    results.append(("Attractor Retrieval", test_attractor_retrieval_equivalence()))

    # Test 4: Full iteration
    results.append(("Full Iteration", test_full_iteration_step_by_step()))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name:.<50} {status}")

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    print("=" * 70)
    print(f"TOTAL: {passed_count}/{total_count} tests passed")
    print("=" * 70)

    return all(passed for _, passed in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
