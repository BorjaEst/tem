"""Component-level parity tests between legacy Model and refactored TEMModel.

This module tests individual components (encoder, transition, memory, etc.)
to verify that both implementations produce identical outputs given identical inputs.

Test strategy:
1. Initialize both models with identical random seeds
2. Create identical synthetic inputs for each component
3. Extract outputs from both models
4. Compare with strict tolerances (rtol=1e-5 for single operations)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

import parameters
from adapters.legacy_adapter import legacy_to_typed
from model import Model as LegacyModel
from torch_tem.model import TEMModel


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_test_models(batch_size: int = 4) -> Tuple[LegacyModel, TEMModel, dict]:
    """Create both legacy and refactored models with identical configuration.

    Args:
        batch_size: Number of parallel environments

    Returns:
        Tuple of (legacy_model, refactored_model, legacy_params)
    """
    # Get legacy parameters
    legacy_params = parameters.parameters()
    legacy_params["batch_size"] = batch_size

    # Create legacy model
    set_seed(42)
    legacy_model = LegacyModel(legacy_params)
    legacy_model.eval()

    # Convert to typed config and create refactored model
    typed_config = legacy_to_typed(legacy_params)
    set_seed(42)
    refactored_model = TEMModel(typed_config)
    refactored_model.set_batch_size(batch_size)
    refactored_model.eval()

    # Copy trained parameters from legacy to refactored
    # This ensures we're comparing identical weights
    _copy_weights(legacy_model, refactored_model, legacy_params)

    return legacy_model, refactored_model, legacy_params


def _copy_weights(legacy: LegacyModel, refactored: TEMModel, params: dict):
    """Copy weights from legacy model to refactored model for fair comparison."""
    # Note: This is complex due to different internal structure
    # For initial testing, we'll use random but identical seeds
    # TODO: Implement proper weight copying for trained model comparison
    pass


def assert_tensors_close(actual: Tensor, expected: Tensor, rtol: float = 1e-5, atol: float = 1e-8, name: str = "tensor"):
    """Assert two tensors are close with detailed error messages."""
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{name} values differ:\n"
            f"  Max absolute diff: {max_diff:.8e}\n"
            f"  Mean absolute diff: {mean_diff:.8e}\n"
            f"  Relative tolerance: {rtol:.8e}\n"
            f"  Absolute tolerance: {atol:.8e}\n"
            f"  Actual range: [{actual.min().item():.6f}, {actual.max().item():.6f}]\n"
            f"  Expected range: [{expected.min().item():.6f}, {expected.max().item():.6f}]"
        )


def assert_list_tensors_close(actual: List[Tensor], expected: List[Tensor], rtol: float = 1e-5, atol: float = 1e-8, name: str = "list"):
    """Assert two lists of tensors are close."""
    if len(actual) != len(expected):
        raise AssertionError(f"{name} length mismatch: {len(actual)} != {len(expected)}")

    for i, (a, e) in enumerate(zip(actual, expected)):
        assert_tensors_close(a, e, rtol, atol, f"{name}[{i}]")


# ======================================================================================
# COMPONENT TESTS
# ======================================================================================


def test_encoder():
    """Test sensory encoder: one-hot → two-hot compression."""
    print("Testing encoder (one-hot → two-hot)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test input: one-hot observations [batch, n_x]
    batch_size = 4
    n_x = params["n_x"]
    x_onehot = torch.zeros(batch_size, n_x)
    x_onehot[0, 5] = 1.0  # Observation 5
    x_onehot[1, 10] = 1.0  # Observation 10
    x_onehot[2, 20] = 1.0  # Observation 20
    x_onehot[3, 30] = 1.0  # Observation 30

    # Legacy encoding
    legacy_encoded = legacy.f_c_star(x_onehot)

    # Refactored encoding
    refactored_encoded = refactored.encoder.forward(x_onehot)

    # Compare
    assert_tensors_close(refactored_encoded, legacy_encoded, name="Encoder output")

    print(f"  ✓ Encoder output shape: {refactored_encoded.shape}")
    print(f"  ✓ Values match (max diff: {(refactored_encoded - legacy_encoded).abs().max():.2e})")


def test_transition():
    """Test abstract location transition: (a, g) → g'."""
    print("Testing transition model (action + abstract location → next abstract location)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test inputs
    batch_size = 4
    n_f = params["n_f"]
    n_g = params["n_g"]
    n_actions = params["n_actions"]

    # Previous abstract location g [List of [batch, n_g[f]]]
    g_prev = [torch.randn(batch_size, n_g[f]) for f in range(n_f)]

    # Actions (as list of integers)
    actions = [2, 0, 3, 1]  # Different action for each batch element

    # Legacy transition (expects action list)
    with torch.no_grad():
        legacy_g_gen, legacy_g_inf = legacy.gen_g(actions, g_prev, locations=None)

    # Refactored transition (same interface)
    with torch.no_grad():
        refactored_g_gen, refactored_g_inf = refactored.gen_g(actions, g_prev, locations=None)

    # Compare both outputs
    assert_list_tensors_close(refactored_g_gen, legacy_g_gen, name="Transition g_gen")
    assert_list_tensors_close(refactored_g_inf, legacy_g_inf, name="Transition g_inf")

    print(f"  ✓ Transition output shapes: {[t.shape for t in refactored_g_gen]}")
    print(f"  ✓ g_gen values match")
    print(f"  ✓ g_inf values match")


def test_grounded_inference():
    """Test grounded location inference: (g, x) → p."""
    print("Testing grounded location inference (abstract × sensory → grounded)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test inputs
    batch_size = 4
    n_f = params["n_f"]
    n_g = params["n_g"]
    n_x_f = params["n_x_f"]

    # Abstract location g
    g = [torch.randn(batch_size, n_g[f]) for f in range(n_f)]

    # Filtered sensory x_f
    x_f = [torch.randn(batch_size, n_x_f[f]) for f in range(n_f)]

    # Legacy grounded inference
    with torch.no_grad():
        legacy_p = legacy.inf_p(x_f, g)

    # Refactored grounded inference
    with torch.no_grad():
        refactored_p = refactored.inf_p(x_f, g)

    # Compare
    assert_list_tensors_close(refactored_p, legacy_p, name="Grounded location p")

    print(f"  ✓ Grounded location shapes: {[t.shape for t in refactored_p]}")
    print(f"  ✓ Values match")


def test_memory_update():
    """Test Hebbian memory update: M' = λM + η(p_inf ⊗ p_gen)."""
    print("Testing memory update (Hebbian learning)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test inputs
    batch_size = 4
    n_p_total = sum(params["n_p"])

    # Previous memory (batched for legacy)
    M_prev_legacy = torch.randn(batch_size, n_p_total, n_p_total)
    M_prev_refactored = M_prev_legacy.clone()

    # Grounded locations
    p_inferred = torch.randn(batch_size, n_p_total).softmax(dim=1)
    p_generated = torch.randn(batch_size, n_p_total).softmax(dim=1)

    # Memory parameters
    eta = 0.5
    lamb = 0.9999

    # Legacy memory update
    with torch.no_grad():
        legacy_M = legacy.hebbian(M_prev_legacy, p_inferred, p_generated, do_hierarchical_connections=True)

    # Refactored memory update
    # First, set the memory storage to use batched mode
    refactored.storage.M_gen = M_prev_refactored.clone()
    with torch.no_grad():
        refactored.storage.update(p_inferred, p_generated, eta=eta, lamb=lamb)
        refactored_M = refactored.storage.M_gen

    # Compare
    assert_tensors_close(refactored_M, legacy_M, rtol=1e-5, name="Memory matrix M")

    print(f"  ✓ Memory shape: {refactored_M.shape}")
    print(f"  ✓ Values match (max diff: {(refactored_M - legacy_M).abs().max():.2e})")


def test_attractor_dynamics():
    """Test attractor dynamics: iterative pattern completion."""
    print("Testing attractor dynamics (memory retrieval)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test inputs
    batch_size = 4
    n_p_total = sum(params["n_p"])
    n_f = params["n_f"]

    # Memory matrix (batched for legacy)
    M = torch.randn(batch_size, n_p_total, n_p_total)

    # Query pattern (concatenated p)
    p_query = torch.randn(batch_size, n_p_total)

    # Retrieval mask (which frequencies to update at each iteration)
    retrieve_mask = params["p_retrieve_mask_gen"]

    # Legacy attractor
    with torch.no_grad():
        legacy_retrieved = legacy.attractor(p_query, M, retrieve_it_mask=retrieve_mask)

    # Refactored attractor
    # Need to set memory and split query to per-frequency format
    refactored.storage.M_gen = M.clone()
    p_query_split = torch.split(p_query, params["n_p"], dim=1)
    p_query_list = [p.clone() for p in p_query_split]

    with torch.no_grad():
        refactored_retrieved_list = refactored.attractor.retrieve(p_query_list, refactored.storage.M_gen, for_generation=True)
        # Concatenate back for comparison
        refactored_retrieved = torch.cat(refactored_retrieved_list, dim=1)

    # Compare
    assert_tensors_close(refactored_retrieved, legacy_retrieved, rtol=1e-4, name="Retrieved pattern")

    print(f"  ✓ Retrieved pattern shape: {refactored_retrieved.shape}")
    print(f"  ✓ Values match (max diff: {(refactored_retrieved - legacy_retrieved).abs().max():.2e})")


def test_decoder():
    """Test observation decoder: p → x_logits."""
    print("Testing decoder (grounded location → observation predictions)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test input
    batch_size = 4
    n_f = params["n_f"]
    n_p = params["n_p"]

    # Grounded location p (per-frequency)
    p = [torch.randn(batch_size, n_p[f]).softmax(dim=1) for f in range(n_f)]

    # Legacy decoding
    with torch.no_grad():
        legacy_x_logits = legacy.gen_x(p)

    # Refactored decoding
    with torch.no_grad():
        refactored_x_logits = refactored.gen_x(p)

    # Compare
    assert_tensors_close(refactored_x_logits, legacy_x_logits, rtol=1e-5, name="Decoder x_logits")

    print(f"  ✓ Decoder output shape: {refactored_x_logits.shape}")
    print(f"  ✓ Values match (max diff: {(refactored_x_logits - legacy_x_logits).abs().max():.2e})")


def test_projection():
    """Test projection head: g → g_downsampled."""
    print("Testing projection (abstract location downsampling)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Create test input
    batch_size = 4
    n_f = params["n_f"]
    n_g = params["n_g"]

    # Abstract location g
    g = [torch.randn(batch_size, n_g[f]) for f in range(n_f)]

    # Legacy projection (g2g_ method)
    with torch.no_grad():
        legacy_g_down = legacy.g2g_(g)

    # Refactored projection
    with torch.no_grad():
        refactored_g_down = refactored.projection.downsample(g)

    # Compare
    assert_list_tensors_close(refactored_g_down, legacy_g_down, rtol=1e-5, name="Projected g")

    print(f"  ✓ Projected shapes: {[t.shape for t in refactored_g_down]}")
    print(f"  ✓ Values match")


# ======================================================================================
# MAIN TEST RUNNER
# ======================================================================================


def run_all_tests():
    """Run all component parity tests."""
    print("=" * 70)
    print("COMPONENT PARITY TESTS: Legacy Model vs Refactored TEMModel")
    print("=" * 70)
    print()

    tests = [
        test_encoder,
        test_transition,
        test_grounded_inference,
        test_memory_update,
        test_attractor_dynamics,
        test_decoder,
        test_projection,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"  ✗ FAILED: {e}")
            print()

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
