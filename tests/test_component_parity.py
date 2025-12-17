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
from adapters.legacy_adapter import legacy_to_typed
from torch import Tensor

import parameters
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
    refactored_model = TEMModel(typed_config.architecture)  # Pass only architecture config
    refactored_model.eval()

    # Copy trained parameters from legacy to refactored
    # This ensures we're comparing identical weights
    _copy_weights(legacy_model, refactored_model, legacy_params)

    return legacy_model, refactored_model, legacy_params


def _copy_weights(legacy: LegacyModel, refactored: TEMModel, params: dict):
    """Copy weights from legacy model to refactored model for fair comparison."""
    print("DEBUG: Copying weights...")
    with torch.no_grad():
        # 1. Sensory Processor (w_x, b_x)
        if hasattr(legacy, "w_x") and hasattr(refactored.inference.processor, "w_x"):
            print(f"DEBUG: w_x legacy {legacy.w_x.shape}, refactored {refactored.inference.processor.w_x.shape}")
            # Handle scalar vs vector w_x
            if legacy.w_x.numel() == 1 and refactored.inference.processor.w_x.numel() > 1:
                refactored.inference.processor.w_x.data.fill_(legacy.w_x.item())
            else:
                refactored.inference.processor.w_x.data.copy_(legacy.w_x.data)

            print(f"DEBUG: b_x legacy {legacy.b_x.shape}, refactored {refactored.inference.processor.b_x.shape}")
            # Handle shape mismatch for b_x
            if legacy.b_x.shape != refactored.inference.processor.b_x.shape:
                refactored.inference.processor.b_x.data.copy_(legacy.b_x.data.view_as(refactored.inference.processor.b_x))
            else:
                refactored.inference.processor.b_x.data.copy_(legacy.b_x.data)

        # 2. Transition (MLP_D_a, D_no_a, MLP_sigma_g_path)
        print("DEBUG: Copying MLP_D_a...")
        # MLP_D_a needs input dimension expansion (4D → 5D) for gym-standard actions
        _copy_mlp(legacy.MLP_D_a, refactored.generative.transition.MLP_D_a, expand_input_dim=True, has_static_action=params["has_static_action"])
        print("DEBUG: Copying MLP_sigma_g_path...")
        _copy_mlp(legacy.MLP_sigma_g_path, refactored.generative.transition.MLP_sigma_g_path)
        for i in range(len(legacy.D_no_a)):
            refactored.generative.transition.D_no_a[i].data.copy_(legacy.D_no_a[i].data)

        # 3. Abstract Inference (MLP_mu_g_mem, MLP_sigma_g_mem, g_init)
        print("DEBUG: Copying MLP_mu_g_mem...")
        _copy_mlp(legacy.MLP_mu_g_mem, refactored.inference.abstract.mlp_mu_g_mem)
        print("DEBUG: Copying MLP_sigma_g_mem...")
        _copy_mlp(legacy.MLP_sigma_g_mem, refactored.inference.abstract.mlp_sigma_g_mem)
        for i in range(len(legacy.g_init)):
            refactored.inference.abstract.g_init[i].data.copy_(legacy.g_init[i].data)
            refactored.inference.abstract.logsig_g_init[i].data.copy_(legacy.logsig_g_init[i].data)

        # 4. Grounded Inference (w_p)
        if hasattr(legacy, "w_p"):
            for i in range(len(legacy.w_p)):
                refactored.inference.grounded.w_p[i].data.copy_(legacy.w_p[i].data)

        # Copy W_repeat and W_tile
        if hasattr(legacy, "W_repeat"):
            for f in range(len(legacy.W_repeat)):
                getattr(refactored.inference.grounded, f"W_repeat_{f}").copy_(legacy.W_repeat[f])
                getattr(refactored.inference.grounded, f"W_tile_{f}").copy_(legacy.W_tile[f])

        # 5. Decoder (f_x)
        print("DEBUG: Copying MLP_c_star...")
        _copy_mlp(legacy.MLP_c_star, refactored.generative.decoder.mlp_decoder)

        # 6. Projection (alpha)
        for i in range(len(legacy.alpha)):
            refactored.projection.alpha[i].data.copy_(legacy.alpha[i].data)

        # 7. Attractor Masks
        if hasattr(legacy, "p_retrieve_mask_inf"):
            for i in range(len(legacy.p_retrieve_mask_inf)):
                refactored.attractor.p_retrieve_mask_inf[i].copy_(legacy.p_retrieve_mask_inf[i])
                refactored.attractor.p_retrieve_mask_gen[i].copy_(legacy.p_retrieve_mask_gen[i])


def _expand_action_weights(legacy_weight, has_static_action=True):
    """Expand action MLP weights from legacy (4D) to gym standard (5D).

    Legacy encoding with has_static_action=True:
    - Action 0 → [0,0,0,0] (all zeros, handled by special logic)
    - Action 1-4 → one-hot(0-3) in 4D space

    Gym encoding:
    - Action 0-4 → standard one-hot(0-4) in 5D space

    To maintain equivalence:
    - Prepend zero column (action 0 = "stay" with no learned transition)
    - Keep remaining columns as-is (action 1-4 map to same movements)
    """
    if not has_static_action:
        return legacy_weight

    # Prepend zero column for "stay" action
    zero_column = torch.zeros(legacy_weight.shape[0], 1, device=legacy_weight.device)
    expanded = torch.cat([zero_column, legacy_weight], dim=1)
    return expanded


def _copy_mlp(legacy_mlp, refactored_mlp, expand_input_dim=False, has_static_action=False):
    """Copy weights between MLPs.

    Args:
        expand_input_dim: If True, expand input layer from 4D to 5D (for MLP_D_a)
        has_static_action: Whether legacy used special static action encoding
    """
    # Legacy MLP has w (ModuleList of ModuleLists)
    # Refactored MLP has networks list of Sequentials

    # Check if legacy MLP is list-based (multiple modules)
    is_list = legacy_mlp.is_list
    print(f"DEBUG: _copy_mlp is_list={is_list}")

    if is_list:
        n_modules = legacy_mlp.N
        for i in range(n_modules):
            # Layer 1 (legacy w[i][0]) - may need input dimension expansion
            print(f"DEBUG: Copying module {i} layer 0 weight. Legacy: {legacy_mlp.w[i][0].weight.shape}, Refactored: {refactored_mlp.networks[i][0].weight.shape}")
            legacy_weight = legacy_mlp.w[i][0].weight.data
            if expand_input_dim and has_static_action:
                expanded_weight = _expand_action_weights(legacy_weight, has_static_action)
                refactored_mlp.networks[i][0].weight.data.copy_(expanded_weight)
            else:
                refactored_mlp.networks[i][0].weight.data.copy_(legacy_weight)

            if legacy_mlp.w[i][0].bias is not None:
                refactored_mlp.networks[i][0].bias.data.copy_(legacy_mlp.w[i][0].bias.data)

            # Layer 2 (legacy w[i][1]) - no expansion needed
            if len(refactored_mlp.networks[i]) > 2:
                print(f"DEBUG: Copying module {i} layer 2 weight. Legacy: {legacy_mlp.w[i][1].weight.shape}, Refactored: {refactored_mlp.networks[i][2].weight.shape}")
                refactored_mlp.networks[i][2].weight.data.copy_(legacy_mlp.w[i][1].weight.data)
                if legacy_mlp.w[i][1].bias is not None:
                    refactored_mlp.networks[i][2].bias.data.copy_(legacy_mlp.w[i][1].bias.data)
    else:
        # Single module (legacy w[0])
        print(f"DEBUG: Copying single module layer 0 weight. Legacy: {legacy_mlp.w[0][0].weight.shape}, Refactored: {refactored_mlp.networks[0][0].weight.shape}")
        legacy_weight = legacy_mlp.w[0][0].weight.data
        if expand_input_dim and has_static_action:
            expanded_weight = _expand_action_weights(legacy_weight, has_static_action)
            refactored_mlp.networks[0][0].weight.data.copy_(expanded_weight)
        else:
            refactored_mlp.networks[0][0].weight.data.copy_(legacy_weight)

        if legacy_mlp.w[0][0].bias is not None:
            refactored_mlp.networks[0][0].bias.data.copy_(legacy_mlp.w[0][0].bias.data)

        if len(refactored_mlp.networks[0]) > 2:
            print(f"DEBUG: Copying single module layer 2 weight. Legacy: {legacy_mlp.w[0][1].weight.shape}, Refactored: {refactored_mlp.networks[0][2].weight.shape}")
            refactored_mlp.networks[0][2].weight.data.copy_(legacy_mlp.w[0][1].weight.data)
            if legacy_mlp.w[0][1].bias is not None:
                refactored_mlp.networks[0][2].bias.data.copy_(legacy_mlp.w[0][1].bias.data)


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
    legacy_encoded = legacy.f_c(x_onehot)

    # Refactored encoding
    refactored_encoded = refactored.inference.encoder.forward(x_onehot)

    # Compare
    # Ensure both are float for comparison
    assert_tensors_close(refactored_encoded.float(), legacy_encoded.float(), name="Encoder output")

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

    # Dummy locations for legacy (it iterates over them)
    locations = [{"shiny": None} for _ in range(batch_size)]

    # Legacy transition (expects action list)
    with torch.no_grad():
        legacy_g_gen, legacy_g_inf = legacy.gen_g(actions, g_prev, locations=locations)

    # Refactored transition (same interface)
    with torch.no_grad():
        refactored_g_gen, refactored_g_inf = refactored.generative.gen_g(actions, g_prev, locations=locations)

    # Compare both outputs
    assert_list_tensors_close(refactored_g_gen, legacy_g_gen, name="Transition g_gen")

    # Unpack g_inf (g, sigma_g)
    ref_g, ref_sigma = refactored_g_inf
    leg_g, leg_sigma = legacy_g_inf

    assert_list_tensors_close(ref_g, leg_g, name="Transition g_inf (g)")
    assert_list_tensors_close(ref_sigma, leg_sigma, name="Transition g_inf (sigma)")

    print(f"  ✓ Transition output shapes: {[t.shape for t in refactored_g_gen]}")
    print(f"  ✓ g_gen values match")
    print(f"  ✓ g_inf values match")


def test_grounded_inference():
    """Test grounded location inference: (g, x) → p."""
    print("Testing grounded location inference (abstract × sensory → grounded)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Verify W_repeat and W_tile
    print("  Verifying W_repeat and W_tile...")
    n_f = params["n_f"]
    for f in range(n_f):
        w_rep_leg = legacy.hyper["W_repeat"][f]
        w_rep_ref = getattr(refactored.inference.grounded, f"W_repeat_{f}")
        if not torch.allclose(w_rep_leg, w_rep_ref):
            print(f"  ✗ W_repeat_{f} mismatch! Max diff: {(w_rep_leg - w_rep_ref).abs().max():.2e}")

        w_tile_leg = legacy.hyper["W_tile"][f]
        w_tile_ref = getattr(refactored.inference.grounded, f"W_tile_{f}")
        if not torch.allclose(w_tile_leg, w_tile_ref):
            print(f"  ✗ W_tile_{f} mismatch! Max diff: {(w_tile_leg - w_tile_ref).abs().max():.2e}")

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
        # Legacy inf_p expects expanded inputs (g_, x_)
        # We need to manually prepare them
        legacy_g_ = legacy.g2g_(g)
        # x_f is already filtered, but legacy x2x_ expects filtered x
        # Wait, legacy x2x_ takes x (filtered) and does normalization + expansion
        legacy_x_ = legacy.x2x_(x_f)
        legacy_p = legacy.inf_p(legacy_x_, legacy_g_)

    # Refactored grounded inference
    with torch.no_grad():
        # Prepare inputs for grounded module
        # 1. Downsample g (Legacy g2g_ does f_g(g) which is downsample, no normalization)
        g_down = refactored.projection.downsample(g)

        # 2. Compute grounded location
        # Refactored grounded module takes downsampled g and filtered x
        # It handles expansion internally
        # BUT it expects x to be normalized (lec.Processor does this)
        # Legacy x2x_ does normalization internally.
        # So we must normalize x_f for refactored to match legacy input parity.
        # We use legacy.f_n to ensure exact same normalization.
        x_f_norm = legacy.f_n(x_f)

        refactored_p = refactored.inference.grounded(g_down, x_f_norm)

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
    refactored.storage.batch_size = batch_size
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

    # Split query for legacy (expects list)
    p_query_split = torch.split(p_query, params["n_p"], dim=1)
    p_query_list = [p.clone() for p in p_query_split]

    # Legacy attractor
    with torch.no_grad():
        legacy_retrieved = legacy.attractor(p_query_list, M, retrieve_it_mask=retrieve_mask)

    # Refactored attractor
    # Need to set memory and split query to per-frequency format
    refactored.storage.M_gen = M.clone()

    with torch.no_grad():
        # Refactored retrieve expects concatenated p_query
        refactored_retrieved = refactored.attractor.retrieve(p_query, refactored.storage.M_gen, for_inference=False)

    # Compare
    # Legacy returns list, refactored returns tensor
    # Let's concatenate legacy too
    legacy_retrieved_cat = torch.cat(legacy_retrieved, dim=1)
    assert_tensors_close(refactored_retrieved, legacy_retrieved_cat, rtol=1e-4, name="Retrieved pattern")

    print(f"  ✓ Retrieved pattern shape: {refactored_retrieved.shape}")
    print(f"  ✓ Values match (max diff: {(refactored_retrieved - legacy_retrieved_cat).abs().max():.2e})")


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
        # Legacy gen_x expects p as a tensor (for the first frequency/highest resolution)
        # It uses W_tile[0], so we pass p[0]
        _, legacy_x_logits = legacy.gen_x(p[0])

    # Refactored decoding
    with torch.no_grad():
        prediction = refactored.generative.gen_x(p)
        refactored_x_logits = prediction.logits[0]

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

    # Legacy projection (f_g method for downsampling)
    with torch.no_grad():
        legacy_g_down = legacy.f_g(g)

    # Refactored projection
    with torch.no_grad():
        # Refactored downsample expects normalized g?
        # Legacy f_g takes g directly.
        # Refactored downsample takes g.
        # But test code was doing normalize_g first.
        # Let's check if legacy f_g expects normalized g.
        # Legacy f_g: downsampled = [matmul(g[f], g_downsample[f])...]
        # It doesn't normalize.
        # So we should pass g directly to refactored downsample if we want to match legacy f_g.
        # But refactored downsample might expect normalized g?
        # Refactored projection.downsample just does matmul.
        refactored_g_down = refactored.projection.downsample(g)

    # Compare
    assert_list_tensors_close(refactored_g_down, legacy_g_down, rtol=1e-5, name="Projected g")

    print(f"  ✓ Projected shapes: {[t.shape for t in refactored_g_down]}")
    print(f"  ✓ Values match")


def test_abstract_inference():
    """Test abstract location inference: (p_x, g_gen, x) → g_inf."""
    print("Testing abstract location inference (memory + path integration → abstract)...")

    legacy, refactored, params = create_test_models(batch_size=4)

    # Disable sampling for deterministic comparison
    legacy.hyper["do_sample"] = False
    refactored.config.do_sample = False

    # Create test inputs
    batch_size = 4
    n_f = params["n_f"]
    n_g = params["n_g"]
    n_p = params["n_p"]
    n_x = params["n_x"]

    # Retrieved grounded location p_x (from memory)
    p_x = [torch.randn(batch_size, n_p[f]) for f in range(n_f)]

    # Path integrated abstract location g_gen (mu, sigma)
    # g_gen in model.py is a tuple of lists: (mu_g, sigma_g)
    mu_g_path = [torch.randn(batch_size, n_g[f]) for f in range(n_f)]
    sigma_g_path = [torch.rand(batch_size, n_g[f]) for f in range(n_f)]  # sigma must be positive? model doesn't enforce it but usually is.
    g_gen = (mu_g_path, sigma_g_path)

    # Observation x (one-hot)
    x = torch.zeros(batch_size, n_x)
    x[0, 0] = 1.0
    x[1, 1] = 1.0
    x[2, 2] = 1.0
    x[3, 3] = 1.0

    # Locations (for shiny objects)
    locations = [{"shiny": None} for _ in range(batch_size)]

    # Legacy inference
    with torch.no_grad():
        legacy_g_inf = legacy.inf_g(p_x, g_gen, x, locations)

    # Refactored inference
    with torch.no_grad():
        # Prepare inputs for refactored forward
        # 1. g_downsampled
        W_repeat = [getattr(refactored.inference.grounded, f"W_repeat_{f}") for f in range(n_f)]
        g_downsampled = refactored.projection.inverse_project(p_x, W_repeat)

        # 2. recon_error
        prediction = refactored.generative.gen_x(p_x)
        from torch_tem import utils as tem_utils

        recon_error = tem_utils.squared_error_freq([x], prediction.values)

        # Refactored inference.abstract.forward(p_x, g_gen, x, locations)
        refactored_g_inf = refactored.inference.abstract.forward(transition=g_gen, g_downsampled=g_downsampled, shiny_signals=None, p2g_scale_offset=0.0, recon_error=recon_error)

    # Compare
    assert_list_tensors_close(refactored_g_inf, legacy_g_inf, rtol=1e-5, name="Abstract location g_inf")

    print(f"  ✓ Abstract location shapes: {[t.shape for t in refactored_g_inf]}")
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
        test_abstract_inference,
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
