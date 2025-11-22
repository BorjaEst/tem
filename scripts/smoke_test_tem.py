"""Basic smoke test for TEMModel to verify implementation integrity.

This script tests:
1. Model instantiation with default configuration
2. Single iteration with synthetic data
3. Full forward pass with random walk
4. Tensor shape correctness throughout pipeline
5. State class validation and properties
"""

import torch

from torch_tem.config import EnvironmentConfig, InferenceConfig, ModelConfig
from torch_tem.core import State
from torch_tem.model import TEMModel


def test_instantiation():
    """Test that TEMModel can be instantiated with default config."""
    print("Testing instantiation...")

    # Create configs
    arch_config = ModelConfig()
    inf_config = InferenceConfig()
    env_config = EnvironmentConfig()

    # Create merged architecture config with environment fields
    # TransitionModel needs n_actions from environment and do_sample from inference
    # We create a merged object that has all fields from all three configs
    class MergedArchConfig:
        def __init__(self, arch, env, inf):
            # Copy all fields from architecture config
            for attr in dir(arch):
                if not attr.startswith("_") and attr != "model_config":
                    setattr(self, attr, getattr(arch, attr))
            # Add environment fields needed by components
            self.n_actions = env.n_actions
            self.has_static_action = env.has_static_action
            # Add inference fields needed by components
            self.do_sample = inf.do_sample

    merged_arch = MergedArchConfig(arch_config, env_config, inf_config)

    # Create params wrapper with merged architecture and original inference
    class Params:
        def __init__(self, architecture, inference):
            self.architecture = architecture
            self.inference = inference

    params = Params(merged_arch, inf_config)
    model = TEMModel(params)

    print(f"✓ Model instantiated successfully")
    print(f"  - Architecture: n_f={arch_config.n_f}, n_g={arch_config.n_g}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, merged_arch, inf_config


def test_single_iteration(model, arch_config, inf_config):
    """Test single iteration with synthetic data."""
    print("\nTesting single iteration...")

    batch_size = 4

    # Create synthetic inputs
    x = torch.randn(batch_size, arch_config.n_x).softmax(dim=1)  # One-hot-like observations
    a_prev = torch.randint(0, 4, (batch_size,)).tolist()  # Actions
    locations = [{"shiny": None} for _ in range(batch_size)]

    # Initialize states - use None for M_prev to let model initialize it
    M_prev = None

    x_prev = [torch.zeros(batch_size, n_x_f) for n_x_f in arch_config.n_x_f]
    g_prev = [torch.randn(batch_size, n_g) for n_g in arch_config.n_g]

    # Create initial state
    state_0 = model.init_state(locations, x_prev, a_prev, M_prev)

    # Run single iteration using state values
    L, M, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf = model.iteration(x, locations, state_0.a, state_0.M, state_0.x_inf, state_0.g_inf)

    # Verify outputs
    assert len(L) == 8, f"Expected 8 loss components, got {len(L)}"
    assert len(M) == len(state_0.M), f"Memory count mismatch: {len(M)} != {len(state_0.M)}"
    assert len(g_inf) == arch_config.n_f, f"g_inf frequency mismatch"
    assert len(p_inf) == arch_config.n_f, f"p_inf frequency mismatch"

    print(f"✓ Single iteration completed")
    print(f"  - Losses: {[f'{l.mean().item():.4f}' for l in L]}")
    print(f"  - Memory matrices: {len(M)}")
    print(f"  - g_inf shapes: {[g.shape for g in g_inf]}")


def test_forward_pass(model, arch_config):
    """Test full forward pass with random walk."""
    print("\nTesting forward pass...")

    batch_size = 2
    n_steps = 5

    # Create synthetic walk
    walk = []
    for t in range(n_steps):
        locations = [{"shiny": None} for _ in range(batch_size)]
        x = torch.randn(batch_size, arch_config.n_x).softmax(dim=1)
        a = torch.randint(0, 4, (batch_size,)).tolist()
        walk.append((locations, x, a))

    # Run forward pass
    steps = model.forward(walk, prev_M=None)

    # Verify outputs
    assert len(steps) == n_steps, f"Expected {n_steps} steps, got {len(steps)}"
    assert all(isinstance(step, State) for step in steps), "All steps should be State objects"

    print(f"✓ Forward pass completed")
    print(f"  - Steps processed: {len(steps)}")
    print(f"  - State fields: g, x, a, L, M, locations, g_gen, p_gen, x_gen, x_logits, x_inf, g_inf, p_inf")


def test_state_properties(model, arch_config):
    """Test State class properties and validation."""
    print("\nTesting State class properties...")

    batch_size = 3

    # Create synthetic walk with one step
    locations = [{"shiny": None} for _ in range(batch_size)]
    x = torch.randn(batch_size, arch_config.n_x).softmax(dim=1)
    a = torch.randint(0, 4, (batch_size,)).tolist()
    walk = [(locations, x, a)]

    # Run forward pass to get a State
    steps = model.forward(walk, prev_M=None)
    state = steps[0]

    # Test properties
    assert state.batch_size == batch_size, f"batch_size property incorrect: {state.batch_size} != {batch_size}"
    assert state.n_freqs == arch_config.n_f, f"n_freqs property incorrect: {state.n_freqs} != {arch_config.n_f}"

    # Test detach method
    detached = state.detach()
    assert isinstance(detached, State), "detach() should return State instance"
    assert detached is not state, "detach() should return new instance"
    assert detached.batch_size == state.batch_size, "Detached state should preserve batch_size"

    # Test correct method
    new_g = [torch.randn(batch_size, n_g) for n_g in arch_config.n_g]
    new_p = [torch.randn(batch_size, n_p) for n_p in arch_config.n_p]
    state.correct(new_g, new_p)
    assert state.g_inf is new_g, "correct() should update g_inf"
    assert state.p_inf is new_p, "correct() should update p_inf"

    print(f"✓ State properties validated")
    print(f"  - batch_size: {state.batch_size}")
    print(f"  - n_freqs: {state.n_freqs}")
    print(f"  - detach() returns new instance: {detached is not state}")
    print(f"  - correct() mutates in place: True")


def test_tensor_shapes(model, arch_config):
    """Verify tensor shapes throughout the pipeline."""
    print("\nVerifying tensor shapes...")

    batch_size = 3

    # Test encoding
    x = torch.randn(batch_size, arch_config.n_x).softmax(dim=1)
    x_c = model.encoder(x)
    assert x_c.shape == (batch_size, arch_config.n_x_c), f"Encoder output shape mismatch"

    # Test processor
    x_prev = [torch.zeros(batch_size, n_x_f) for n_x_f in arch_config.n_x_f]
    x_f = model.processor(x_c, x_prev)
    assert len(x_f) == arch_config.n_f, f"Processor output frequency mismatch"

    # Test transition
    g_prev = [torch.randn(batch_size, n_g) for n_g in arch_config.n_g]
    a_prev = torch.randint(0, 4, (batch_size,)).tolist()
    locations = [{"shiny": None} for _ in range(batch_size)]
    g_gen, (g_mu, sigma_g) = model.gen_g(a_prev, g_prev, locations)
    assert len(g_gen) == arch_config.n_f, f"Transition output frequency mismatch"

    # Test projection
    g_normalized = model.projection.normalize_g(g_prev)
    g_down = model.projection.downsample(g_normalized)
    assert len(g_down) == arch_config.n_f, f"Projection output frequency mismatch"

    # Test grounded inference
    x_tiled = model.tiling(x_f)
    p = model.grounded(g_down, x_f)
    assert len(p) == arch_config.n_f, f"Grounded inference output frequency mismatch"
    for f, p_f in enumerate(p):
        expected_shape = (batch_size, arch_config.n_p[f])
        assert p_f.shape == expected_shape, f"p[{f}] shape {p_f.shape} != {expected_shape}"

    # Test decoder
    x_recon, x_logits = model.decoder(p)
    assert x_recon.shape == (batch_size, arch_config.n_x), f"Decoder output shape mismatch"

    print(f"✓ All tensor shapes verified")
    print(f"  - Encoding: {x.shape} → {x_c.shape}")
    print(f"  - Filtering: {arch_config.n_x_c} → {[x_f_i.shape[1] for x_f_i in x_f]}")
    print(f"  - Transition: {[g.shape for g in g_prev]} → {[g.shape for g in g_gen]}")
    print(f"  - Grounded: {[p_f.shape for p_f in p]}")
    print(f"  - Decoding: {[p_f.shape for p_f in p]} → {x_recon.shape}")


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("TEM Model Smoke Test")
    print("=" * 60)

    try:
        # Test 1: Instantiation
        model, arch_config, inf_config = test_instantiation()

        # Test 2: Single iteration
        test_single_iteration(model, arch_config, inf_config)

        # Test 3: Forward pass
        test_forward_pass(model, arch_config)

        # Test 4: State properties
        test_state_properties(model, arch_config)

        # Test 5: Tensor shapes
        test_tensor_shapes(model, arch_config)

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == "__main__":
    main()
