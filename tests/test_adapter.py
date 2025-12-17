#!/usr/bin/env python3
"""Test script to verify the adapter works correctly.

This script tests:
1. Can import the adapter
2. Can create a model with the adapter
3. Can run a forward pass
4. Can compute losses
5. Can update parameters via .hyper dict
"""

import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
import torch

# Test 1: Import the simple adapter
print("=" * 70)
print("TEST 1: Import simple adapter")
print("=" * 70)

try:
    # Import from tests/adapters directory
    adapters_path = os.path.join(project_root, "tests", "adapters")
    sys.path.insert(0, adapters_path)
    import simple_adapter as model

    print("✅ Successfully imported simple_adapter")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

# Test 2: Import parameters and create model
print("\n" + "=" * 70)
print("TEST 2: Create model with adapter")
print("=" * 70)

try:
    import parameters

    params = parameters.parameters()
    tem = model.Model(params)
    print(f"✅ Successfully created model")
    print(f"   Model type: {type(tem).__name__}")
    print(f"   Has .hyper: {hasattr(tem, 'hyper')}")
    print(f"   Batch size: {params['batch_size']}")
except Exception as e:
    print(f"❌ Failed to create model: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 3: Create test walk and run forward pass
print("\n" + "=" * 70)
print("TEST 3: Run forward pass")
print("=" * 70)

try:
    import world

    # Create single environment
    env = world.World("./envs/5x5.json", randomise_observations=True)

    # Generate short walk
    walk_steps = 5
    batch_size = params["batch_size"]

    # Create environments for batch
    environments = [world.World("./envs/5x5.json", randomise_observations=True) for _ in range(batch_size)]

    # Generate walks
    walks = [env.generate_walks(walk_steps, 1)[0] for env in environments]

    # Format walk for model
    chunk = []
    for step_idx in range(walk_steps):
        locations = [walks[env_idx][step_idx][0] for env_idx in range(batch_size)]
        observations = [walks[env_idx][step_idx][1] for env_idx in range(batch_size)]
        actions = [walks[env_idx][step_idx][2] for env_idx in range(batch_size)]
        observations = torch.stack(observations, dim=0)
        chunk.append([locations, observations, actions])

    # Run forward pass
    with torch.no_grad():
        forward = tem(chunk, prev_iter=None)

    print(f"✅ Successfully ran forward pass")
    print(f"   Steps returned: {len(forward)}")
    print(f"   First step type: {type(forward[0]).__name__}")
    print(f"   Has .L (losses): {hasattr(forward[0], 'L')}")
    print(f"   Has .M (memory): {hasattr(forward[0], 'M')}")
    print(f"   Has .correct(): {hasattr(forward[0], 'correct')}")

except Exception as e:
    print(f"❌ Failed forward pass: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 4: Compute losses
print("\n" + "=" * 70)
print("TEST 4: Compute losses")
print("=" * 70)

try:
    loss_weights = params["loss_weights"]

    loss = torch.tensor(0.0)
    for step in forward:
        step_loss = loss_weights * torch.stack([l[0] for l in step.L])
        loss = loss + torch.sum(step_loss)

    print(f"✅ Successfully computed losses")
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Loss components per step: {len(forward[0].L)}")

except Exception as e:
    print(f"❌ Failed loss computation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 5: Update hyperparameters
print("\n" + "=" * 70)
print("TEST 5: Update hyperparameters via .hyper")
print("=" * 70)

try:
    old_eta = tem.hyper["eta"]
    old_lambda = tem.hyper["lambda"]

    # Update parameters
    tem.hyper["eta"] = 0.5
    tem.hyper["lambda"] = 0.999

    print(f"✅ Successfully updated hyperparameters")
    print(f"   Old eta: {old_eta:.4f} → New eta: {tem.hyper['eta']:.4f}")
    print(f"   Old lambda: {old_lambda:.6f} → New lambda: {tem.hyper['lambda']:.6f}")

except Exception as e:
    print(f"❌ Failed hyperparameter update: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test 6: Compute accuracy
print("\n" + "=" * 70)
print("TEST 6: Compute accuracy")
print("=" * 70)

try:
    acc_p, acc_g, acc_gt = np.mean([[np.mean(a) for a in step.correct()] for step in forward], axis=0)
    acc_p, acc_g, acc_gt = [a * 100 for a in (acc_p, acc_g, acc_gt)]

    print(f"✅ Successfully computed accuracy")
    print(f"   Accuracy <p>: {acc_p:.2f}%")
    print(f"   Accuracy <g>: {acc_g:.2f}%")
    print(f"   Accuracy <gt>: {acc_gt:.2f}%")

except Exception as e:
    print(f"❌ Failed accuracy computation: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# All tests passed!
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nThe simple adapter is working correctly.")
print("You can now use it in run.py by changing the import:")
print("  from tests.adapters import simple_adapter as model")
