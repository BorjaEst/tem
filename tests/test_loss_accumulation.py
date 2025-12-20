"""Unit tests for loss accumulation utilities.

Tests the LossAccumulator dataclass and utility functions (init_loss_accumulator,
accumulate_loss, finalize_loss_accumulator) for correctness, edge cases, and
type safety.
"""

import pytest
import torch

from torch_tem.losses import LossAccumulator, LossOutput, accumulate_loss, finalize_loss_accumulator, init_loss_accumulator


class TestLossAccumulator:
    """Test suite for LossAccumulator dataclass."""

    def test_initialization_defaults(self):
        """LossAccumulator should initialize with all zeros."""
        acc = LossAccumulator()

        assert acc.total == 0.0
        assert acc.lx == 0.0
        assert acc.lg == 0.0
        assert acc.lp == 0.0
        assert acc.l_reg_g == 0.0
        assert acc.l_reg_p == 0.0
        assert acc.n_steps == 0

    def test_initialization_custom_values(self):
        """LossAccumulator should accept custom initialization values."""
        acc = LossAccumulator(
            total=10.5,
            lx=3.2,
            lg=2.1,
            lp=4.8,
            l_reg_g=0.2,
            l_reg_p=0.1,
            n_steps=5,
        )

        assert acc.total == 10.5
        assert acc.lx == 3.2
        assert acc.lg == 2.1
        assert acc.lp == 4.8
        assert acc.l_reg_g == 0.2
        assert acc.l_reg_p == 0.1
        assert acc.n_steps == 5

    def test_mutable_accumulator(self):
        """LossAccumulator should be mutable for in-place updates."""
        acc = LossAccumulator()
        acc.total += 5.0
        acc.n_steps += 1

        assert acc.total == 5.0
        assert acc.n_steps == 1


class TestInitLossAccumulator:
    """Test suite for init_loss_accumulator factory function."""

    def test_returns_zero_accumulator(self):
        """init_loss_accumulator should return a zero-initialized accumulator."""
        acc = init_loss_accumulator()

        assert isinstance(acc, LossAccumulator)
        assert acc.total == 0.0
        assert acc.lx == 0.0
        assert acc.lg == 0.0
        assert acc.lp == 0.0
        assert acc.l_reg_g == 0.0
        assert acc.l_reg_p == 0.0
        assert acc.n_steps == 0

    def test_independent_instances(self):
        """Each call should return independent accumulator instances."""
        acc1 = init_loss_accumulator()
        acc2 = init_loss_accumulator()

        acc1.total += 10.0
        assert acc2.total == 0.0  # acc2 should be unaffected


class TestAccumulateLoss:
    """Test suite for accumulate_loss function."""

    def test_accumulate_single_loss_without_regularization(self):
        """Accumulate a single LossOutput without regularization terms."""
        acc = init_loss_accumulator()
        loss = LossOutput(
            total=torch.tensor(10.0),
            lx=torch.tensor(4.0),
            lg=torch.tensor(3.0),
            lp=torch.tensor(3.0),
            l_reg_g=None,
            l_reg_p=None,
        )

        result = accumulate_loss(acc, loss)

        assert result is acc  # Returns same instance (in-place update)
        assert acc.total == 10.0
        assert acc.lx == 4.0
        assert acc.lg == 3.0
        assert acc.lp == 3.0
        assert acc.l_reg_g == 0.0  # Should remain 0 when None
        assert acc.l_reg_p == 0.0  # Should remain 0 when None
        assert acc.n_steps == 1

    def test_accumulate_single_loss_with_regularization(self):
        """Accumulate a single LossOutput with regularization terms."""
        acc = init_loss_accumulator()
        loss = LossOutput(
            total=torch.tensor(12.5),
            lx=torch.tensor(4.0),
            lg=torch.tensor(3.0),
            lp=torch.tensor(3.0),
            l_reg_g=torch.tensor(1.5),
            l_reg_p=torch.tensor(1.0),
        )

        accumulate_loss(acc, loss)

        assert acc.total == 12.5
        assert acc.lx == 4.0
        assert acc.lg == 3.0
        assert acc.lp == 3.0
        assert acc.l_reg_g == 1.5
        assert acc.l_reg_p == 1.0
        assert acc.n_steps == 1

    def test_accumulate_multiple_losses(self):
        """Accumulate multiple LossOutput instances sequentially."""
        acc = init_loss_accumulator()

        loss1 = LossOutput(
            total=torch.tensor(10.0),
            lx=torch.tensor(4.0),
            lg=torch.tensor(3.0),
            lp=torch.tensor(3.0),
            l_reg_g=None,
            l_reg_p=None,
        )

        loss2 = LossOutput(
            total=torch.tensor(15.0),
            lx=torch.tensor(6.0),
            lg=torch.tensor(4.5),
            lp=torch.tensor(4.5),
            l_reg_g=torch.tensor(2.0),
            l_reg_p=torch.tensor(1.5),
        )

        loss3 = LossOutput(
            total=torch.tensor(12.0),
            lx=torch.tensor(5.0),
            lg=torch.tensor(3.5),
            lp=torch.tensor(3.5),
            l_reg_g=torch.tensor(1.0),
            l_reg_p=None,
        )

        accumulate_loss(acc, loss1)
        accumulate_loss(acc, loss2)
        accumulate_loss(acc, loss3)

        assert acc.total == pytest.approx(37.0)
        assert acc.lx == pytest.approx(15.0)
        assert acc.lg == pytest.approx(11.0)
        assert acc.lp == pytest.approx(11.0)
        assert acc.l_reg_g == pytest.approx(3.0)  # 0 + 2.0 + 1.0
        assert acc.l_reg_p == pytest.approx(1.5)  # 0 + 1.5 + 0
        assert acc.n_steps == 3

    def test_accumulate_in_loop(self):
        """Accumulate losses in a typical training loop pattern."""
        acc = init_loss_accumulator()
        n_timesteps = 10

        for t in range(n_timesteps):
            loss = LossOutput(
                total=torch.tensor(1.0 * (t + 1)),
                lx=torch.tensor(0.3 * (t + 1)),
                lg=torch.tensor(0.3 * (t + 1)),
                lp=torch.tensor(0.4 * (t + 1)),
                l_reg_g=torch.tensor(0.05) if t % 2 == 0 else None,
                l_reg_p=torch.tensor(0.03) if t % 3 == 0 else None,
            )
            accumulate_loss(acc, loss)

        # Sum from 1 to 10 = 55
        assert acc.total == pytest.approx(55.0)
        assert acc.lx == pytest.approx(16.5)  # 0.3 * 55
        assert acc.lg == pytest.approx(16.5)  # 0.3 * 55
        assert acc.lp == pytest.approx(22.0)  # 0.4 * 55
        assert acc.l_reg_g == pytest.approx(0.25)  # 0.05 * 5 (t = 0, 2, 4, 6, 8)
        assert acc.l_reg_p == pytest.approx(0.12)  # 0.03 * 4 (t = 0, 3, 6, 9)
        assert acc.n_steps == 10

    def test_handles_scalar_tensors(self):
        """Accumulate should handle 0-dimensional tensors correctly."""
        acc = init_loss_accumulator()
        loss = LossOutput(
            total=torch.tensor(5.0),
            lx=torch.tensor(2.0),
            lg=torch.tensor(1.5),
            lp=torch.tensor(1.5),
        )

        accumulate_loss(acc, loss)

        assert isinstance(acc.total, float)
        assert isinstance(acc.lx, float)


class TestFinalizeLossAccumulator:
    """Test suite for finalize_loss_accumulator function."""

    def test_finalize_single_timestep(self):
        """Finalize accumulator with single timestep (average = value)."""
        acc = init_loss_accumulator()
        loss = LossOutput(
            total=torch.tensor(10.0),
            lx=torch.tensor(4.0),
            lg=torch.tensor(3.0),
            lp=torch.tensor(3.0),
            l_reg_g=torch.tensor(1.5),
            l_reg_p=torch.tensor(1.0),
        )
        accumulate_loss(acc, loss)

        avg_loss, components = finalize_loss_accumulator(acc)

        assert avg_loss == pytest.approx(10.0)
        assert components["lx"] == pytest.approx(4.0)
        assert components["lg"] == pytest.approx(3.0)
        assert components["lp"] == pytest.approx(3.0)
        assert components["l_reg_g"] == pytest.approx(1.5)
        assert components["l_reg_p"] == pytest.approx(1.0)

    def test_finalize_multiple_timesteps(self):
        """Finalize accumulator with multiple timesteps computes correct averages."""
        acc = init_loss_accumulator()

        for _ in range(5):
            loss = LossOutput(
                total=torch.tensor(10.0),
                lx=torch.tensor(4.0),
                lg=torch.tensor(3.0),
                lp=torch.tensor(3.0),
                l_reg_g=torch.tensor(1.0),
                l_reg_p=torch.tensor(0.5),
            )
            accumulate_loss(acc, loss)

        avg_loss, components = finalize_loss_accumulator(acc)

        # All should average to same values (constant input)
        assert avg_loss == pytest.approx(10.0)
        assert components["lx"] == pytest.approx(4.0)
        assert components["lg"] == pytest.approx(3.0)
        assert components["lp"] == pytest.approx(3.0)
        assert components["l_reg_g"] == pytest.approx(1.0)
        assert components["l_reg_p"] == pytest.approx(0.5)

    def test_finalize_varying_values(self):
        """Finalize should compute correct averages for varying loss values."""
        acc = init_loss_accumulator()

        losses = [
            LossOutput(
                total=torch.tensor(10.0),
                lx=torch.tensor(4.0),
                lg=torch.tensor(3.0),
                lp=torch.tensor(3.0),
            ),
            LossOutput(
                total=torch.tensor(20.0),
                lx=torch.tensor(8.0),
                lg=torch.tensor(6.0),
                lp=torch.tensor(6.0),
            ),
            LossOutput(
                total=torch.tensor(15.0),
                lx=torch.tensor(6.0),
                lg=torch.tensor(4.5),
                lp=torch.tensor(4.5),
            ),
        ]

        for loss in losses:
            accumulate_loss(acc, loss)

        avg_loss, components = finalize_loss_accumulator(acc)

        # Average of [10, 20, 15] = 15.0
        assert avg_loss == pytest.approx(15.0)
        # Average of [4, 8, 6] = 6.0
        assert components["lx"] == pytest.approx(6.0)
        # Average of [3, 6, 4.5] = 4.5
        assert components["lg"] == pytest.approx(4.5)
        assert components["lp"] == pytest.approx(4.5)
        # No regularization → averages to 0
        assert components["l_reg_g"] == pytest.approx(0.0)
        assert components["l_reg_p"] == pytest.approx(0.0)

    def test_finalize_with_partial_regularization(self):
        """Finalize should handle cases where regularization is sometimes None."""
        acc = init_loss_accumulator()

        # First loss: no regularization
        accumulate_loss(
            acc,
            LossOutput(
                total=torch.tensor(10.0),
                lx=torch.tensor(4.0),
                lg=torch.tensor(3.0),
                lp=torch.tensor(3.0),
                l_reg_g=None,
                l_reg_p=None,
            ),
        )

        # Second loss: with regularization
        accumulate_loss(
            acc,
            LossOutput(
                total=torch.tensor(12.0),
                lx=torch.tensor(4.0),
                lg=torch.tensor(3.0),
                lp=torch.tensor(3.0),
                l_reg_g=torch.tensor(2.0),
                l_reg_p=torch.tensor(1.0),
            ),
        )

        avg_loss, components = finalize_loss_accumulator(acc)

        # Average total: (10 + 12) / 2 = 11.0
        assert avg_loss == pytest.approx(11.0)
        # Average regularization: (0 + 2.0) / 2 = 1.0
        assert components["l_reg_g"] == pytest.approx(1.0)
        assert components["l_reg_p"] == pytest.approx(0.5)

    def test_finalize_zero_timesteps_raises_error(self):
        """Finalize should raise ValueError when n_steps is 0."""
        acc = init_loss_accumulator()

        with pytest.raises(ValueError, match="Cannot finalize accumulator with zero timesteps"):
            finalize_loss_accumulator(acc)

    def test_components_dict_structure(self):
        """Finalize should return dictionary with all expected component keys."""
        acc = init_loss_accumulator()
        accumulate_loss(
            acc,
            LossOutput(
                total=torch.tensor(10.0),
                lx=torch.tensor(4.0),
                lg=torch.tensor(3.0),
                lp=torch.tensor(3.0),
            ),
        )

        avg_loss, components = finalize_loss_accumulator(acc)

        # Check return types
        assert isinstance(avg_loss, float)
        assert isinstance(components, dict)

        # Check all expected keys are present
        expected_keys = {"lx", "lg", "lp", "l_reg_g", "l_reg_p"}
        assert set(components.keys()) == expected_keys

        # Check all values are floats
        for key, value in components.items():
            assert isinstance(value, float), f"Component {key} is not a float: {type(value)}"


class TestIntegrationPatterns:
    """Integration tests for typical usage patterns in training loops."""

    def test_full_workflow_training_loop(self):
        """Test complete accumulation workflow as used in training."""
        # Initialize
        acc = init_loss_accumulator()

        # Simulate rollout (e.g., 10 timesteps)
        rollout_length = 10
        for t in range(rollout_length):
            # Simulate model forward pass producing loss
            loss_output = LossOutput(
                total=torch.tensor(5.0 + 0.1 * t),
                lx=torch.tensor(2.0 + 0.05 * t),
                lg=torch.tensor(1.5 + 0.025 * t),
                lp=torch.tensor(1.5 + 0.025 * t),
                l_reg_g=torch.tensor(0.1) if t < 5 else None,
                l_reg_p=torch.tensor(0.05),
            )

            # Accumulate
            acc = accumulate_loss(acc, loss_output)

        # Finalize
        avg_loss, components = finalize_loss_accumulator(acc)

        # Verify correctness
        assert acc.n_steps == rollout_length
        assert isinstance(avg_loss, float)
        assert avg_loss > 0.0

        # All components should be present
        for key in ["lx", "lg", "lp", "l_reg_g", "l_reg_p"]:
            assert key in components
            assert isinstance(components[key], float)

    def test_reusable_across_batches(self):
        """Test that accumulators can be reinitialized for each batch."""
        results = []

        for batch_idx in range(3):
            # Fresh accumulator per batch
            acc = init_loss_accumulator()

            # Different number of timesteps per batch
            n_steps = 5 + batch_idx * 2

            for t in range(n_steps):
                loss = LossOutput(
                    total=torch.tensor(10.0),
                    lx=torch.tensor(4.0),
                    lg=torch.tensor(3.0),
                    lp=torch.tensor(3.0),
                )
                accumulate_loss(acc, loss)

            avg_loss, components = finalize_loss_accumulator(acc)
            results.append(avg_loss)

        # All batches should have same average (constant loss)
        assert all(abs(r - 10.0) < 1e-6 for r in results)

    def test_backward_compatibility_with_dict_pattern(self):
        """Ensure new pattern produces same results as old dictionary pattern."""
        # Old pattern (manual dict)
        old_accumulated_loss = 0.0
        old_accumulated_components = {"lx": 0.0, "lg": 0.0, "lp": 0.0, "l_reg_g": 0.0, "l_reg_p": 0.0}

        # New pattern (LossAccumulator)
        new_acc = init_loss_accumulator()

        # Apply same losses
        losses = [
            LossOutput(
                total=torch.tensor(10.5),
                lx=torch.tensor(4.2),
                lg=torch.tensor(3.1),
                lp=torch.tensor(3.2),
                l_reg_g=torch.tensor(0.8),
                l_reg_p=None,
            ),
            LossOutput(
                total=torch.tensor(12.3),
                lx=torch.tensor(5.0),
                lg=torch.tensor(3.6),
                lp=torch.tensor(3.7),
                l_reg_g=None,
                l_reg_p=torch.tensor(0.5),
            ),
        ]

        for loss in losses:
            # Old pattern
            old_accumulated_loss += loss.total.item()
            old_accumulated_components["lx"] += loss.lx.item()
            old_accumulated_components["lg"] += loss.lg.item()
            old_accumulated_components["lp"] += loss.lp.item()
            if loss.l_reg_g is not None:
                old_accumulated_components["l_reg_g"] += loss.l_reg_g.item()
            if loss.l_reg_p is not None:
                old_accumulated_components["l_reg_p"] += loss.l_reg_p.item()

            # New pattern
            accumulate_loss(new_acc, loss)

        # Old averaging
        n_steps = len(losses)
        old_avg_loss = old_accumulated_loss / n_steps
        old_components = {k: v / n_steps for k, v in old_accumulated_components.items()}

        # New averaging
        new_avg_loss, new_components = finalize_loss_accumulator(new_acc)

        # Results should match
        assert old_avg_loss == pytest.approx(new_avg_loss)
        for key in old_components:
            assert old_components[key] == pytest.approx(new_components[key]), f"Mismatch in component: {key}"
