import unittest

import torch

from torch_tem.types import Transition
from torch_tem.utils import inv_var_trans, inv_var_weight


class TestUtilsFusion(unittest.TestCase):
    def test_inv_var_weight_none_sigma_is_perfect_precision(self) -> None:
        mu1 = torch.tensor([[1.0, 2.0]])
        mu2 = torch.tensor([[10.0, 20.0]])
        sigma1 = torch.ones_like(mu1)

        mu, sigma = inv_var_weight([mu1, mu2], [sigma1, None])

        self.assertTrue(torch.allclose(mu, mu2))
        self.assertTrue(torch.allclose(sigma, torch.zeros_like(mu1)))

    def test_inv_var_trans_base_uncertainty_none_updates_mean_only(self) -> None:
        base_mean = torch.zeros((3, 2))
        base = Transition(mean=[base_mean], uncertainty=None)

        mask = torch.tensor([True, False, True])
        corr_mean = torch.tensor([[5.0, 5.0], [6.0, 6.0]])
        corr = Transition(mean=[corr_mean], uncertainty=None)

        out = inv_var_trans(base, corr, mask=mask, freqs=range(1))

        expected = base_mean.clone()
        expected[mask] = corr_mean
        self.assertIsNone(out.uncertainty)
        self.assertTrue(torch.allclose(out.mean[0], expected))

    def test_inv_var_trans_corr_uncertainty_none_drives_sigma_to_zero(self) -> None:
        base_mean = torch.zeros((3, 2))
        base_sigma = torch.ones((3, 2))
        base = Transition(mean=[base_mean], uncertainty=[base_sigma])

        mask = torch.tensor([True, False, True])
        corr_mean = torch.tensor([[5.0, 5.0], [6.0, 6.0]])
        corr = Transition(mean=[corr_mean], uncertainty=None)

        out = inv_var_trans(base, corr, mask=mask, freqs=range(1))

        self.assertIsNotNone(out.uncertainty)
        self.assertTrue(torch.allclose(out.mean[0][mask], corr_mean))
        self.assertTrue(torch.allclose(out.uncertainty[0][mask], torch.zeros_like(corr_mean)))


if __name__ == "__main__":
    unittest.main()
