import unittest

import torch

from deceptron import DIPG, DeceptronModule, SolverConfig, estimate_rjcp_dataset


class CustomModelTest(unittest.TestCase):
    def test_stateless_maps_with_different_state_and_observation_dimensions(self):
        def forward(x):
            return torch.cat((x, x.sum(dim=-1, keepdim=True)), dim=-1)

        model = DeceptronModule(forward, lambda y: y[..., :2])
        truth = torch.tensor([0.2, 0.6], dtype=torch.float64)
        result = DIPG(model, model(truth), torch.zeros_like(truth),
                      SolverConfig(rho=1, tolerance_eps=1e-10))

        self.assertTrue(result["success"])
        self.assertEqual(result["final_rjcp"], 0.0)
        self.assertEqual(estimate_rjcp_dataset(model, truth.unsqueeze(0)), 0.0)
        torch.testing.assert_close(result["x_hat"], truth)
