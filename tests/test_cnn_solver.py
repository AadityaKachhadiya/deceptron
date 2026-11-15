import unittest

import torch

from deceptron import DeceptronCNN3D, SolverConfig, solve_dipg


class CNNShapeTest(unittest.TestCase):
    def test_single_field_matches_a_batch_of_one(self):
        model = DeceptronCNN3D(2, 2, 2, hidden_channels=2)
        field = torch.linspace(0, 1, steps=8)
        for mapping in (model.f, model.g):
            self.assertEqual(mapping(field).shape, field.shape)
            torch.testing.assert_close(mapping(field), mapping(field[None])[0])

    def test_dipg_accepts_a_single_cnn_observation(self):
        model = DeceptronCNN3D(2, 2, 2, hidden_channels=2)
        observation = torch.ones(8)
        result = solve_dipg(
            model, observation, torch.zeros_like(observation),
            SolverConfig(max_iterations=1, tolerance_eps=0),
        )
        self.assertEqual(result["x_hat"].shape, observation.shape)
        self.assertTrue(torch.isfinite(result["x_hat"]).all())


if __name__ == "__main__":
    unittest.main()
