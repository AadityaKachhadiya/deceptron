# Deceptron

Deceptron is a PyTorch package for solving repeated nonlinear inverse
problems with a learned reverse map. It implements D-IPG and the Jacobian
Composition Penalty (JCP).

[A. L. Kachhadiya. Local Inverse Geometry Can Be Amortized. *arXiv preprint* arXiv:2605.13068, 2026](https://doi.org/10.48550/arXiv.2605.13068).

## Installation

The package requires Python 3.9 or later, PyTorch 2.0 or later, and
NumPy 1.24 or later. Install from GitHub with Git and pip.

```bash
python -m pip install git+https://github.com/AadityaKachhadiya/deceptron.git
```

## Quick start

This example uses a known reverse map for the measurement equation `y = 2x`.

```python
import torch
from deceptron import DIPG, DeceptronModule, SolverConfig
model = DeceptronModule(lambda x: 2 * x, lambda y: y / 2)
result = DIPG(model, torch.tensor([0.8]), torch.zeros(1), SolverConfig(tolerance_eps=1e-5))
print(result["x_hat"])  # approximately tensor([0.4000])
```

- [Open in Colab](https://colab.research.google.com/github/AadityaKachhadiya/deceptron/blob/main/examples/quickstart.ipynb) to train a small reverse map and view a Heat-3D reconstruction.
- [Train with your own data](user_guide.md#training-with-your-own-data) using your own maps or paired states and measurements.
- [Reproduce the paper](reproduce/README.md) to run the seven PDE benchmarks, checkpoint replays, and figure generation.
