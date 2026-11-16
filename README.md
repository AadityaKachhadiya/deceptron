# Deceptron

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AadityaKachhadiya/deceptron/blob/main/examples/quickstart.ipynb)
[![Documentation](https://app.readthedocs.org/projects/deceptron/badge/?version=latest)](https://deceptron.readthedocs.io/en/latest/)
[![Python 3.9 or later](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](#system-requirements)
[![PyTorch 2.0 or later](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](#system-requirements)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deceptron is a PyTorch package for solving repeated nonlinear inverse problems with a learned reverse map.

[A. L. Kachhadiya. Local Inverse Geometry Can Be Amortized. arXiv:2605.13068, 2026](https://doi.org/10.48550/arXiv.2605.13068).

## System requirements

- Python 3.9 or later, PyTorch 2.0 or later, and NumPy 1.24 or later.
- Git for installation from GitHub. The install command resolves the Python dependencies.
- A CPU is sufficient for the quick start and checkpoint replays. Use a CUDA-enabled PyTorch environment for GPU benchmark training.

## Installation

Install the latest release from PyPI.

```bash
python -m pip install deceptron
```

To install the latest code from the GitHub `main` branch, run:

```bash
python -m pip install git+https://github.com/AadityaKachhadiya/deceptron.git
```

## Quick start

This example calls D-IPG with an exact reverse map for the measurement equation `y = 2x`.

```python
import torch
from deceptron import DIPG, DeceptronModule, SolverConfig
model = DeceptronModule(lambda x: 2 * x, lambda y: y / 2)
result = DIPG(model, torch.tensor([0.8]), torch.zeros(1), SolverConfig(tolerance_eps=1e-5))
print(result["x_hat"])  # approximately tensor([0.4000])
```

- [Run the quick-start notebook](examples/quickstart.ipynb) to train a reverse map for a custom measurement equation and display a pretrained Heat-3D recovery.
- [Train on your own data](docs/user_guide.md#training-with-your-own-data) using your forward model or paired states and measurements.

## Reproduce the paper

The [experiment guide](reproduce/README.md) lists the scripts, data
generation, and expected outputs for the seven PDE benchmarks. See
[Data](reproduce/data/README.md) for the held-out arrays and frozen
checkpoints. To run the complete Heat-3D benchmark from source:

```bash
git clone https://github.com/AadityaKachhadiya/deceptron.git
cd deceptron
python -m pip install -e ".[demo]" pandas scipy
python reproduce/experiments/heat3d.py
```

The script generates the data, trains the models, and writes
`runs/heat3d/heat3d_cnn_main_results.csv` for comparison with the
[archived result](reproduce/results/heat3d_cnn_main_results.csv).

- [Retrain another benchmark](reproduce/README.md#pde-inverse-problems) from the seven-PDE experiment table.
- [Replay a paper result](reproduce/README.md#checkpoint-replay) using the Allen–Cahn or Navier–Stokes checkpoints and test data.
- [Regenerate the figures](reproduce/README.md#figures) from the archived evaluation traces.

## Benchmark figures

<p align="center">
  <img src="examples/assets/navier_stokes_withheld_t2.png" width="620" alt="Navier–Stokes recovery from one noisy snapshot and absolute error at the withheld time">
</p>

**Navier–Stokes.** Recover the initial vorticity from one noisy snapshot
and compare the resulting flow with the withheld snapshot. [Run and data](reproduce/README.md#navierstokes-with-a-withheld-observation)

## Training cost and break-even

The table gives the approximate number of inverse solves needed to offset
the full Deceptron training cost using the original benchmark timings.
The estimate depends on the benchmark and the baseline solver.

| Benchmark | Gauss–Newton | Levenberg–Marquardt | L-BFGS |
| --- | ---: | ---: | ---: |
| Heat-3D | 1,266 | 1,597 | — |
| Advection–diffusion 2D | 546 | 550 | 1,692 |
| Allen–Cahn 2D | 586 | 139 | 6,049 |

The separate Allen–Cahn comparison below uses a shared measurement
tolerance for D-IPG and Levenberg–Marquardt. Under that protocol, the
projected break-even point is **628 solves**.

<p align="center">
  <img src="examples/assets/allen_cahn_break_even.png" width="380" alt="Cumulative wall-clock time for Allen–Cahn, with a break-even point at 628 solves">
</p>

[Run and timing protocol](reproduce/README.md#matched-target-allencahn-and-amortization).

## Cite this work

For work that uses this repository, cite the 2026 preprint.

```bibtex
@misc{kachhadiya2026localinversegeometryamortized,
      title={Local Inverse Geometry Can Be Amortized},
      author={Aaditya L. Kachhadiya},
      year={2026},
      eprint={2605.13068},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.13068},
}
```

**Earlier workshop paper.** [A. L. Kachhadiya. Deceptron: Learned Local Inverses for Fast and Stable Physics Inversion. *Machine Learning and the Physical Sciences Workshop, NeurIPS*, 2025](https://ml4physicalsciences.github.io/2025/files/NeurIPS_ML4PS_2025_13.pdf).
See the [workshop BibTeX](docs/user_guide.md#workshop-paper) when citing that version.

## Questions

For help using the package, data, or benchmark scripts, open a
[GitHub issue](https://github.com/AadityaKachhadiya/deceptron/issues).
Include the command you ran and any error output.

## License

Deceptron is released under the [MIT License](LICENSE).
