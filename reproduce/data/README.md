# Data

These NumPy archives contain the held-out arrays exported from the
original benchmark datasets. The experiment scripts generate training
and validation data using the PDE simulators and settings in the code.

| Benchmark | Held-out data | Cases |
| --- | --- | ---: |
| Heat-1D | [NPZ](heat1d_dataset_seed7_test.npz) | 300 |
| Heat-2D | [NPZ](heat2d_dataset_seed7_test.npz) | 100 |
| Heat-3D | [NPZ](heat3d_dataset_seed7_test.npz) | 80 |
| Advection–diffusion | [NPZ](advdiff2d_dataset_seed7_test.npz) | 80 |
| Allen–Cahn | [NPZ](allen_cahn2d_dataset_seed7_test.npz) | 80 |
| Darcy | [NPZ](darcy2d_dataset_seed7_test.npz) | 80 |
| Navier–Stokes | [NPZ](navier_stokes_dataset_1600_200_80_test.npz) | 80 |

`x_test` contains the true unknown states. `y_clean_test` contains the
forward observations before additive noise. `y_noisy_test_norm`
contains the normalized measurements used for inversion, with
`y_test_norm` carrying that role in the advection–diffusion archive.
`y_mean` and `y_std` record the training-set normalization. See
[manifest.json](manifest.json) for array shapes, source names, and
SHA-256 checksums.

For example, from the repository root:

```python
import numpy as np

with np.load("reproduce/data/navier_stokes_dataset_1600_200_80_test.npz") as data:
    observed_t1 = data["y_noisy_test_norm"][:, 0]
    withheld_t2 = data["y_clean_test"][:, 1]
```

The single-snapshot inversion uses `observed_t1`. The clean second
snapshot is the reference for scoring the PDE rollout.

The [checkpoints](checkpoints/) contain the frozen models used by the
[Allen–Cahn and Navier–Stokes replay commands](../README.md#checkpoint-replay).
The data and checkpoint manifest preserves the names of the source archives.
