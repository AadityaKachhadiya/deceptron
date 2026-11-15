# Reproduce the paper

## Installation

From the repository root, install the benchmark dependencies.

```bash
python -m pip install -e ".[demo]" pandas scipy
```

Use a GPU for full training. The original timings were measured on an
NVIDIA T4. In Colab, select a GPU runtime, change into the cloned
repository with `%cd /content/deceptron`, and run the shell commands with
`!`. Checkpoint replay and figure generation run on CPU.

## PDE inverse problems

Choose a script from the table and run it from the repository root.
For example, the Heat-3D script generates data, trains both JCP variants,
and evaluates the solvers on 80 held-out cases.

```bash
python reproduce/experiments/heat3d.py
```

| Benchmark | Inverse problem | Script | Notebook | Archived results |
| --- | --- | --- | --- | --- |
| Heat-1D | Initial temperature from a noisy, mixed final field | [heat1d.py](experiments/heat1d.py) | [Notebook](notebooks/Heat1D.ipynb) | [CSV](results/heat1d_main_results.csv) |
| Heat-2D | Initial temperature from a blurred, noisy final field | [heat2d.py](experiments/heat2d.py) | [Notebook](notebooks/Heat2D.ipynb) | [CSV](results/heat2d_cnn_main_results.csv) |
| Heat-3D | Initial temperature from a blurred, noisy final field | [heat3d.py](experiments/heat3d.py) | [Notebook](notebooks/Heat3D.ipynb) | [CSV](results/heat3d_cnn_main_results.csv) |
| Advection–diffusion | Initial scalar field from a blurred, downsampled, noisy final field | [advection_diffusion.py](experiments/advection_diffusion.py) | [Notebook](notebooks/Advection_diffusion.ipynb) | [CSV](results/advdiff_degraded_main_results.csv) |
| Allen–Cahn | Initial phase field from a noisy later phase field | [allen_cahn.py](experiments/allen_cahn.py) | [Notebook](notebooks/AllenCahn2D.ipynb) | [CSV](results/allen_cahn2d_main_results.csv) |
| Darcy | Permeability from four noisy pressure fields | [darcy.py](experiments/darcy.py) | [Notebook](notebooks/Darcy2D.ipynb) | [CSV](results/darcy_multisource_main_results.csv) |
| Navier–Stokes | Initial vorticity from two noisy snapshots | [navier_stokes.py](experiments/navier_stokes.py) | [Notebook](notebooks/NavierStokes2D.ipynb) | [CSV](results/navier_stokes_main_results.csv) |

Scripts use the original seed-7 settings and write CSVs, training
histories, and checkpoints under `runs/<script name>/`. Use
`--output-dir` with a new path to repeat a run. The notebooks contain the
original experiment code, extended plots, and seed sweeps.

A fresh Colab run of advection–diffusion reproduced 80/80 successes for
D-IPG (+JCP), with mean state RMSE 0.0560. Fresh full-budget GPU runs of
the other six benchmarks remain to be verified.

### Data and evaluation

Each script generates synthetic training, validation, and test data
using its PDE simulator. The [data directory](data/README.md) also
provides the original held-out arrays for all seven benchmarks.

The benchmark CSVs report **mean state RMSE** and **success rate**. State
RMSE compares the recovered field with the true field. Success rate is
the fraction of cases meeting the original criterion for that benchmark:

- **Heat diffusion:** normalized measurement RMSE ≤ 0.30 (1D), 0.40 (2D), or 0.38 (3D).
- **Advection–diffusion, Allen–Cahn, Darcy, and Navier–Stokes:** state RMSE ≤ 0.10, 0.10, 0.18, and 0.12, respectively.

Measurement RMSE compares predicted and observed measurements in
normalized observation space. Use the criterion in this list when
comparing a fresh result with its archived CSV.

## Checkpoint replay

These comparisons use the supplied frozen checkpoints and held-out
data. Both evaluate 80 cases on CPU.

<a name="matched-target-allencahn-and-amortization"></a>

### Allen–Cahn at a shared measurement tolerance

```bash
python reproduce/experiments/allen_cahn_matched.py
```

D-IPG (+JCP) and Levenberg–Marquardt both stop at a normalized
measurement RMSE of 0.12 and are evaluated on the recovered field.
Both reach the measurement target in all 80 cases. Mean state RMSE is
0.09399 for D-IPG and 0.14650 for LM. The script saves
`runs/allen_cahn_matched/allen_cahn_matched_cpu_replay.csv`.
[Archived scores](results/allen_cahn_matched_cpu_replay.csv)

The break-even estimate uses the original T4 timings of 0.1394 s and
2.6783 s per solve, with the full 1,592.89 s training cost charged to
D-IPG. The projected crossover is **628 solves**. The original timing
run used a warm-up, alternating solver order, and GPU synchronization.
The CPU replay verifies the reconstruction metrics.

### Navier–Stokes with a withheld observation

```bash
python reproduce/experiments/navier_stokes_withheld_t2.py
```

The inversion uses the first noisy vorticity snapshot. The PDE solver
then evolves each recovery to the second snapshot to measure predictive
error. The script writes the per-case CSV and a field comparison for
case 52 under `runs/navier_stokes_withheld_t2/`.
[Protocol](results/ns_partial_observation_matched_protocol_seed7.csv) ·
[Archived scores](results/navier_stokes_withheld_t2_per_case.csv)

<img src="../examples/assets/navier_stokes_withheld_t2.png" width="620" alt="Vorticity predictions and absolute errors at the withheld time">

Both replays matched the archived RMSE values within `5e-7` in local CPU checks.

## Figures

```bash
python reproduce/experiments/figures_deceptron.py
```

The script reads the [archived traces and ablation results](results/traces/)
and writes solver profiles, Allen–Cahn basin diagnostics, the reliability
matrix, and appendix plots to `runs/figures/`. The paper uses the same
results with its own layout.

## Summary tables

```bash
python reproduce/summarize_results.py
```

This command reads the archived CSVs and prints the seven benchmark
summaries, the Allen–Cahn comparison and break-even estimate, and the
Navier–Stokes errors. It runs with the Python standard library.
