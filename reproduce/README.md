# Reproduce the paper

From the repository root, run:

```bash
python reproduce/summarize_results.py
```

This reads the archived CSVs and reports the seven benchmark summaries,
the matched-target Allen–Cahn reconstruction metrics and break-even
calculation, and the single-snapshot Navier–Stokes errors. It uses the
Python standard library and takes no training or GPU compute.

## Original PDE experiments

Each notebook preserves the source cells from the corresponding
original experiment with execution outputs cleared. Use a GPU Colab
runtime for the original notebooks. They contain the data-generation,
model training, and evaluation code, including the seed-7 setup.
The original notebooks have separate sections for extended analyses
and seed sweeps. The commands above recover the saved summary numbers.
Full retraining in a new runtime is pending verification.

| PDE | Original notebook | Archived summary | Original success criterion |
| --- | --- | --- | --- |
| Heat-1D | [Notebook](notebooks/Heat1D.ipynb) | [CSV](results/heat1d_main_results.csv) | Normalized measurement RMSE ≤ 0.30 |
| Heat-2D | [Notebook](notebooks/Heat2D.ipynb) | [CSV](results/heat2d_cnn_main_results.csv) | Normalized measurement RMSE ≤ 0.40 |
| Heat-3D | [Notebook](notebooks/Heat3D.ipynb) | [CSV](results/heat3d_cnn_main_results.csv) | Normalized measurement RMSE ≤ 0.38 |
| Advection–diffusion | [Notebook](notebooks/Advection_diffusion.ipynb) | [CSV](results/advdiff_degraded_main_results.csv) | State RMSE ≤ 0.10 |
| Allen–Cahn | [Notebook](notebooks/AllenCahn2D.ipynb) | [CSV](results/allen_cahn2d_main_results.csv) | State RMSE ≤ 0.10 |
| Darcy | [Notebook](notebooks/Darcy2D.ipynb) | [CSV](results/darcy_multisource_main_results.csv) | State RMSE ≤ 0.18 |
| Navier–Stokes | [Notebook](notebooks/NavierStokes2D.ipynb) | [CSV](results/navier_stokes_main_results.csv) | State RMSE ≤ 0.12 |

For the [Heat-3D quick start](../README.md#recover-an-initial-temperature-field),
the model and held-out observation come from the original run. A CPU replay
with the archived checkpoint reproduced the original mean state RMSE
(0.063546), mean normalized measurement RMSE (0.290711), and 80/80 measurement
target hits. The quick start bundles one of the 80 cases.

## Matched-target Allen–Cahn and amortization

The [paired CPU replay](results/allen_cahn_matched_cpu_replay.csv)
uses seed 7, the archived +JCP checkpoint, zero initialization, and
LM damping of 0.0001, selected on 20 validation cases. The shared test
target is normalized measurement RMSE at most 0.12. Recovery success
uses state RMSE at most 0.10. Both methods hit the target in all 80
test cases. Mean state RMSE is 0.09399 for D-IPG and 0.14650 for LM.

The break-even calculation uses the original reported GPU timings
(0.1394 s and 2.6783 s per solve) and charges the full 1,592.89 s
training pipeline to D-IPG. This gives 628 whole solves. The original
comparison used a single NVIDIA T4, warmed up both solvers, alternated
their evaluation order, and synchronized the GPU before timing.
The attached CPU replay provides the paired reconstruction metrics.
The original per-case GPU timing log is still being located.

## Navier–Stokes with a withheld observation

The [saved protocol](results/ns_partial_observation_matched_protocol_seed7.csv)
specifies the frozen models, zero initialization, and observation mask
for all 80 paired cases. The observed input is the noisy vorticity at
the first time. The second snapshot is withheld during inversion.
The same PDE solver advances each recovered initial state to the
withheld time.

The [per-case rollout scores](results/navier_stokes_withheld_t2_per_case.csv)
include both state RMSE and withheld-time RMSE. The summary command
recomputes the four reported means and includes the −JCP variant.
