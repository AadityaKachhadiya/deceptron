"""Recompute manuscript summaries from the bundled archived result files."""

import csv
import math
from pathlib import Path
from statistics import mean

RESULTS = Path(__file__).with_name("results")

BENCHMARKS = (
    ("Heat-1D", "heat1d_main_results.csv"),
    ("Heat-2D", "heat2d_cnn_main_results.csv"),
    ("Heat-3D", "heat3d_cnn_main_results.csv"),
    ("Advection–diffusion", "advdiff_degraded_main_results.csv"),
    ("Allen–Cahn", "allen_cahn2d_main_results.csv"),
    ("Darcy", "darcy_multisource_main_results.csv"),
    ("Navier–Stokes", "navier_stokes_main_results.csv"),
)


def read_rows(filename):
    with (RESULTS / filename).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def mean_metric(rows, name):
    return mean(float(row[name]) for row in rows)


def main():
    print("Original benchmark summaries | D-IPG (+JCP)")
    for name, filename in BENCHMARKS:
        row, = (
            row for row in read_rows(filename)
            if row["method"] == "D-IPG" and row["ablation"] == "+JCP"
        )
        print(
            f"{name:20} n={row['num_instances']:>3}  "
            f"success={float(row['success_rate']):7.2%}  "
            f"state RMSE={float(row['mean_rmse_x']):.5f}"
        )

    allen = read_rows("allen_cahn_matched_cpu_replay.csv")
    print("\nAllen–Cahn matched target | CPU replay of reconstruction metrics")
    for method in ("dipg", "lm"):
        hits = sum(float(row[f"{method}_measurement_rmse"]) <= 0.12 for row in allen)
        recovered = sum(float(row[f"{method}_state_rmse"]) <= 0.10 for row in allen)
        state_rmse = mean_metric(allen, f"{method}_state_rmse")
        print(
            f"{method:4} target hits={hits}/{len(allen)}  "
            f"state successes={recovered}/{len(allen)}  state RMSE={state_rmse:.5f}"
        )

    # The GPU timings are the reported manuscript values, not CPU replay times.
    training_seconds = 1592.89
    dipg_seconds, lm_seconds = 0.1394, 2.6783
    crossing = training_seconds / (lm_seconds - dipg_seconds)
    print(f"Break-even using reported GPU timings: {math.ceil(crossing)} solves")

    ns = read_rows("navier_stokes_withheld_t2_per_case.csv")
    print("\nNavier–Stokes | initial-state and withheld-time RMSE")
    for method in ("fno", "one_shot_plus", "dipg_plus", "dipg_minus"):
        rows = [row for row in ns if row["method"] == method]
        print(
            f"{method:15} n={len(rows)}  state={mean_metric(rows, 'state_rmse'):.5f}  "
            f"withheld={mean_metric(rows, 'withheld_t2_rmse'):.5f}"
        )


if __name__ == "__main__":
    main()
