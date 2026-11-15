"""Replay the matched-target Allen–Cahn comparison on the archived test split."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import allen_cahn as benchmark
from _output import prepare_output


DATA = Path(__file__).resolve().parents[1] / "data"


def main():
    prepare_output("allen_cahn_matched")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    benchmark.device = torch.device("cpu")

    metadata = json.loads((DATA / "allen_cahn2d_run_metadata_seed7.json").read_text())
    config = benchmark.AllenCahn2DConfig(**metadata["config"])
    lm_config = replace(config, lm_damping=1e-4)
    model = benchmark.DeceptronAllenCahnModel(
        config.x_channels, config.y_channels, config.hidden_channels,
        config.num_res_blocks, config.negative_slope,
    )
    checkpoint = DATA / "checkpoints" / "allen_cahn2d_model_plus_jcp.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().requires_grad_(False)

    with np.load(DATA / "allen_cahn2d_dataset_seed7_test.npz") as data:
        truth = torch.from_numpy(data["x_test"])
        observations = torch.from_numpy(data["y_noisy_test_norm"])

    output = Path("allen_cahn_matched_cpu_replay.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "case", "dipg_state_rmse", "lm_state_rmse",
            "dipg_measurement_rmse", "lm_measurement_rmse",
        ])
        for case, (state, observation) in enumerate(zip(truth, observations)):
            state, observation = state.unsqueeze(0), observation.unsqueeze(0)
            initial = torch.zeros_like(state)
            dipg = benchmark.solve_dipg(model, observation, initial, config)
            lm = benchmark.solve_lm(model.f, observation, initial, lm_config)
            writer.writerow([
                case, benchmark.rmse_x(dipg["x_hat"], state),
                benchmark.rmse_x(lm["x_hat"], state),
                dipg["final_residual"], lm["final_residual"],
            ])
            if (case + 1) % 20 == 0:
                print(f"{case + 1}/{len(truth)} cases", flush=True)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
