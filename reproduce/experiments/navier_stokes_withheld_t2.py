"""Replay the 80-case single-snapshot Navier–Stokes comparison."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from _fno import FNO2dInverse
from _output import prepare_output
from navier_stokes import DeceptronNSModel, NavierStokes2DConfig, solve_navier_stokes_snapshots


DATA = Path(__file__).resolve().parents[1] / "data"
FIGURE_CASE = 52


def load_deceptron(config, variant):
    model = DeceptronNSModel(
        config.x_channels, config.y_channels, config.hidden_channels,
        config.num_res_blocks, config.negative_slope,
    )
    path = DATA / "checkpoints" / f"navier_stokes_model_{variant}_jcp.pt"
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval().requires_grad_(False)


def solve(model, observed, initial, config):
    x = initial.clone()
    for _ in range(16):
        x_req = x.detach().requires_grad_(True)
        prediction = model.f(x_req)
        residual = prediction[:, :1] - observed
        objective = 0.5 * residual.square().mean()
        gradient, = torch.autograd.grad(objective, x_req)

        with torch.no_grad():
            alpha = 1.6
            for _ in range(config.max_backtracking_steps + 1):
                corrected = torch.cat((prediction[:, :1] - alpha * residual, prediction[:, 1:]), dim=1)
                proposal = model.g(corrected)
                trial = ((1 - config.rho) * x + config.rho * proposal).clamp(-1, 1)
                trial_residual = model.f(trial)[:, :1] - observed
                threshold = objective + config.armijo_c * config.rho * (gradient * (proposal - x)).sum()
                if 0.5 * trial_residual.square().mean() <= threshold:
                    x = trial
                    break
                alpha *= 0.5
            else:
                break
    return x.detach()


def save_figure(observed, reference, predictions):
    names = ("FNO", "One-shot (+JCP)", "D-IPG (+JCP)")
    fields = [predictions[name] for name in ("fno", "one_shot_plus", "dipg_plus")]
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 3.8), layout="constrained")
    for axis, title, field in zip(axes[0], ("Withheld $t_2$", *names), (reference, *fields)):
        field_image = axis.imshow(field, cmap="RdBu_r", vmin=-1.1, vmax=1.1, origin="lower")
        axis.set_title(title)
    axes[1, 0].imshow(observed, cmap="RdBu_r", vmin=-1.1, vmax=1.1, origin="lower")
    axes[1, 0].set_xlabel("Observed $t_1$")
    for axis, field in zip(axes[1, 1:], fields):
        error_image = axis.imshow(abs(field - reference), cmap="magma", vmin=0, vmax=0.7, origin="lower")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(field_image, ax=axes[0], shrink=0.85, label="Vorticity")
    fig.colorbar(error_image, ax=axes[1], shrink=0.85, label="Absolute error at $t_2$")
    fig.savefig("navier_stokes_withheld_t2.png", dpi=180)
    plt.close(fig)


def main():
    prepare_output("navier_stokes_withheld_t2")
    torch.set_num_threads(2)
    plt.switch_backend("Agg")
    config = NavierStokes2DConfig()
    plus = load_deceptron(config, "plus")
    minus = load_deceptron(config, "minus")
    checkpoint = torch.load(DATA / "checkpoints" / "ns_fno_w24_m4_seed7.pt",
                            map_location="cpu", weights_only=True)
    fno = FNO2dInverse(config.y_channels, config.x_channels, checkpoint["width"],
                       checkpoint["modes"], checkpoint["depth"], checkpoint["projection_width"])
    fno.load_state_dict(checkpoint["state_dict"])
    fno.eval().requires_grad_(False)

    with np.load(DATA / "navier_stokes_dataset_1600_200_80_test.npz") as data:
        truth = data["x_test"][:, 0]
        observed_raw = data["y_noisy_test"][:, 0]
        reference = data["y_clean_test"][:, 1]
        observations = torch.from_numpy(data["y_noisy_test_norm"][:, :1].copy())

    _, check = solve_navier_stokes_snapshots(truth[0], config)
    np.testing.assert_allclose(check, reference[0], rtol=0, atol=1e-6)
    output = Path("navier_stokes_withheld_t2_per_case.csv")
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["case", "method", "state_rmse", "withheld_t2_rmse"])
        for case, observation in enumerate(observations):
            observed = observation.unsqueeze(0)
            initial = torch.zeros_like(observed)
            with torch.no_grad():
                completed = torch.cat((observed, plus.f(initial)[:, 1:]), dim=1)
                fields = {
                    "fno": fno(completed).clamp(-1, 1),
                    "one_shot_plus": plus.g(completed).clamp(-1, 1),
                }
            fields["dipg_plus"] = solve(plus, observed, initial, config)
            fields["dipg_minus"] = solve(minus, observed, initial, config)

            predictions = {}
            for method, field in fields.items():
                recovered = field[0, 0].numpy()
                _, predicted = solve_navier_stokes_snapshots(recovered, config)
                predictions[method] = predicted
                writer.writerow([
                    case, method, np.sqrt(np.mean((recovered - truth[case]) ** 2)),
                    np.sqrt(np.mean((predicted - reference[case]) ** 2)),
                ])
            if case == FIGURE_CASE:
                save_figure(observed_raw[case], reference[case], predictions)
            if (case + 1) % 10 == 0:
                print(f"{case + 1}/{len(truth)} cases", flush=True)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
