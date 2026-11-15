"""Recover one held-out initial temperature field with a frozen checkpoint."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from deceptron import DeceptronCNN3D, SolverConfig, solve_dipg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("heat3d_recovery.png"))
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.manual_seed(7)
    fixture_path = Path(__file__).with_name("assets") / "heat3d_seed7_case07.pt"
    case = torch.load(fixture_path, map_location="cpu", weights_only=True)
    model = DeceptronCNN3D(**case["model_config"])
    model.load_state_dict(case["model_state"])
    model.eval().requires_grad_(False)

    truth = case["state"]
    result = solve_dipg(
        model, case["normalized_observation"], torch.zeros_like(truth),
        SolverConfig(**case["solver_config"]),
    )
    recovered = result["x_hat"]
    state_rmse = (recovered - truth).square().mean().sqrt().item()

    grid = tuple(case["model_config"][axis] for axis in ("nx", "ny", "nz"))
    middle = grid[-1] // 2
    panels = (
        ("Noisy measurement", case["observation"]),
        ("D-IPG recovery", recovered),
        ("True initial field", truth),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.55), layout="constrained")
    for ax, (title, field) in zip(axes, panels):
        image = ax.imshow(
            field.reshape(grid)[:, :, middle].T.numpy(), origin="lower",
            interpolation="nearest", cmap="viridis",
            vmin=None if title == "Noisy measurement" else 0,
            vmax=None if title == "Noisy measurement" else 1,
        )
        ax.set_title(title)
        ax.set_axis_off()
        if title == "Noisy measurement":
            fig.colorbar(image, ax=ax, shrink=0.78)
    fig.colorbar(image, ax=axes[1:], shrink=0.78, label="Initial field")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Heat-3D, seed {case['seed']}, held-out case {case['case_index']}")
    print(f"State RMSE: {state_rmse:.5f}")
    print(f"Normalized measurement RMSE: {result['final_residual']:.5f}")
    print(f"Reconstruction saved to {args.output}")


if __name__ == "__main__":
    main()
