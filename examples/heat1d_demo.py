"""
Heat-1D inverse problem benchmark using the deceptron package.

From the repository root:
    pip install -e .

Or install directly from GitHub:
    pip install git+https://github.com/AadityaKachhadiya/deceptron.git

Run with:
    python examples/heat1d_demo.py
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from deceptron import (
    DeceptronMLP,
    TrainConfig,
    train_forward,
    train_reverse,
    train_reverse_jcp,
    SolverConfig,
    solve_dipg,
    solve_gradient_descent,
    solve_gauss_newton,
    solve_levenberg_marquardt,
    estimate_rjcp_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


@dataclass
class Cfg:
    n_grid: int = 64
    t_final: float = 0.08
    diffusivity: float = 0.18

    obs_nonlinear_cubic: float = 0.18
    obs_nonlinear_sin: float = 0.08
    obs_mix_strength: float = 0.10
    noise_std: float = 0.01

    n_train: int = 4000
    n_val: int = 500
    n_test: int = 300

    basis_keep: int = 10
    x_low: float = 0.0
    x_high: float = 1.0

    negative_slope: float = 0.10
    reverse_hidden_multiplier: int = 2
    batch_size: int = 128


cfg = Cfg()


class NonlinearHeat1D(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()

        grid = torch.linspace(0.0, 1.0, c.n_grid + 2)[1:-1]
        k = torch.arange(1, c.n_grid + 1, dtype=torch.float32)

        basis = torch.sin(math.pi * grid[:, None] * k[None, :]) * math.sqrt(2.0 / (c.n_grid + 1))
        decay = torch.exp(-c.diffusivity * (math.pi * k) ** 2 * c.t_final)

        mix = torch.eye(c.n_grid)
        for i in range(c.n_grid):
            if i > 0:
                mix[i, i - 1] = c.obs_mix_strength
            if i < c.n_grid - 1:
                mix[i, i + 1] = c.obs_mix_strength
        mix = mix / mix.sum(dim=1, keepdim=True)

        self.cubic = c.obs_nonlinear_cubic
        self.sinc = c.obs_nonlinear_sin

        self.register_buffer("basis", basis)
        self.register_buffer("decay", decay)
        self.register_buffer("mix", mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        coeff = x @ self.basis
        u_t = (coeff * self.decay) @ self.basis.T
        z = u_t @ self.mix.T
        return z + self.cubic * z**3 + self.sinc * torch.sin(3.0 * z)


true_model = NonlinearHeat1D(cfg).to(device)
print(f"true model dimension: R^{cfg.n_grid} -> R^{cfg.n_grid}")


def build_dataset(c: Cfg, tm: nn.Module, seed: int = 7):
    torch.manual_seed(seed)

    grid = torch.linspace(0.0, 1.0, c.n_grid + 2)[1:-1]
    k = torch.arange(1, c.basis_keep + 1, dtype=torch.float32)

    basis = torch.sin(math.pi * grid[:, None] * k[None, :])
    basis = basis / basis.norm(dim=0, keepdim=True)

    n_total = c.n_train + c.n_val + c.n_test
    coeff = torch.randn(n_total, c.basis_keep)
    coeff[:, 0] += 0.25 * torch.randn(n_total)
    coeff[:, 1:] *= 1.0 / torch.arange(2, c.basis_keep + 1, dtype=torch.float32)

    x = coeff @ basis.T
    x_min = x.min(1, keepdim=True).values
    x_max = x.max(1, keepdim=True).values
    x = c.x_low + (c.x_high - c.x_low) * (x - x_min) / (x_max - x_min + 1e-8)

    with torch.no_grad():
        y_clean = tm(x.to(device)).cpu()

    y_noisy = y_clean + c.noise_std * torch.randn_like(y_clean)

    nt, nv = c.n_train, c.n_val
    y_mean = y_clean[:nt].mean(0, keepdim=True)
    y_std = y_clean[:nt].std(0, keepdim=True).clamp(1e-6)

    def normalize(y: torch.Tensor) -> torch.Tensor:
        return (y - y_mean) / y_std

    return {
        "x_train": x[:nt],
        "x_val": x[nt : nt + nv],
        "x_test": x[nt + nv :],
        "y_train_norm": normalize(y_clean[:nt]),
        "y_val_norm": normalize(y_clean[nt : nt + nv]),
        "y_test_norm": normalize(y_noisy[nt + nv :]),
    }


data = build_dataset(cfg, true_model, seed=7)
tr_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
)
val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
)
x_val = data["x_val"].to(device)

print(f"dataset: train={cfg.n_train}, val={cfg.n_val}, test={cfg.n_test}")


model = DeceptronMLP(
    dim=cfg.n_grid,
    negative_slope=cfg.negative_slope,
    hidden_multiplier=cfg.reverse_hidden_multiplier,
).to(device)

train_cfg = TrainConfig(
    forward_epochs=140,
    forward_lr=2e-3,
    forward_wd=1e-6,
    reverse_epochs=100,
    reverse_lr=2e-3,
    reverse_wd=1e-6,
    jcp_epochs=120,
    jcp_lr=1e-3,
    jcp_wd=1e-6,
    reconstruction_weight=1.0,
    cycle_weight=0.25,
    bias_tie_weight=5e-4,
    composition_weight=1e-3,
    probe_jcp_weight=1.0,
    jcp_num_probes_train=2,
    jcp_num_probes_eval=4,
    jcp_batch_subsample=16,
    y_tilde_noise=0.02,
    gradient_clip=5.0,
    eval_subset_rjcp=64,
    use_cosine_lr=False,
)

print("stage 1")
model_s1 = train_forward(model, tr_loader, val_loader, x_val, train_cfg, device)

print("stage 2")
model_s2 = train_reverse(model_s1, tr_loader, val_loader, x_val, train_cfg, device)

print("stage 3 (+JCP)")
model_plus = train_reverse_jcp(
    model_s2,
    tr_loader,
    val_loader,
    x_val,
    train_cfg,
    device,
    use_jcp=True,
    mlp_extra_losses=True,
)

print("stage 3 (-JCP)")
model_minus = train_reverse_jcp(
    model_s2,
    tr_loader,
    val_loader,
    x_val,
    train_cfg,
    device,
    use_jcp=False,
    mlp_extra_losses=True,
)

rjcp_plus = estimate_rjcp_dataset(model_plus, x_val, num_probes=4, max_samples=64)
rjcp_minus = estimate_rjcp_dataset(model_minus, x_val, num_probes=4, max_samples=64)
print(f"+JCP RJCP = {rjcp_plus:.4f}")
print(f"-JCP RJCP = {rjcp_minus:.4f}")
print(f"ratio = {rjcp_minus / rjcp_plus:.2f}x")


solver_cfg = SolverConfig(
    max_iterations=120,
    tolerance_eps=0.30,
    armijo_c=1e-4,
    max_backtracking_steps=8,
    rho=0.4,
    x_low=cfg.x_low,
    x_high=cfg.x_high,
    dipg_step_size=1.0,
    gn_step_size=1.0,
    lm_step_size=1.0,
    lm_damping=1e-2,
    num_probes_eval=2,
)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.cpu() - b.cpu()) ** 2)))


def evaluate(mdl: nn.Module, label: str) -> pd.DataFrame:
    x_test = data["x_test"]
    y_test = data["y_test_norm"]
    dev = next(mdl.parameters()).device

    methods = {
        "D-IPG": lambda y, x0: solve_dipg(mdl, y, x0, solver_cfg),
        "GD": lambda y, x0: solve_gradient_descent(mdl.f, y, x0, solver_cfg),
        "GN": lambda y, x0: solve_gauss_newton(mdl.f, y, x0, solver_cfg),
        "LM": lambda y, x0: solve_levenberg_marquardt(mdl.f, y, x0, solver_cfg),
    }

    rows = []
    for method_name, method_fn in methods.items():
        per = [
            method_fn(y_test[i].to(dev), torch.zeros_like(x_test[i]))
            for i in range(x_test.shape[0])
        ]
        rmses = [rmse(r["x_hat"], x_test[i]) for i, r in enumerate(per)]

        rows.append(
            {
                "ablation": label,
                "method": method_name,
                "success_rate": float(np.mean([r["success"] for r in per])),
                "median_iters": float(np.median([r["iters"] for r in per])),
                "mean_rmse_x": float(np.mean(rmses)),
                "mean_time_sec": float(np.mean([r["time_sec"] for r in per])),
            }
        )
        print(f"completed: {label} | {method_name}")

    return pd.DataFrame(rows)


print("evaluating")
results = pd.concat(
    [
        evaluate(model_plus, "+JCP"),
        evaluate(model_minus, "-JCP"),
    ],
    ignore_index=True,
)

print("\n" + "=" * 68)
print(f"{'Ablation':8} {'Method':6} {'SR':>7} {'Med.It':>7} {'RMSE':>8} {'Time':>9}")
print("=" * 68)
for _, row in results.iterrows():
    print(
        f"{row['ablation']:8} "
        f"{row['method']:6} "
        f"{row['success_rate']:>7.1%} "
        f"{row['median_iters']:>7.0f} "
        f"{row['mean_rmse_x']:>8.4f} "
        f"{row['mean_time_sec']:>8.4f}s"
    )
print("=" * 68)

results.to_csv("heat1d_main_results.csv", index=False)
torch.save(model_plus.state_dict(), "heat1d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "heat1d_minus_jcp.pt")
