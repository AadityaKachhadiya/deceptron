"""
Heat-3D inverse problem benchmark using the deceptron package.

From the repository root:
    pip install -e .

Or install directly from GitHub:
    pip install git+https://github.com/AadityaKachhadiya/deceptron.git

Run with:
    python examples/heat3d_demo.py
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deceptron import (
    DeceptronCNN3D,
    TrainConfig,
    train_forward,
    train_reverse,
    train_reverse_jcp,
    SolverConfig,
    solve_gradient_descent,
    solve_gauss_newton,
    solve_levenberg_marquardt,
    estimate_rjcp_dataset,
)
from deceptron.jcp import single_sample_probe_jcp
from deceptron.solvers import _phi_grad, _normalized_residual, _armijo_accept

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


@dataclass
class Cfg:
    nx: int = 8
    ny: int = 8
    nz: int = 8
    dim: int = 512

    t_final: float = 0.012
    diffusivity: float = 0.10

    obs_blur_strength: float = 0.025
    obs_nonlinear_cubic: float = 0.020
    obs_nonlinear_sin: float = 0.008
    noise_std: float = 0.0015

    n_train: int = 4000
    n_val: int = 400
    n_test: int = 80

    modes_x: int = 3
    modes_y: int = 3
    modes_z: int = 3

    x_low: float = 0.0
    x_high: float = 1.0

    negative_slope: float = 0.10
    hidden_channels: int = 12
    batch_size: int = 32


cfg = Cfg()
print(f"grid: {cfg.nx}x{cfg.ny}x{cfg.nz} | dim={cfg.dim}")


class NonlinearHeat3D(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        self.nx, self.ny, self.nz = c.nx, c.ny, c.nz
        self.blur = c.obs_blur_strength
        self.cubic = c.obs_nonlinear_cubic
        self.sinc = c.obs_nonlinear_sin

        gx = torch.linspace(0, 1, c.nx + 2)[1:-1]
        gy = torch.linspace(0, 1, c.ny + 2)[1:-1]
        gz = torch.linspace(0, 1, c.nz + 2)[1:-1]

        kx = torch.arange(1, c.nx + 1, dtype=torch.float32)
        ky = torch.arange(1, c.ny + 1, dtype=torch.float32)
        kz = torch.arange(1, c.nz + 1, dtype=torch.float32)

        bx = torch.sin(math.pi * gx[:, None] * kx[None, :]) * math.sqrt(2 / (c.nx + 1))
        by = torch.sin(math.pi * gy[:, None] * ky[None, :]) * math.sqrt(2 / (c.ny + 1))
        bz = torch.sin(math.pi * gz[:, None] * kz[None, :]) * math.sqrt(2 / (c.nz + 1))

        decay = torch.exp(
            -c.diffusivity
            * (
                (math.pi * kx[:, None, None]) ** 2
                + (math.pi * ky[None, :, None]) ** 2
                + (math.pi * kz[None, None, :]) ** 2
            )
            * c.t_final
        )

        kernel = torch.zeros(3, 3, 3)
        kernel[1, 1, 1] = 8.0
        kernel[0, 1, 1] = 1.0
        kernel[2, 1, 1] = 1.0
        kernel[1, 0, 1] = 1.0
        kernel[1, 2, 1] = 1.0
        kernel[1, 1, 0] = 1.0
        kernel[1, 1, 2] = 1.0
        kernel = kernel / kernel.sum()

        self.register_buffer("bx", bx)
        self.register_buffer("by", by)
        self.register_buffer("bz", bz)
        self.register_buffer("decay", decay)
        self.register_buffer("blur_kernel", kernel.view(1, 1, 3, 3, 3))

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(-1, self.nx, self.ny, self.nz)
        coeff = torch.einsum("bxyz,xi,yj,zk->bijk", x, self.bx, self.by, self.bz)
        u = torch.einsum("bijk,ijk,xi,yj,zk->bxyz", coeff, self.decay, self.bx, self.by, self.bz)
        blur = F.conv3d(u.unsqueeze(1), self.blur_kernel, padding=1).squeeze(1)
        z = (1 - self.blur) * u + self.blur * blur
        return (z + self.cubic * z**3 + self.sinc * torch.sin(2 * z)).reshape(x_flat.shape[0], -1)


true_model = NonlinearHeat3D(cfg).to(device)
print(f"true model dimension: R^{cfg.dim} -> R^{cfg.dim}")


def build_dataset(c: Cfg, tm: nn.Module, seed: int = 7):
    torch.manual_seed(seed)

    n_total = c.n_train + c.n_val + c.n_test

    gx = torch.linspace(0, 1, c.nx + 2)[1:-1]
    gy = torch.linspace(0, 1, c.ny + 2)[1:-1]
    gz = torch.linspace(0, 1, c.nz + 2)[1:-1]

    kx = torch.arange(1, c.modes_x + 1, dtype=torch.float32)
    ky = torch.arange(1, c.modes_y + 1, dtype=torch.float32)
    kz = torch.arange(1, c.modes_z + 1, dtype=torch.float32)

    bx = torch.sin(math.pi * gx[:, None] * kx[None, :])
    bx = bx / bx.norm(dim=0, keepdim=True)

    by = torch.sin(math.pi * gy[:, None] * ky[None, :])
    by = by / by.norm(dim=0, keepdim=True)

    bz = torch.sin(math.pi * gz[:, None] * kz[None, :])
    bz = bz / bz.norm(dim=0, keepdim=True)

    coeff = torch.randn(n_total, c.modes_x, c.modes_y, c.modes_z)
    coeff = coeff * (1 / torch.arange(1, c.modes_x + 1, dtype=torch.float32))[None, :, None, None]
    coeff = coeff * (1 / torch.arange(1, c.modes_y + 1, dtype=torch.float32))[None, None, :, None]
    coeff = coeff * (1 / torch.arange(1, c.modes_z + 1, dtype=torch.float32))[None, None, None, :]

    x = torch.einsum("bijk,xi,yj,zk->bxyz", coeff, bx, by, bz)

    xx = gx[:, None, None].repeat(1, c.ny, c.nz)
    yy = gy[None, :, None].repeat(c.nx, 1, c.nz)
    zz = gz[None, None, :].repeat(c.nx, c.ny, 1)

    for b in range(n_total):
        cx = float(torch.rand(1) * 0.4 + 0.3)
        cy = float(torch.rand(1) * 0.4 + 0.3)
        cz = float(torch.rand(1) * 0.4 + 0.3)
        sx = float(torch.rand(1) * 0.04 + 0.1)
        sy = float(torch.rand(1) * 0.04 + 0.1)
        sz = float(torch.rand(1) * 0.04 + 0.1)
        amp = float(torch.rand(1) * 0.08 + 0.02)
        x[b] += amp * torch.exp(
            -(
                (xx - cx) ** 2 / (2 * sx**2)
                + (yy - cy) ** 2 / (2 * sy**2)
                + (zz - cz) ** 2 / (2 * sz**2)
            )
        )

    x_flat = x.reshape(n_total, -1)
    x_min = x_flat.min(1, keepdim=True).values
    x_max = x_flat.max(1, keepdim=True).values
    x_flat = c.x_low + (c.x_high - c.x_low) * (x_flat - x_min) / (x_max - x_min + 1e-8)

    with torch.no_grad():
        y_clean = tm(x_flat.to(device)).cpu()

    y_noisy = y_clean + c.noise_std * torch.randn_like(y_clean)

    nt, nv = c.n_train, c.n_val
    y_mean = y_clean[:nt].mean(0, keepdim=True)
    y_std = y_clean[:nt].std(0, keepdim=True).clamp(1e-6)

    def normalize(y: torch.Tensor) -> torch.Tensor:
        return (y - y_mean) / y_std

    return {
        "x_train": x_flat[:nt],
        "x_val": x_flat[nt : nt + nv],
        "x_test": x_flat[nt + nv :],
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


model = DeceptronCNN3D(
    nx=cfg.nx,
    ny=cfg.ny,
    nz=cfg.nz,
    hidden_channels=cfg.hidden_channels,
    negative_slope=cfg.negative_slope,
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
    cycle_weight=0.15,
    probe_jcp_weight=0.35,
    jcp_num_probes_train=1,
    jcp_num_probes_eval=2,
    jcp_batch_subsample=6,
    y_tilde_noise=0.005,
    gradient_clip=5.0,
    eval_subset_rjcp=24,
    use_cosine_lr=True,
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
    mlp_extra_losses=False,
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
    mlp_extra_losses=False,
)

rjcp_plus = estimate_rjcp_dataset(model_plus, x_val, num_probes=2, max_samples=24)
rjcp_minus = estimate_rjcp_dataset(model_minus, x_val, num_probes=2, max_samples=24)
print(f"+JCP RJCP = {rjcp_plus:.4f}")
print(f"-JCP RJCP = {rjcp_minus:.4f}")
print(f"ratio = {rjcp_minus / rjcp_plus:.2f}x")


def _f_wrap(model: nn.Module):
    return lambda x: model.f(x.reshape(1, -1)).reshape(-1)


def _g_wrap(model: nn.Module):
    return lambda y: model.g(y.reshape(1, -1)).reshape(-1)


def solve_dipg_3d(model: nn.Module, y_star: torch.Tensor, x0: torch.Tensor, s: SolverConfig):
    model.eval()
    dev = next(model.parameters()).device

    x = x0.clone().to(dev).reshape(-1)
    y_star = y_star.to(dev).reshape(-1)
    f = _f_wrap(model)
    g = _g_wrap(model)

    lo, hi = s.x_low, s.x_high
    accepted = 0
    t0 = time.perf_counter()

    for it in range(s.max_iterations):
        _, _, resid_t = _phi_grad(f, x, y_star)
        if torch.sqrt(torch.mean(resid_t**2)).item() <= s.tolerance_eps:
            break

        alpha = s.dipg_step_size
        ok = False
        y_t = f(x)

        for _ in range(s.max_backtracking_steps + 1):
            p = g(y_t - alpha * resid_t) - x
            x_trial, phi_trial, rhs = _armijo_accept(f, x, p, y_star, s.rho, s.armijo_c, lo, hi)
            if phi_trial <= rhs:
                x = x_trial.detach().reshape(-1)
                ok = True
                accepted += 1
                break
            alpha *= 0.5

        if not ok:
            break

    final_residual = _normalized_residual(f, x, y_star)
    final_rjcp = float(single_sample_probe_jcp(model, x.detach().clone().requires_grad_(True), 1).detach())

    return {
        "x_hat": x.detach().cpu(),
        "iters": it + 1,
        "success": float(final_residual <= s.tolerance_eps),
        "final_residual": final_residual,
        "accepted_frac": accepted / max(it + 1, 1),
        "time_sec": time.perf_counter() - t0,
        "final_rjcp": final_rjcp,
    }


def solve_baseline_3d(solver_fn, model: nn.Module, y_star: torch.Tensor, x0: torch.Tensor, s: SolverConfig):
    f = _f_wrap(model)
    dev = next(model.parameters()).device
    return solver_fn(f, y_star.to(dev).reshape(-1), x0.to(dev).reshape(-1), s)


solver_cfg = SolverConfig(
    max_iterations=80,
    tolerance_eps=0.38,
    armijo_c=1e-4,
    max_backtracking_steps=8,
    rho=0.4,
    x_low=cfg.x_low,
    x_high=cfg.x_high,
    dipg_step_size=1.0,
    gn_step_size=1.0,
    lm_step_size=1.0,
    lm_damping=1e-2,
    num_probes_eval=1,
)


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.cpu() - b.cpu()) ** 2)))


def evaluate(mdl: nn.Module, label: str) -> pd.DataFrame:
    x_test = data["x_test"]
    y_test = data["y_test_norm"]

    methods = {
        "D-IPG": lambda y, x0: solve_dipg_3d(mdl, y, x0, solver_cfg),
        "GD": lambda y, x0: solve_baseline_3d(solve_gradient_descent, mdl, y, x0, solver_cfg),
        "GN": lambda y, x0: solve_baseline_3d(solve_gauss_newton, mdl, y, x0, solver_cfg),
        "LM": lambda y, x0: solve_baseline_3d(solve_levenberg_marquardt, mdl, y, x0, solver_cfg),
    }

    rows = []
    for method_name, method_fn in methods.items():
        per = [method_fn(y_test[i], torch.zeros(cfg.dim)) for i in range(x_test.shape[0])]
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

results.to_csv("heat3d_main_results.csv", index=False)
torch.save(model_plus.state_dict(), "heat3d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "heat3d_minus_jcp.pt")
