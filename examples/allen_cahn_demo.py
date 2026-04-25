"""
Allen–Cahn 2D benchmark using the deceptron package.

From the repository root:
    pip install -e .

Or install directly from GitHub:
    pip install git+https://github.com/AadityaKachhadiya/deceptron.git

Run with:
    python examples/allen_cahn_demo.py
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.data import DataLoader, TensorDataset

from deceptron import (
    TrainConfig,
    train_forward,
    train_reverse,
    train_reverse_jcp,
    estimate_rjcp_dataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


@dataclass
class Cfg:
    nx: int = 32
    ny: int = 32
    x_channels: int = 1
    y_channels: int = 1

    epsilon: float = 0.04
    final_time: float = 0.12
    dt: float = 0.002
    noise_std: float = 0.002

    n_train: int = 2000
    n_val: int = 240
    n_test: int = 80

    hidden_channels: int = 48
    num_res_blocks: int = 4
    negative_slope: float = 0.10
    batch_size: int = 64

    max_iterations: int = 80
    tolerance_eps: float = 0.12
    armijo_c: float = 1e-4
    rho: float = 0.4
    max_backtracking_steps: int = 8

    dipg_step_size: float = 1.0
    gn_step_size: float = 1.0

    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 80
    lbfgs_history_size: int = 20

    cg_max_iter: int = 30
    cg_tol: float = 1e-6

    success_rmse_threshold: float = 0.10

    probe_jcp_weight: float = 0.01
    jcp_num_probes_train: int = 1
    jcp_num_probes_eval: int = 2
    jcp_batch_subsample: int = 8
    recon_guard_factor: float = 1.005
    y_tilde_noise: float = 0.01


cfg = Cfg()


class _Res(nn.Module):
    def __init__(self, ch: int, negative_slope: float):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular")
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.leaky_relu(self.c1(x), self.negative_slope)
        y = self.c2(y)
        return F.leaky_relu(x + y, self.negative_slope)


class DeceptronAC(nn.Module):
    def __init__(self, c: Cfg):
        super().__init__()
        h = c.hidden_channels
        ns = c.negative_slope

        blocks_f = [
            nn.Conv2d(c.x_channels, h, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(ns),
        ]
        blocks_f += [_Res(h, ns) for _ in range(c.num_res_blocks)]
        blocks_f += [nn.Conv2d(h, c.y_channels, 3, padding=1, padding_mode="circular")]

        blocks_r = [
            nn.Conv2d(c.y_channels, h, 3, padding=1, padding_mode="circular"),
            nn.LeakyReLU(ns),
        ]
        blocks_r += [_Res(h, ns) for _ in range(c.num_res_blocks)]
        blocks_r += [nn.Conv2d(h, 1, 3, padding=1, padding_mode="circular")]

        self.forward_map = nn.Sequential(*blocks_f)
        self.reverse_map = nn.Sequential(*blocks_r)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.reverse_map(y))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


def _pde(u0: np.ndarray, c: Cfg) -> np.ndarray:
    u = u0.astype(np.float64)

    kx = 2 * np.pi * np.fft.fftfreq(c.nx, 1.0 / c.nx)
    ky = 2 * np.pi * np.fft.fftfreq(c.ny, 1.0 / c.ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    denom = 1.0 + c.dt * c.epsilon**2 * (KX**2 + KY**2)

    for _ in range(int(round(c.final_time / c.dt))):
        u = np.fft.ifft2(np.fft.fft2(u + c.dt * (u - u**3)) / denom).real

    return np.clip(u, -1.0, 1.0).astype(np.float32)


def _ic(nx: int, ny: int) -> np.ndarray:
    yy, xx = np.mgrid[0:nx, 0:ny]
    xx = (xx + 0.5) / ny
    yy = (yy + 0.5) / nx

    field = -np.ones((nx, ny), dtype=np.float32)

    for _ in range(np.random.randint(2, 6)):
        cx, cy = np.random.uniform(0.15, 0.85), np.random.uniform(0.15, 0.85)
        r = np.random.uniform(0.05, 0.16)
        field[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = 1.0

    field = field + 0.05 * np.random.randn(nx, ny).astype(np.float32)
    return np.clip(field, -1.0, 1.0)


def build_dataset(c: Cfg, seed: int = 7):
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_total = c.n_train + c.n_val + c.n_test
    xs = [_ic(c.nx, c.ny)[None] for _ in range(n_total)]
    ys = [_pde(xs[i][0], c)[None] for i in range(n_total)]

    x_all = torch.tensor(np.stack(xs), dtype=torch.float32)
    y_clean = torch.tensor(np.stack(ys), dtype=torch.float32)
    y_noisy = torch.clamp(y_clean + c.noise_std * torch.randn_like(y_clean), -1.0, 1.0)

    nt, nv = c.n_train, c.n_val
    y_mean = y_clean[:nt].mean(dim=(0, 2, 3), keepdim=True)
    y_std = y_clean[:nt].std(dim=(0, 2, 3), keepdim=True).clamp(1e-6)

    def normalize(y: torch.Tensor) -> torch.Tensor:
        return (y - y_mean) / y_std

    return {
        "x_train": x_all[:nt],
        "x_val": x_all[nt : nt + nv],
        "x_test": x_all[nt + nv :],
        "y_train_norm": normalize(y_clean[:nt]),
        "y_val_norm": normalize(y_clean[nt : nt + nv]),
        "y_test_norm": normalize(y_noisy[nt + nv :]),
    }


print("building dataset...")
data = build_dataset(cfg, seed=7)

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


model = DeceptronAC(cfg).to(device)

train_cfg = TrainConfig(
    forward_epochs=120,
    forward_lr=2e-3,
    forward_wd=1e-6,
    reverse_epochs=80,
    reverse_lr=1e-3,
    reverse_wd=1e-6,
    jcp_epochs=40,
    jcp_lr=5e-4,
    jcp_wd=1e-6,
    reconstruction_weight=1.0,
    cycle_weight=0.05,
    probe_jcp_weight=cfg.probe_jcp_weight,
    jcp_num_probes_train=cfg.jcp_num_probes_train,
    jcp_num_probes_eval=cfg.jcp_num_probes_eval,
    jcp_batch_subsample=cfg.jcp_batch_subsample,
    y_tilde_noise=cfg.y_tilde_noise,
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


def _phi(f, x, y):
    return 0.5 * torch.mean((f(x) - y) ** 2)


def _phi_g(f, x, y):
    xr = x.detach().clone().requires_grad_(True)
    resid = f(xr) - y
    grad = torch.autograd.grad(0.5 * torch.mean(resid * resid), xr)[0]
    return grad.detach(), resid.detach()


def _arm(f, x, p, y):
    grad, _ = _phi_g(f, x, y)
    phi_t = _phi(f, x, y)
    x_trial = torch.clamp((1 - cfg.rho) * x + cfg.rho * (x + p), -1.0, 1.0)
    return x_trial, _phi(f, x_trial, y), phi_t + cfg.armijo_c * cfg.rho * torch.sum(grad * p)


def _cg(matvec, b):
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs = r.dot(r)

    for _ in range(cfg.cg_max_iter):
        Ap = matvec(p)
        alpha = rs / p.dot(Ap).clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r.dot(r)
        if rs_new.sqrt() < cfg.cg_tol:
            break
        p = r + (rs_new / rs.clamp_min(1e-12)) * p
        rs = rs_new

    return x


def solve_dipg_ac(model: nn.Module, y_star: torch.Tensor, x0: torch.Tensor):
    model.eval()
    x = x0.clone().to(device)
    y_star = y_star.to(device)
    t0 = time.perf_counter()

    for it in range(cfg.max_iterations):
        _, resid = _phi_g(model.f, x, y_star)
        if torch.sqrt(torch.mean(resid**2)).item() <= cfg.tolerance_eps:
            break

        alpha = cfg.dipg_step_size
        accepted = False
        y_t = model.f(x)

        for _ in range(cfg.max_backtracking_steps + 1):
            x_trial, phi_trial, rhs = _arm(model.f, x, model.g(y_t - alpha * resid) - x, y_star)
            if phi_trial <= rhs:
                x = x_trial.detach()
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break

    final_residual = float(torch.sqrt(torch.mean((model.f(x) - y_star) ** 2)))
    return {
        "x_hat": x.detach().cpu(),
        "iters": it + 1,
        "success": float(final_residual <= cfg.tolerance_eps),
        "time_sec": time.perf_counter() - t0,
    }


def solve_gn_ac(f, y_star: torch.Tensor, x0: torch.Tensor):
    x = x0.clone().to(device)
    y_star = y_star.to(device)
    t0 = time.perf_counter()

    for it in range(cfg.max_iterations):
        _, resid = _phi_g(f, x, y_star)
        if torch.sqrt(torch.mean(resid**2)).item() <= cfg.tolerance_eps:
            break

        xr = x.detach().clone().requires_grad_(True)
        rhs = -torch.autograd.grad((f(xr) * resid.detach()).sum(), xr)[0] / resid.numel()

        def matvec(v):
            _, jv = jvp(f, (x.detach(),), (v,))
            x_req = x.detach().clone().requires_grad_(True)
            return torch.autograd.grad((f(x_req) * jv.detach()).sum(), x_req)[0] / jv.numel()

        dx = _cg(matvec, rhs)
        alpha = cfg.gn_step_size
        accepted = False

        for _ in range(cfg.max_backtracking_steps + 1):
            x_trial, phi_trial, rhs_armijo = _arm(f, x, alpha * dx, y_star)
            if phi_trial <= rhs_armijo:
                x = x_trial.detach()
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            break

    final_residual = float(torch.sqrt(torch.mean((f(x) - y_star) ** 2)))
    return {
        "x_hat": x.detach().cpu(),
        "iters": it + 1,
        "success": float(final_residual <= cfg.tolerance_eps),
        "time_sec": time.perf_counter() - t0,
    }


def solve_lbfgs_ac(f, y_star: torch.Tensor, x0: torch.Tensor):
    x = nn.Parameter(x0.clone().to(device).contiguous())
    y_star = y_star.to(device).contiguous()

    opt = torch.optim.LBFGS(
        [x],
        lr=cfg.lbfgs_lr,
        max_iter=cfg.lbfgs_max_iter,
        history_size=cfg.lbfgs_history_size,
        line_search_fn="strong_wolfe",
    )

    n_iters = {"n": 0}
    t0 = time.perf_counter()

    def closure():
        opt.zero_grad()
        x_clamped = torch.clamp(x, -1.0, 1.0).contiguous()
        loss = 0.5 * torch.mean((f(x_clamped) - y_star) ** 2)
        loss.backward()
        if x.grad is not None:
            x.grad = x.grad.contiguous()
        n_iters["n"] += 1
        return loss

    opt.step(closure)
    x_final = torch.clamp(x.detach(), -1.0, 1.0)
    final_residual = float(torch.sqrt(torch.mean((f(x_final) - y_star) ** 2)))

    return {
        "x_hat": x_final.cpu(),
        "iters": n_iters["n"],
        "success": float(final_residual <= cfg.tolerance_eps),
        "time_sec": time.perf_counter() - t0,
    }


def rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((a.cpu() - b.cpu()) ** 2)))


def evaluate(model_plus: nn.Module, model_minus: nn.Module) -> pd.DataFrame:
    x_test = data["x_test"]
    y_test = data["y_test_norm"]

    specs = [
        ("+JCP", "D-IPG", lambda y, x0: solve_dipg_ac(model_plus, y, x0)),
        ("-JCP", "D-IPG", lambda y, x0: solve_dipg_ac(model_minus, y, x0)),
        ("Base", "GN", lambda y, x0: solve_gn_ac(model_plus.f, y, x0)),
        ("Base", "L-BFGS", lambda y, x0: solve_lbfgs_ac(model_plus.f, y, x0)),
    ]

    rows = []
    for ablation, method_name, method_fn in specs:
        per = [
            method_fn(y_test[i : i + 1].to(device), torch.zeros_like(x_test[i : i + 1]).to(device))
            for i in range(x_test.shape[0])
        ]
        rmses = [rmse(p["x_hat"], x_test[i : i + 1]) for i, p in enumerate(per)]

        rows.append(
            {
                "ablation": ablation,
                "method": method_name,
                "success_rate": float(np.mean([r <= cfg.success_rmse_threshold for r in rmses])),
                "median_iters": float(np.median([p["iters"] for p in per])),
                "mean_rmse_x": float(np.mean(rmses)),
                "mean_time_sec": float(np.mean([p["time_sec"] for p in per])),
            }
        )
        print(f"completed: {ablation} | {method_name}")

    return pd.DataFrame(rows)


print("evaluating")
results = evaluate(model_plus, model_minus)

print("\n" + "=" * 66)
print(f"{'Ablation':8} {'Method':8} {'SR':>7} {'Med.It':>7} {'RMSE':>8} {'Time':>9}")
print("=" * 66)
for _, row in results.iterrows():
    print(
        f"{row['ablation']:8} "
        f"{row['method']:8} "
        f"{row['success_rate']:>7.1%} "
        f"{row['median_iters']:>7.0f} "
        f"{row['mean_rmse_x']:>8.4f} "
        f"{row['mean_time_sec']:>8.4f}s"
    )
print("=" * 66)

results.to_csv("allen_cahn2d_main_results.csv", index=False)
torch.save(model_plus.state_dict(), "allen_cahn2d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "allen_cahn2d_minus_jcp.pt")
