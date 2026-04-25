# =============================================================
# Deceptron — Heat-1D Demo
# Exact protocol from Heat1D.py, using the package
# =============================================================

# ── CELL 1: Install ───────────────────────────────────────────
import subprocess, sys, os, importlib

if not os.path.exists("./deceptron_pkg/setup.py"):
    subprocess.run(["unzip", "-q", "deceptron_pkg.zip"], check=True)

subprocess.run([
    sys.executable, "-m", "pip", "install", "-e", "./deceptron_pkg",
    "--force-reinstall", "-q"
], check=True)
importlib.invalidate_caches()
src = os.path.abspath("./deceptron_pkg")
if src not in sys.path:
    sys.path.insert(0, src)
for mod in list(sys.modules):
    if "deceptron" in mod:
        del sys.modules[mod]

import deceptron
print(f"✓  deceptron v{deceptron.__version__}  from {deceptron.__file__}")


# ── CELL 2: Imports ───────────────────────────────────────────
import math, time, random, copy
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

from deceptron import (
    DeceptronMLP,
    TrainConfig, train_forward, train_reverse, train_reverse_jcp,
    SolverConfig, solve_dipg, solve_gradient_descent,
    solve_gauss_newton, solve_levenberg_marquardt,
    estimate_rjcp_dataset,
)

# ── No global seed, no global device ─────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  |  torch: {torch.__version__}")


# ── CELL 3: Problem config (mirrors Heat1DExperimentConfig) ──
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
print("✓ Config")


# ── CELL 4: True model (exact from Heat1D.py) ─────────────────
class NonlinearHeat1DTrueModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        grid  = torch.linspace(0.0, 1.0, c.n_grid + 2)[1:-1]
        k     = torch.arange(1, c.n_grid + 1, dtype=torch.float32)
        basis = torch.sin(math.pi * grid[:, None] * k[None, :])
        basis = basis * math.sqrt(2.0 / (c.n_grid + 1))
        decay = torch.exp(-c.diffusivity * (math.pi * k) ** 2 * c.t_final)
        mix   = torch.eye(c.n_grid)
        for i in range(c.n_grid):
            if i > 0:              mix[i, i-1] = c.obs_mix_strength
            if i < c.n_grid - 1:  mix[i, i+1] = c.obs_mix_strength
        mix = mix / mix.sum(dim=1, keepdim=True)
        self.cubic    = c.obs_nonlinear_cubic
        self.sin_coef = c.obs_nonlinear_sin
        self.register_buffer("basis", basis)
        self.register_buffer("decay", decay)
        self.register_buffer("mix",   mix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c   = x @ self.basis
        u_t = (c * self.decay) @ self.basis.T
        z   = u_t @ self.mix.T
        return z + self.cubic * z ** 3 + self.sin_coef * torch.sin(3.0 * z)

true_model = NonlinearHeat1DTrueModel(cfg).to(device)
print(f"✓ True model: R^{cfg.n_grid} → R^{cfg.n_grid}")


# ── CELL 5: Dataset (exact from Heat1D.py) ────────────────────
def sample_heat1d(n, c, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    grid  = torch.linspace(0.0, 1.0, c.n_grid + 2)[1:-1]
    k     = torch.arange(1, c.basis_keep + 1, dtype=torch.float32)
    basis = torch.sin(math.pi * grid[:, None] * k[None, :])
    basis = basis / basis.norm(dim=0, keepdim=True)
    coeff = torch.randn(n, c.basis_keep)
    coeff[:, 0] += 0.25 * torch.randn(n)
    coeff[:, 1:] *= 1.0 / torch.arange(2, c.basis_keep + 1, dtype=torch.float32)
    x     = coeff @ basis.T
    xmin  = x.min(1, keepdim=True).values
    xmax  = x.max(1, keepdim=True).values
    x     = (x - xmin) / (xmax - xmin + 1e-8)
    return c.x_low + (c.x_high - c.x_low) * x

def build_dataset(c, tm, seed=7):
    n = c.n_train + c.n_val + c.n_test
    x_all = sample_heat1d(n, c, seed=seed)
    with torch.no_grad():
        y_clean = tm(x_all.to(device)).cpu()
    y_noisy = y_clean + c.noise_std * torch.randn_like(y_clean)
    nt, nv  = c.n_train, c.n_val
    ym = y_clean[:nt].mean(0, keepdim=True)
    ys = y_clean[:nt].std(0, keepdim=True).clamp(1e-6)
    def nm(y): return (y - ym) / ys
    return {
        "x_train": x_all[:nt],
        "x_val":   x_all[nt:nt+nv],
        "x_test":  x_all[nt+nv:],
        "y_train_norm": nm(y_clean[:nt]),
        "y_val_norm":   nm(y_clean[nt:nt+nv]),
        "y_test_norm":  nm(y_noisy[nt+nv:]),  # noisy targets for eval (exact as paper)
        "y_mean": ym, "y_std": ys,
    }

# seed=7 matches paper seed; change to verify results are stable
data = build_dataset(cfg, true_model, seed=7)

BS = cfg.batch_size
def make_loader(x, y, shuffle):
    return DataLoader(TensorDataset(x, y), batch_size=BS, shuffle=shuffle)

tr_loader  = make_loader(data["x_train"], data["y_train_norm"], shuffle=True)
val_loader = make_loader(data["x_val"],   data["y_val_norm"],   shuffle=False)
x_val      = data["x_val"].to(device)

print(f"✓ Dataset: train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}")


# ── CELL 6: TrainConfig (exact Heat-1D paper values) ─────────
train_cfg = TrainConfig(
    forward_epochs=140, forward_lr=2e-3, forward_wd=1e-6,
    reverse_epochs=100, reverse_lr=2e-3, reverse_wd=1e-6,
    jcp_epochs=120,     jcp_lr=1e-3,     jcp_wd=1e-6,
    reconstruction_weight=1.0,
    cycle_weight=0.25,          # Heat-3D uses 0.15
    bias_tie_weight=5e-4,       # MLP only
    composition_weight=1e-3,    # MLP only
    probe_jcp_weight=1.0,       # Heat-3D uses 0.35
    jcp_num_probes_train=2,     # Heat-3D uses 1
    jcp_num_probes_eval=4,      # Heat-3D uses 2
    jcp_batch_subsample=16,     # Heat-3D uses 6
    y_tilde_noise=0.02,         # Heat-3D uses 0.005
    gradient_clip=5.0,
    eval_subset_rjcp=64,
    use_cosine_lr=False,        # Heat-3D uses True
)
print("✓ TrainConfig")


# ── CELL 7: Stage 1 — forward map ────────────────────────────
model = DeceptronMLP(
    dim=cfg.n_grid,
    negative_slope=cfg.negative_slope,
    hidden_multiplier=cfg.reverse_hidden_multiplier,
).to(device)

print("Stage 1: Training forward map...")
model_s1 = train_forward(
    model, tr_loader, val_loader, x_val,
    train_cfg, device, verbose=True,
)
print(f"✓ Stage 1 done")


# ── CELL 8: Stage 2 — reverse (no JCP) ───────────────────────
print("\nStage 2: Reverse map (no JCP)...")
model_s2 = train_reverse(
    model_s1, tr_loader, val_loader, x_val,
    train_cfg, device, verbose=True,
)
print(f"✓ Stage 2 done")


# ── CELL 9: Stage 3 — +JCP and -JCP ─────────────────────────
print("\nStage 3 (+JCP)...")
model_plus = train_reverse_jcp(
    model_s2, tr_loader, val_loader, x_val,
    train_cfg, device,
    use_jcp=True,
    mlp_extra_losses=True,   # bias_tie + composition (MLP protocol)
    verbose=True,
)

print("\nStage 3 (-JCP, ablation)...")
model_minus = train_reverse_jcp(
    model_s2, tr_loader, val_loader, x_val,
    train_cfg, device,
    use_jcp=False,
    mlp_extra_losses=True,
    verbose=True,
)

rjcp_p = estimate_rjcp_dataset(model_plus,  x_val, num_probes=4, max_samples=64)
rjcp_m = estimate_rjcp_dataset(model_minus, x_val, num_probes=4, max_samples=64)
print(f"\n+JCP RJCP = {rjcp_p:.4f}")
print(f"-JCP RJCP = {rjcp_m:.4f}")
print(f"Ratio     = {rjcp_m/rjcp_p:.2f}×  (paper: ~3.6×)")


# ── CELL 10: SolverConfig ─────────────────────────────────────
solver_cfg = SolverConfig(
    max_iterations=120,
    tolerance_eps=0.30,         # Heat-3D uses 0.38
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
print("✓ SolverConfig")


# ── CELL 11: Evaluate (exact from Heat1D.py) ─────────────────
def rmse_x(a, b):
    return float(torch.sqrt(torch.mean((a.cpu() - b.cpu()) ** 2)))

def evaluate_all(mdl, label, data, solver_cfg):
    """
    Exact evaluate_all_methods_on_test from Heat1D.py.
    Uses y_noisy_test_norm (same as paper).
    x0 = zeros_like(x_true) (same as paper).
    """
    x_test      = data["x_test"]
    y_test_norm = data["y_test_norm"]   # y_noisy_test_norm in paper
    dev         = next(mdl.parameters()).device

    methods = {
        "D-IPG":            lambda y, x0: solve_dipg(mdl, y, x0, solver_cfg),
        "Gradient Descent": lambda y, x0: solve_gradient_descent(mdl.f, y, x0, solver_cfg),
        "GN":               lambda y, x0: solve_gauss_newton(mdl.f, y, x0, solver_cfg),
        "LM":               lambda y, x0: solve_levenberg_marquardt(mdl.f, y, x0, solver_cfg),
    }
    rows = []
    for mname, mfn in methods.items():
        per = []
        for i in range(x_test.shape[0]):
            y_star = y_test_norm[i].to(dev)
            x_true = x_test[i]
            x0     = torch.zeros_like(x_true)   # zeros, same as paper
            res    = mfn(y_star, x0)
            per.append({
                "rmse_x": rmse_x(res["x_hat"], x_true),
                **{k: v for k, v in res.items() if k != "x_hat"},
            })
        df = pd.DataFrame(per)
        rows.append({
            "ablation":         label,
            "method":           mname,
            "num_instances":    len(df),
            "success_rate":     df["success"].mean(),
            "mean_iters":       df["iters"].mean(),
            "median_iters":     df["iters"].median(),
            "mean_rmse_x":      df["rmse_x"].mean(),
            "mean_time_sec":    df["time_sec"].mean(),
            "mean_final_residual": df["final_residual"].mean(),
        })
        if "final_rjcp" in df.columns:
            rows[-1]["mean_final_rjcp"] = df["final_rjcp"].mean()
        if "mean_cosine_with_neg_grad" in df.columns:
            rows[-1]["mean_cosine_with_neg_grad"] = df["mean_cosine_with_neg_grad"].mean()
        print(f"  done: {label} | {mname}")
    return pd.DataFrame(rows)

print("Evaluating +JCP model...")
res_plus  = evaluate_all(model_plus,  "+JCP", data, solver_cfg)
print("\nEvaluating -JCP model...")
res_minus = evaluate_all(model_minus, "-JCP", data, solver_cfg)
all_results = pd.concat([res_plus, res_minus], ignore_index=True)

print("\n" + "=" * 70)
print(f"{'Ablation':8} {'Method':18} {'SR':>7} {'Med.It':>7} "
      f"{'RMSE':>8} {'Time':>9}")
print("=" * 70)
for _, r in all_results.iterrows():
    print(f"{r['ablation']:8} {r['method']:18} {r['success_rate']:>7.1%} "
          f"{r['median_iters']:>7.1f} {r['mean_rmse_x']:>8.4f} "
          f"{r['mean_time_sec']:>9.4f}s")
print("=" * 70)

print("\nExpected (from paper seed=7):")
print("  +JCP D-IPG  SR≈24%  med_iters≈40  RMSE≈0.27")
print("  +JCP LM     SR≈35%  med_iters≈7   RMSE≈0.29")
print("  +JCP GN     SR≈1%   med_iters≈2   RMSE≈0.53")


# ── CELL 12: Save ─────────────────────────────────────────────
all_results.to_csv("heat1d_main_results.csv", index=False)
torch.save(model_s1.state_dict(),    "heat1d_stage1.pt")
torch.save(model_s2.state_dict(),    "heat1d_stage2.pt")
torch.save(model_plus.state_dict(),  "heat1d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "heat1d_minus_jcp.pt")
print("✓ Saved results and weights")
print("  heat1d_main_results.csv")
print("  heat1d_stage1.pt, heat1d_stage2.pt")
print("  heat1d_plus_jcp.pt, heat1d_minus_jcp.pt")
