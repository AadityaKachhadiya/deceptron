# =============================================================
# Deceptron — Heat-3D Demo
# Exact protocol from Heat3D.py using the deceptron package
#
# Key differences from Heat-1D:
#   arch          : CNN (not MLP)
#   cycle_weight  : 0.15  (not 0.25)
#   probe_jcp_weight: 0.35 (not 1.0)
#   jcp_probes_train: 1   (not 2)
#   jcp_probes_eval : 2   (not 4)
#   jcp_subsample : 6     (not 16)
#   y_tilde_noise : 0.005 (not 0.02)
#   use_cosine_lr : True  (not False)
#   mlp_extra_losses: False (no bias_tie/composition)
#   tolerance_eps : 0.38  (not 0.30)
#   max_iterations: 80    (not 120)
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
if src not in sys.path: sys.path.insert(0, src)
for mod in list(sys.modules):
    if "deceptron" in mod: del sys.modules[mod]
import deceptron
print(f"✓  deceptron v{deceptron.__version__}  from {deceptron.__file__}")


# ── CELL 2: Imports ───────────────────────────────────────────
import math, time, random, copy
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass

from deceptron import (
    DeceptronCNN3D,
    TrainConfig, train_forward, train_reverse, train_reverse_jcp,
    SolverConfig, solve_dipg, solve_gradient_descent,
    solve_gauss_newton, solve_levenberg_marquardt,
    estimate_rjcp_dataset,
)
from deceptron.jcp import single_sample_probe_jcp
from deceptron.solvers import _phi_grad, _normalized_residual, _full_jacobian, _solve_linear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  |  torch: {torch.__version__}")


# ── CELL 3: Config (exact Heat3DCNNConfig) ────────────────────
@dataclass
class Cfg:
    nx: int = 8;  ny: int = 8;  nz: int = 8;  dim: int = 512
    t_final: float = 0.012;     diffusivity: float = 0.10
    obs_blur_strength: float = 0.025
    obs_nonlinear_cubic: float = 0.020
    obs_nonlinear_sin: float = 0.008
    noise_std: float = 0.0015
    n_train: int = 4000;  n_val: int = 400;  n_test: int = 80
    modes_x: int = 3;  modes_y: int = 3;  modes_z: int = 3
    x_low: float = 0.0;  x_high: float = 1.0
    negative_slope: float = 0.10;  hidden_channels: int = 12
    batch_size: int = 32

cfg = Cfg()
print(f"✓ Config  dim={cfg.dim}  ({cfg.nx}×{cfg.ny}×{cfg.nz})")


# ── CELL 4: True model (exact from Heat3D.py) ─────────────────
class NonlinearHeat3DTrueModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.nx, self.ny, self.nz = c.nx, c.ny, c.nz
        self.obs_blur_strength    = c.obs_blur_strength
        self.obs_nonlinear_cubic  = c.obs_nonlinear_cubic
        self.obs_nonlinear_sin    = c.obs_nonlinear_sin

        gx = torch.linspace(0.0, 1.0, c.nx+2)[1:-1]
        gy = torch.linspace(0.0, 1.0, c.ny+2)[1:-1]
        gz = torch.linspace(0.0, 1.0, c.nz+2)[1:-1]
        kx = torch.arange(1, c.nx+1, dtype=torch.float32)
        ky = torch.arange(1, c.ny+1, dtype=torch.float32)
        kz = torch.arange(1, c.nz+1, dtype=torch.float32)

        bx = torch.sin(math.pi*gx[:,None]*kx[None,:])*math.sqrt(2.0/(c.nx+1))
        by = torch.sin(math.pi*gy[:,None]*ky[None,:])*math.sqrt(2.0/(c.ny+1))
        bz = torch.sin(math.pi*gz[:,None]*kz[None,:])*math.sqrt(2.0/(c.nz+1))
        decay = torch.exp(-c.diffusivity*(
            (math.pi*kx[:,None,None])**2 +
            (math.pi*ky[None,:,None])**2 +
            (math.pi*kz[None,None,:])**2)*c.t_final)
        kernel = torch.zeros(3,3,3)
        kernel[1,1,1]=8.; kernel[0,1,1]=kernel[2,1,1]=1.
        kernel[1,0,1]=kernel[1,2,1]=1.; kernel[1,1,0]=kernel[1,1,2]=1.
        kernel = kernel/kernel.sum()

        self.register_buffer("bx",bx); self.register_buffer("by",by)
        self.register_buffer("bz",bz); self.register_buffer("decay",decay)
        self.register_buffer("blur_kernel",kernel.view(1,1,3,3,3))

    def heat_semigroup(self, x_flat):
        x = x_flat.view(-1,self.nx,self.ny,self.nz)
        c = torch.einsum("bxyz,xi,yj,zk->bijk",x,self.bx,self.by,self.bz)
        return torch.einsum("bijk,ijk,xi,yj,zk->bxyz",c,self.decay,self.bx,self.by,self.bz)

    def forward(self, x_flat):
        u    = self.heat_semigroup(x_flat)
        blr  = F.conv3d(u.unsqueeze(1),self.blur_kernel,padding=1).squeeze(1)
        z    = (1-self.obs_blur_strength)*u + self.obs_blur_strength*blr
        y    = z + self.obs_nonlinear_cubic*z**3 + self.obs_nonlinear_sin*torch.sin(2.0*z)
        return y.reshape(y.shape[0],-1)

true_model = NonlinearHeat3DTrueModel(cfg).to(device)
print(f"✓ True model: R^{cfg.dim} → R^{cfg.dim}")


# ── CELL 5: Dataset (exact from Heat3D.py) ────────────────────
def sample_heat3d(n, c, seed=None):
    if seed is not None: torch.manual_seed(seed)
    gx=torch.linspace(0,1,c.nx+2)[1:-1]; gy=torch.linspace(0,1,c.ny+2)[1:-1]
    gz=torch.linspace(0,1,c.nz+2)[1:-1]
    kx=torch.arange(1,c.modes_x+1,dtype=torch.float32)
    ky=torch.arange(1,c.modes_y+1,dtype=torch.float32)
    kz=torch.arange(1,c.modes_z+1,dtype=torch.float32)
    bx=torch.sin(math.pi*gx[:,None]*kx[None,:]); bx=bx/bx.norm(0,keepdim=True)
    by=torch.sin(math.pi*gy[:,None]*ky[None,:]); by=by/by.norm(0,keepdim=True)
    bz=torch.sin(math.pi*gz[:,None]*kz[None,:]); bz=bz/bz.norm(0,keepdim=True)
    cf=torch.randn(n,c.modes_x,c.modes_y,c.modes_z)
    cf=cf*(1/torch.arange(1,c.modes_x+1,dtype=torch.float32))[None,:,None,None]
    cf=cf*(1/torch.arange(1,c.modes_y+1,dtype=torch.float32))[None,None,:,None]
    cf=cf*(1/torch.arange(1,c.modes_z+1,dtype=torch.float32))[None,None,None,:]
    x=torch.einsum("bijk,xi,yj,zk->bxyz",cf,bx,by,bz)
    xx=gx[:,None,None].repeat(1,c.ny,c.nz)
    yy=gy[None,:,None].repeat(c.nx,1,c.nz)
    zz=gz[None,None,:].repeat(c.nx,c.ny,1)
    for b in range(n):
        cx=float(torch.rand(1)*0.4+0.3); cy=float(torch.rand(1)*0.4+0.3)
        cz=float(torch.rand(1)*0.4+0.3); sx=float(torch.rand(1)*0.04+0.10)
        sy=float(torch.rand(1)*0.04+0.10); sz=float(torch.rand(1)*0.04+0.10)
        amp=float(torch.rand(1)*0.08+0.02)
        x[b]+=amp*torch.exp(-((xx-cx)**2/(2*sx**2)+(yy-cy)**2/(2*sy**2)+(zz-cz)**2/(2*sz**2)))
    xf=x.reshape(n,-1); xm=xf.min(1,keepdim=True).values; xM=xf.max(1,keepdim=True).values
    return c.x_low+(c.x_high-c.x_low)*(xf-xm)/(xM-xm+1e-8)

def build_dataset(c, tm, seed=7):
    n = c.n_train+c.n_val+c.n_test
    x_all = sample_heat3d(n, c, seed=seed)
    with torch.no_grad():
        y_clean = tm(x_all.to(device)).cpu()
    y_noisy = y_clean + c.noise_std*torch.randn_like(y_clean)
    nt,nv = c.n_train, c.n_val
    ym=y_clean[:nt].mean(0,keepdim=True); ys=y_clean[:nt].std(0,keepdim=True).clamp(1e-6)
    def nm(y): return (y-ym)/ys
    return {
        "x_train":      x_all[:nt],
        "x_val":        x_all[nt:nt+nv],
        "x_test":       x_all[nt+nv:],
        "y_train_norm": nm(y_clean[:nt]),
        "y_val_norm":   nm(y_clean[nt:nt+nv]),
        "y_test_norm":  nm(y_noisy[nt+nv:]),   # noisy targets for eval (exact as paper)
        "y_mean": ym, "y_std": ys,
    }

data = build_dataset(cfg, true_model, seed=7)
BS = cfg.batch_size
def make_loader(x,y,sh): return DataLoader(TensorDataset(x,y),batch_size=BS,shuffle=sh)
tr_loader  = make_loader(data["x_train"], data["y_train_norm"], True)
val_loader = make_loader(data["x_val"],   data["y_val_norm"],   False)
x_val = data["x_val"].to(device)
print(f"✓ Dataset: train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}")


# ── CELL 6: TrainConfig (exact Heat-3D paper values) ─────────
train_cfg = TrainConfig(
    forward_epochs=140, forward_lr=2e-3, forward_wd=1e-6,
    reverse_epochs=100, reverse_lr=2e-3, reverse_wd=1e-6,
    jcp_epochs=120,     jcp_lr=1e-3,     jcp_wd=1e-6,
    reconstruction_weight=1.0,
    cycle_weight=0.15,           # ← 0.15 for CNN (not 0.25)
    probe_jcp_weight=0.35,       # ← 0.35 for CNN (not 1.0)
    jcp_num_probes_train=1,      # ← 1 for CNN (not 2)
    jcp_num_probes_eval=2,       # ← 2 for CNN (not 4)
    jcp_batch_subsample=6,       # ← 6 for CNN (not 16)
    y_tilde_noise=0.005,         # ← 0.005 for CNN (not 0.02)
    gradient_clip=5.0,
    eval_subset_rjcp=24,         # ← 24 for CNN (not 64)
    use_cosine_lr=True,          # ← True for CNN (not False)
)
# Note: bias_tie_weight and composition_weight are in config but
# mlp_extra_losses=False below ensures they are NOT used for CNN
print("✓ TrainConfig (CNN/Heat-3D values)")


# ── CELL 7: Stage 1 ───────────────────────────────────────────
model = DeceptronCNN3D(
    nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
    hidden_channels=cfg.hidden_channels,
    negative_slope=cfg.negative_slope,
).to(device)

print("Stage 1: Training forward map...")
model_s1 = train_forward(
    model, tr_loader, val_loader, x_val,
    train_cfg, device, verbose=True,
)
print("✓ Stage 1 done")


# ── CELL 8: Stage 2 ───────────────────────────────────────────
print("\nStage 2: Reverse map (no JCP)...")
model_s2 = train_reverse(
    model_s1, tr_loader, val_loader, x_val,
    train_cfg, device, verbose=True,
)
print("✓ Stage 2 done")


# ── CELL 9: Stage 3 ───────────────────────────────────────────
print("\nStage 3 (+JCP)...")
model_plus = train_reverse_jcp(
    model_s2, tr_loader, val_loader, x_val,
    train_cfg, device,
    use_jcp=True,
    mlp_extra_losses=False,   # ← False for CNN (no bias_tie/composition)
    verbose=True,
)

print("\nStage 3 (-JCP, ablation)...")
model_minus = train_reverse_jcp(
    model_s2, tr_loader, val_loader, x_val,
    train_cfg, device,
    use_jcp=False,
    mlp_extra_losses=False,
    verbose=True,
)

rjcp_p = estimate_rjcp_dataset(model_plus,  x_val, num_probes=2, max_samples=24)
rjcp_m = estimate_rjcp_dataset(model_minus, x_val, num_probes=2, max_samples=24)
print(f"\n+JCP RJCP = {rjcp_p:.4f}")
print(f"-JCP RJCP = {rjcp_m:.4f}")
print(f"Ratio     = {rjcp_m/rjcp_p:.2f}×")


# ── CELL 10: CNN-aware solvers ────────────────────────────────
# CNN needs x.reshape(-1) and y_star.reshape(-1) consistently
# The package solvers already do .flatten() — but armijo_accept
# calls f(x) which for CNN needs x to have batch dim.
# Solution: wrap model.f to handle both batched and unbatched.

def make_f_batched(model):
    """Wrap model.f to always accept flat (d,) input."""
    def f(x_flat):
        x_flat = x_flat.reshape(-1)
        return model.f(x_flat.unsqueeze(0)).squeeze(0).reshape(-1)
    return f

def make_g_batched(model):
    """Wrap model.g to always accept flat (d,) input."""
    def g(y_flat):
        y_flat = y_flat.reshape(-1)
        return model.g(y_flat.unsqueeze(0)).squeeze(0).reshape(-1)
    return g


def solve_dipg_3d(model, y_star, x0, cfg_s):
    """D-IPG for CNN — exact from Heat3D.py with unsqueeze/reshape."""
    model.eval()
    dev    = next(model.parameters()).device
    x      = x0.clone().to(dev).reshape(-1)
    y_star = y_star.to(dev).reshape(-1)
    f      = make_f_batched(model)
    g      = make_g_batched(model)
    lo, hi = cfg_s.x_low, cfg_s.x_high

    acc=0; res_drop=0.; t0=time.perf_counter()
    for it in range(cfg_s.max_iterations):
        _, grad_t, resid_t = _phi_grad(f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t**2)).item()
        if nr <= cfg_s.tolerance_eps: break
        alpha, ok = cfg_s.dipg_step_size, False
        y_t = f(x)
        for _ in range(cfg_s.max_backtracking_steps+1):
            y_prop = y_t - alpha*resid_t
            x_prop = g(y_prop)
            p      = (x_prop - x).reshape(-1)
            from deceptron.solvers import _armijo_accept
            xt,pt,rhs = _armijo_accept(f, x, p, y_star, cfg_s.rho, cfg_s.armijo_c, lo, hi)
            if pt <= rhs:
                res_drop += (nr - _normalized_residual(f, xt, y_star))
                x=xt.detach().reshape(-1); ok=True; acc+=1; break
            alpha *= 0.5
        if not ok: break
    nr_fin = _normalized_residual(f, x, y_star)
    rjcp   = float(single_sample_probe_jcp(model, x.detach().clone().requires_grad_(True), 1).detach())
    return {"x_hat":x.detach().cpu(),"iters":it+1,
            "success":float(nr_fin<=cfg_s.tolerance_eps),
            "final_residual":nr_fin,"accepted_frac":acc/max(it+1,1),
            "time_sec":time.perf_counter()-t0,"final_rjcp":rjcp,
            "mean_residual_drop_per_accept":res_drop/max(acc,1)}


def solve_baseline_3d(solver_fn, model, y_star, x0, cfg_s):
    """GD/GN/LM for CNN — wraps f with unsqueeze/reshape."""
    f = make_f_batched(model)
    dev = next(model.parameters()).device
    return solver_fn(f, y_star.to(dev).reshape(-1),
                     x0.to(dev).reshape(-1), cfg_s)

print("✓ CNN-aware solvers ready")


# ── CELL 11: SolverConfig ─────────────────────────────────────
solver_cfg = SolverConfig(
    max_iterations=80,           # ← 80 for Heat-3D (not 120)
    tolerance_eps=0.38,          # ← 0.38 for Heat-3D (not 0.30)
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
print("✓ SolverConfig (Heat-3D values)")


# ── CELL 12: Evaluate ─────────────────────────────────────────
def rmse_x(a, b):
    return float(torch.sqrt(torch.mean((a.cpu()-b.cpu())**2)))

def evaluate_all_3d(mdl, label):
    x_test      = data["x_test"]
    y_test_norm = data["y_test_norm"]
    rows = []
    methods = {
        "D-IPG":            lambda y,x0: solve_dipg_3d(mdl, y, x0, solver_cfg),
        "Gradient Descent": lambda y,x0: solve_baseline_3d(solve_gradient_descent, mdl, y, x0, solver_cfg),
        "GN":               lambda y,x0: solve_baseline_3d(solve_gauss_newton,     mdl, y, x0, solver_cfg),
        "LM":               lambda y,x0: solve_baseline_3d(solve_levenberg_marquardt, mdl, y, x0, solver_cfg),
    }
    for mname, mfn in methods.items():
        per = []
        for i in range(x_test.shape[0]):
            dev    = next(mdl.parameters()).device
            y_star = y_test_norm[i]
            x_true = x_test[i]
            x0     = torch.zeros(cfg.dim)
            res    = mfn(y_star, x0)
            per.append({"rmse_x":rmse_x(res["x_hat"],x_true),
                        **{k:v for k,v in res.items() if k!="x_hat"}})
        df = pd.DataFrame(per)
        rows.append({
            "ablation":label,"method":mname,
            "success_rate":df["success"].mean(),
            "median_iters":df["iters"].median(),
            "mean_rmse_x":df["rmse_x"].mean(),
            "mean_time_sec":df["time_sec"].mean(),
        })
        print(f"  done: {label} | {mname}")
    return pd.DataFrame(rows)

print("Evaluating +JCP model...")
res_plus  = evaluate_all_3d(model_plus,  "+JCP")
print("Evaluating -JCP model...")
res_minus = evaluate_all_3d(model_minus, "-JCP")
all_results = pd.concat([res_plus, res_minus], ignore_index=True)

print("\n" + "="*70)
print(f"{'Ablation':8} {'Method':18} {'SR':>7} {'Med.It':>7} {'RMSE':>8} {'Time':>9}")
print("="*70)
for _,r in all_results.iterrows():
    print(f"{r['ablation']:8} {r['method']:18} {r['success_rate']:>7.1%} "
          f"{r['median_iters']:>7.1f} {r['mean_rmse_x']:>8.4f} {r['mean_time_sec']:>9.4f}s")
print("="*70)

print("\nExpected (from paper seed=7):")
print("  +JCP D-IPG  SR=100%  med_iters=5   RMSE≈0.034")
print("  +JCP GN     SR=29%   med_iters=2   RMSE≈0.185")
print("  +JCP LM     SR=100%  med_iters=6   RMSE≈0.047")

# ── CELL 13: Save ─────────────────────────────────────────────
all_results.to_csv("heat3d_main_results.csv", index=False)
torch.save(model_plus.state_dict(),  "heat3d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "heat3d_minus_jcp.pt")
print("✓ Saved")
