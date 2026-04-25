# =============================================================
# Deceptron — Allen-Cahn 2D Demo
# Exact protocol from AllenCahn2D.py
#
# Critical differences from Heat-1D/3D:
#   - 2D CNN with CIRCULAR padding (periodic BC)
#   - tanh output on reverse map (x in [-1,1])
#   - JCP computed from g(y) not x
#   - Success = RMSE < 0.10 (not residual tolerance)
#   - Stage 2 checkpoint: val_recon + 0.05 * val_rjcp
#   - Stage 3 checkpoint: recon_guard_factor logic
#   - GN/LM use CG (JVP+VJP), not explicit Jacobian
#   - L-BFGS as additional baseline
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
print(f"✓  deceptron v{deceptron.__version__}")


# ── CELL 2: Imports ───────────────────────────────────────────
import math, time, random, copy
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.func import jvp
from dataclasses import dataclass, asdict

# JCP from package
from deceptron.jcp import estimate_rjcp_dataset as _estimate_rjcp_pkg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}  |  torch: {torch.__version__}")


# ── CELL 3: Config (exact AllenCahn2DConfig) ──────────────────
@dataclass
class Cfg:
    nx: int = 32;  ny: int = 32
    x_channels: int = 1;  y_channels: int = 1
    epsilon: float = 0.04;  final_time: float = 0.12;  dt: float = 0.002
    noise_std: float = 0.002
    n_train: int = 2000;  n_val: int = 240;  n_test: int = 80
    hidden_channels: int = 48;  num_res_blocks: int = 4
    negative_slope: float = 0.10
    batch_size: int = 64
    forward_epochs: int = 120;  reverse_epochs: int = 80;  reverse_jcp_epochs: int = 40
    forward_lr: float = 2e-3;  reverse_lr: float = 1e-3;  reverse_jcp_lr: float = 5e-4
    forward_weight_decay: float = 1e-6;  reverse_weight_decay: float = 1e-6
    reverse_jcp_weight_decay: float = 1e-6
    reconstruction_weight: float = 1.0;  cycle_weight: float = 0.05
    probe_jcp_weight: float = 0.01
    jcp_num_probes_train: int = 1;  jcp_num_probes_eval: int = 2
    y_tilde_noise: float = 0.01;  gradient_clip: float = 5.0
    jcp_batch_subsample: int = 8
    recon_guard_factor: float = 1.005
    max_iterations: int = 80;  tolerance_eps: float = 0.12
    armijo_c: float = 1e-4;  max_backtracking_steps: int = 8;  rho: float = 0.4
    dipg_step_size: float = 1.0;  gd_step_size: float = 1.0
    gn_step_size: float = 1.0;  lm_step_size: float = 1.0;  lm_damping: float = 1e-2
    lbfgs_lr: float = 1.0;  lbfgs_max_iter: int = 80
    lbfgs_history_size: int = 20;  lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9
    cg_max_iter: int = 30;  cg_tol: float = 1e-6
    eval_subset_rjcp: int = 24
    success_rmse_threshold: float = 0.10   # ← success = RMSE < 0.10

cfg = Cfg()
print(f"✓ Config  {cfg.nx}×{cfg.ny}  dim={cfg.nx*cfg.ny}")


# ── CELL 4: PDE solver (exact from AllenCahn2D.py) ───────────
def solve_allen_cahn_final_state(u0: np.ndarray, c) -> np.ndarray:
    nx, ny = u0.shape
    u = u0.astype(np.float64).copy()
    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=1.0/nx)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=1.0/ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    lap   = KX**2 + KY**2
    n_steps = int(round(c.final_time / c.dt))
    denom = 1.0 + c.dt * (c.epsilon**2) * lap
    for _ in range(n_steps):
        rhs = u + c.dt*(u - u**3)
        u   = np.fft.ifft2(np.fft.fft2(rhs)/denom).real
    return np.clip(u, -1.0, 1.0).astype(np.float32)

def sample_initial_condition(nx: int, ny: int) -> np.ndarray:
    yy, xx = np.mgrid[0:nx, 0:ny]
    xx = (xx+0.5)/ny;  yy = (yy+0.5)/nx
    field = -np.ones((nx,ny), dtype=np.float32)
    family = np.random.choice(["blobs","fronts","rings","mixed"])
    if family in ["blobs","mixed"]:
        for _ in range(np.random.randint(2,6)):
            st = np.random.choice(["circle","ellipse","rectangle"])
            cx,cy = np.random.uniform(0.15,0.85), np.random.uniform(0.15,0.85)
            if st=="circle":
                r = np.random.uniform(0.05,0.16)
                mask = (xx-cx)**2+(yy-cy)**2 <= r**2
            elif st=="ellipse":
                rx,ry = np.random.uniform(0.05,0.18), np.random.uniform(0.05,0.18)
                th = np.random.uniform(0,2*math.pi)
                xr = np.cos(th)*(xx-cx)+np.sin(th)*(yy-cy)
                yr = -np.sin(th)*(xx-cx)+np.cos(th)*(yy-cy)
                mask = (xr/rx)**2+(yr/ry)**2 <= 1.0
            else:
                wx,wy = np.random.uniform(0.07,0.22), np.random.uniform(0.07,0.22)
                th = np.random.uniform(0,2*math.pi)
                xr = np.cos(th)*(xx-cx)+np.sin(th)*(yy-cy)
                yr = -np.sin(th)*(xx-cx)+np.cos(th)*(yy-cy)
                mask = (np.abs(xr)<=wx/2)&(np.abs(yr)<=wy/2)
            field[mask] = 1.0
    if family in ["fronts","mixed"]:
        th  = np.random.uniform(0,2*math.pi)
        off = np.random.uniform(-0.2,0.2)
        ramp = np.cos(th)*(xx-0.5)+np.sin(th)*(yy-0.5)-off
        front = np.tanh(25.0*ramp).astype(np.float32)
        field = np.where(np.abs(front)>0.4, front, field)
    if family in ["rings","mixed"]:
        cx,cy = np.random.uniform(0.25,0.75), np.random.uniform(0.25,0.75)
        r0  = np.random.uniform(0.10,0.24)
        w   = np.random.uniform(0.02,0.05)
        dist = np.sqrt((xx-cx)**2+(yy-cy)**2)
        ring = np.exp(-((dist-r0)**2)/(2*w**2))
        field[ring>0.35] = 1.0
    field = field + 0.05*np.random.randn(nx,ny).astype(np.float32)
    return np.clip(field,-1.0,1.0).astype(np.float32)

print("✓ PDE solver ready")


# ── CELL 5: Dataset (exact from AllenCahn2D.py) ───────────────
@torch.no_grad()
def build_dataset(c, seed=7):
    np.random.seed(seed); torch.manual_seed(seed)
    n = c.n_train+c.n_val+c.n_test
    x_all, y_all = [], []
    for _ in range(n):
        u0 = sample_initial_condition(c.nx, c.ny)
        uT = solve_allen_cahn_final_state(u0, c)
        x_all.append(u0[None]); y_all.append(uT[None])
    x_all = torch.tensor(np.stack(x_all), dtype=torch.float32)
    y_clean = torch.tensor(np.stack(y_all), dtype=torch.float32)
    y_noisy = torch.clamp(y_clean + c.noise_std*torch.randn_like(y_clean), -1.0, 1.0)
    nt,nv = c.n_train, c.n_val
    ym = y_clean[:nt].mean(dim=(0,2,3), keepdim=True)
    ys = y_clean[:nt].std(dim=(0,2,3),  keepdim=True).clamp(1e-6)
    def nm(y): return (y-ym)/ys
    return {"x_train":x_all[:nt], "x_val":x_all[nt:nt+nv], "x_test":x_all[nt+nv:],
            "y_train_norm":nm(y_clean[:nt]), "y_val_norm":nm(y_clean[nt:nt+nv]),
            "y_noisy_test_norm":nm(y_noisy[nt+nv:]),   # eval uses noisy targets
            "y_noisy_test":y_noisy[nt+nv:],            # for visualization
            "y_mean":ym, "y_std":ys}

print("Building dataset (this runs the PDE solver, takes ~2 min)...")
data = build_dataset(cfg, seed=7)
print(f"✓ Dataset: train={cfg.n_train}  val={cfg.n_val}  test={cfg.n_test}")
print(f"  x shape: {data['x_train'].shape}   (B, 1, nx, ny)")

BS = cfg.batch_size
def mk(x,y,sh): return DataLoader(TensorDataset(x,y),batch_size=BS,shuffle=sh)
tr_loader  = mk(data["x_train"], data["y_train_norm"], True)
val_loader = mk(data["x_val"],   data["y_val_norm"],   False)


# ── CELL 6: CNN model (exact from AllenCahn2D.py) ─────────────
class ConvResBlock(nn.Module):
    def __init__(self, ch, ns):
        super().__init__()
        # circular padding — exact as paper (periodic BC for Allen-Cahn)
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular")
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, padding_mode="circular")
        self.ns = ns
    def forward(self, x):
        return F.leaky_relu(x + self.c2(F.leaky_relu(self.c1(x), self.ns)), self.ns)

class ForwardCNN(nn.Module):
    def __init__(self, in_ch, out_ch, hidden, n_blocks, ns):
        super().__init__()
        self.entry  = nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList([ConvResBlock(hidden, ns) for _ in range(n_blocks)])
        self.exit   = nn.Conv2d(hidden, out_ch, 3, padding=1, padding_mode="circular")
        self.ns = ns
    def forward(self, x):
        z = F.leaky_relu(self.entry(x), self.ns)
        for b in self.blocks: z = b(z)
        return self.exit(z)

class ReverseCNN(nn.Module):
    def __init__(self, in_ch, hidden, n_blocks, ns):
        super().__init__()
        self.entry  = nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode="circular")
        self.blocks = nn.ModuleList([ConvResBlock(hidden, ns) for _ in range(n_blocks)])
        self.exit   = nn.Conv2d(hidden, 1, 3, padding=1, padding_mode="circular")
        self.ns = ns
    def forward(self, y):
        z = F.leaky_relu(self.entry(y), self.ns)
        for b in self.blocks: z = b(z)
        return torch.tanh(self.exit(z))  # ← tanh: x in [-1,1]

class DeceptronAC(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.forward_map = ForwardCNN(c.x_channels, c.y_channels,
                                       c.hidden_channels, c.num_res_blocks, c.negative_slope)
        self.reverse_map = ReverseCNN(c.y_channels, c.hidden_channels,
                                       c.num_res_blocks, c.negative_slope)
    def f(self, x): return self.forward_map(x)
    def g(self, y): return self.reverse_map(y)
    def forward(self, x): return self.f(x)

print("✓ Model defined")


# ── CELL 7: JCP (exact from AllenCahn2D.py) ──────────────────
# Key difference: probe is from x with shape (1, nx, ny)
def single_sample_jcp_ac(model, x, num_probes=1):
    """x has shape (1, nx, ny) — keeps spatial dimensions."""
    x_shape = tuple(x.shape)    # (1, nx, ny)
    x_vec   = x.reshape(-1)
    total   = x_vec.new_tensor(0.0)

    def f_single(inp):
        return model.f(inp.reshape(1, *x_shape)).reshape(-1)

    y0 = f_single(x_vec)

    def g_single(inp):
        return model.g(inp.reshape(1, cfg.y_channels, cfg.nx, cfg.ny)).reshape(-1)

    for _ in range(num_probes):
        xi = torch.empty_like(x_vec).bernoulli_(0.5).mul_(2.).sub_(1.)
        _, jf_xi    = jvp(f_single, (x_vec,), (xi,))
        _, jg_jf_xi = jvp(g_single, (y0,),    (jf_xi,))
        total = total + torch.mean((jg_jf_xi - xi)**2)
    return total / num_probes

def batch_jcp_ac(model, x_batch, num_probes=1):
    return torch.stack([single_sample_jcp_ac(model, x_batch[i], num_probes)
                        for i in range(x_batch.shape[0])]).mean()

@torch.no_grad()
def estimate_rjcp_ac(model, x_data, num_probes=2, max_samples=24):
    model.eval()
    x = x_data[:max_samples].to(device)
    return float(torch.stack([
        single_sample_jcp_ac(model, x[i], num_probes).detach().cpu()
        for i in range(x.shape[0])]).mean())

print("✓ JCP helpers ready")


# ── CELL 8: Eval helpers ──────────────────────────────────────
@torch.no_grad()
def eval_fwd_val(model, loader):
    model.eval(); tot, count = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        tot += float(F.mse_loss(model.f(xb), yb, reduction="sum").item())
        count += yb.numel()
    return tot / max(count, 1)

@torch.no_grad()
def eval_rev_val(model, loader):
    """Exact evaluate_reverse_validation from AllenCahn2D.py."""
    model.eval(); totals = {"rec":0., "cyc":0., "loss":0.}; count = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        xh  = model.g(yb)
        yc  = model.f(xh)
        rec = F.mse_loss(xh, xb)
        cyc = F.mse_loss(yc, yb)
        loss = cfg.reconstruction_weight*rec + cfg.cycle_weight*cyc
        bs = xb.shape[0]
        totals["rec"]  += float(rec.item())*bs
        totals["cyc"]  += float(cyc.item())*bs
        totals["loss"] += float(loss.item())*bs
        count += bs
    return {k: v/count for k,v in totals.items()}

def freeze_fwd(m):
    for p in m.forward_map.parameters(): p.requires_grad_(False)

def state_copy(m):
    return {k: v.detach().cpu().clone() for k,v in m.state_dict().items()}

print("✓ Eval helpers ready")


# ── CELL 9: Stage 1 ───────────────────────────────────────────
model = DeceptronAC(cfg).to(device)
opt1  = torch.optim.Adam(model.forward_map.parameters(),
                          lr=cfg.forward_lr, weight_decay=cfg.forward_weight_decay)
sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=cfg.forward_epochs)

best_val1, best_s1 = float("inf"), None
print("Stage 1: Training forward map...")

for epoch in range(cfg.forward_epochs):
    model.train(); tr, count = 0., 0
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt1.zero_grad(set_to_none=True)
        loss = F.mse_loss(model.f(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.forward_map.parameters(), cfg.gradient_clip)
        opt1.step(); tr += float(loss.item())*xb.shape[0]; count += xb.shape[0]
    sched1.step()
    vl = eval_fwd_val(model, val_loader)
    rjcp = estimate_rjcp_ac(model, data["x_val"].to(device), cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp)
    if vl < best_val1: best_val1=vl; best_s1=state_copy(model)
    if (epoch+1)%20==0 or epoch==0:
        print(f"  epoch {epoch+1:03d} | train={tr/count:.6f} | val={vl:.6f} | rjcp={rjcp:.6f}")

model.load_state_dict(best_s1); model_s1=copy.deepcopy(model)
freeze_fwd(model)
print(f"✓ Stage 1 done. best val={best_val1:.6f}")


# ── CELL 10: Stage 2 ──────────────────────────────────────────
opt2   = torch.optim.Adam(model.reverse_map.parameters(),
                           lr=cfg.reverse_lr, weight_decay=cfg.reverse_weight_decay)
sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=cfg.reverse_epochs)

best_sc2, best_s2 = float("inf"), None
print("\nStage 2: Reverse map (no JCP)...")

for epoch in range(cfg.reverse_epochs):
    model.train()
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt2.zero_grad(set_to_none=True)
        xh  = model.g(yb)
        yt  = yb + cfg.y_tilde_noise*torch.randn_like(yb)
        yc  = model.f(model.g(yt))
        loss = (cfg.reconstruction_weight*F.mse_loss(xh, xb) +
                cfg.cycle_weight*F.mse_loss(yc, yt))
        loss.backward()
        nn.utils.clip_grad_norm_(model.reverse_map.parameters(), cfg.gradient_clip)
        opt2.step()
    sched2.step()
    vs   = eval_rev_val(model, val_loader)
    rjcp = estimate_rjcp_ac(model, data["x_val"].to(device), cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp)
    # Stage 2 checkpoint: val_reconstruction + 0.05 * val_rjcp (exact from paper)
    sc = vs["rec"] + 0.05*rjcp
    if sc < best_sc2: best_sc2=sc; best_s2=state_copy(model)
    if (epoch+1)%20==0 or epoch==0:
        print(f"  epoch {epoch+1:03d} | val_rec={vs['rec']:.6f} | val_rjcp={rjcp:.6f}")

model.load_state_dict(best_s2); model_s2=copy.deepcopy(model)
reference_recon = eval_rev_val(model, val_loader)["rec"]
print(f"✓ Stage 2 done. reference_recon={reference_recon:.6f}")


# ── CELL 11: Stage 3 ──────────────────────────────────────────
def train_stage3_ac(model_init, use_jcp, label):
    """
    Exact train_reverse_map_with_jcp_fixed_forward from AllenCahn2D.py.

    Key unique features:
    - JCP probed from x_pred = g(y_subset) with no_grad (not from x directly)
    - Checkpoint uses recon_guard_factor: only reduce RJCP if reconstruction OK
    """
    m = copy.deepcopy(model_init); freeze_fwd(m)
    opt = torch.optim.Adam(m.reverse_map.parameters(),
                            lr=cfg.reverse_jcp_lr, weight_decay=cfg.reverse_jcp_weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.reverse_jcp_epochs)
    best_sc, best_st = float("inf"), None
    print(f"\nStage 3 ({label})...")

    for epoch in range(cfg.reverse_jcp_epochs):
        m.train(); tot=dict(loss=0.,rec=0.,cyc=0.,jcp=0.); count=0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            xh  = m.g(yb)
            yt  = yb + cfg.y_tilde_noise*torch.randn_like(yb)
            yc  = m.f(m.g(yt))
            rec = F.mse_loss(xh, xb)
            cyc = F.mse_loss(yc, yt)
            if use_jcp:
                bs_jcp = min(cfg.jcp_batch_subsample, xb.shape[0])
                y_sub  = yb[:bs_jcp]
                with torch.no_grad():
                    x_pred_sub = m.g(y_sub)  # ← JCP from g(y), not x
                lj = cfg.probe_jcp_weight * batch_jcp_ac(m, x_pred_sub, cfg.jcp_num_probes_train)
            else:
                lj = xb.new_tensor(0.)
            loss = cfg.reconstruction_weight*rec + cfg.cycle_weight*cyc + lj
            loss.backward()
            nn.utils.clip_grad_norm_(m.reverse_map.parameters(), cfg.gradient_clip)
            opt.step()
            bs=xb.shape[0]; count+=bs
            tot["loss"]+=float(loss.item())*bs; tot["rec"]+=float(rec.item())*bs
            tot["cyc"]+=float(cyc.item())*bs;   tot["jcp"]+=float(lj.item())*bs
        sched.step()
        vs   = eval_rev_val(m, val_loader)
        rjcp = estimate_rjcp_ac(m, data["x_val"].to(device), cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp)
        # Checkpoint logic with reconstruction guard (exact from paper)
        recon_ratio = vs["rec"] / max(reference_recon, 1e-12)
        if recon_ratio <= cfg.recon_guard_factor:
            sc = rjcp + 1e-3*vs["loss"]
        else:
            sc = rjcp + 1000.*(recon_ratio - cfg.recon_guard_factor)
        if sc < best_sc: best_sc=sc; best_st=state_copy(m)
        if (epoch+1)%10==0 or epoch==0:
            print(f"  {label} epoch {epoch+1:03d} | val_rec={vs['rec']:.6f} "
                  f"| val_rjcp={rjcp:.6f} | probe_jcp={tot['jcp']/count:.6f}")
    m.load_state_dict(best_st); return m

model_plus  = train_stage3_ac(model_s2, use_jcp=True,  label="+JCP")
model_minus = train_stage3_ac(model_s2, use_jcp=False, label="-JCP")

rjcp_p = estimate_rjcp_ac(model_plus,  data["x_val"].to(device), 2, 24)
rjcp_m = estimate_rjcp_ac(model_minus, data["x_val"].to(device), 2, 24)
print(f"\n+JCP RJCP = {rjcp_p:.4f}")
print(f"-JCP RJCP = {rjcp_m:.4f}")
print(f"Ratio     = {rjcp_m/rjcp_p:.2f}×")


# ── CELL 12: Solvers (exact from AllenCahn2D.py) ──────────────
def phi_val_ac(f, x, y): return 0.5*torch.mean((f(x)-y)**2)
def phi_grad_ac(f, x, y):
    xr = x.detach().clone().requires_grad_(True)
    r  = f(xr)-y
    ph = 0.5*torch.mean(r*r)
    g  = torch.autograd.grad(ph, xr)[0]
    return ph.detach(), g.detach(), r.detach()
def norm_res_ac(f, x, y): return torch.sqrt(torch.mean((f(x)-y)**2)).item()
def clamp_ac(x):   return torch.clamp(x, -1.0, 1.0)  # ← [-1,1] not [0,1]
def armijo_ac(f, x, p, y, rho, c):
    phi_t, grad_t, _ = phi_grad_ac(f, x, y)
    xt    = clamp_ac((1-rho)*x + rho*(x+p))
    pt    = phi_val_ac(f, xt, y)
    rhs   = phi_t + c*rho*torch.sum(grad_t*p)
    return xt, pt, rhs

def jvp_fwd(f, x, v):
    def fs(inp): return f(inp)
    _, jv = jvp(fs, (x,), (v,)); return jv

def vjp_fwd(f, x, u):
    xr = x.detach().clone().requires_grad_(True)
    return torch.autograd.grad(torch.sum(f(xr)*u), xr)[0]

def normal_matvec(f, x, v, lm=0.0):
    jv    = jvp_fwd(f, x, v)
    jtjv  = vjp_fwd(f, x, jv)
    return jtjv/float(jv.numel()) + lm*v

def cg_solve(matvec, b, max_iter=30, tol=1e-6):
    x=torch.zeros_like(b); r=b-matvec(x); p=r.clone()
    rso = torch.sum(r*r)
    for _ in range(max_iter):
        Ap  = matvec(p)
        a   = rso/torch.sum(p*Ap).clamp_min(1e-12)
        x   = x+a*p; r=r-a*Ap; rsn=torch.sum(r*r)
        if rsn.sqrt()<tol: break
        p   = r+(rsn/rso.clamp_min(1e-12))*p; rso=rsn
    return x

def rmse_x(a, b): return float(torch.sqrt(torch.mean((a.cpu()-b.cpu())**2)))

def solve_dipg_ac(model, y_star, x0):
    model.eval()
    x = x0.clone().to(device); y_star=y_star.to(device)
    acc=0; drops=[]; t0=time.perf_counter()
    for it in range(cfg.max_iterations):
        _,grad_t,resid_t = phi_grad_ac(model.f, x, y_star)
        nr = torch.sqrt(torch.mean(resid_t**2)).item()
        if nr <= cfg.tolerance_eps: break
        alpha,ok = cfg.dipg_step_size, False
        y_t = model.f(x)
        for _ in range(cfg.max_backtracking_steps+1):
            yp = y_t - alpha*resid_t
            xp = model.g(yp)
            p  = xp - x
            xt,pt,rhs = armijo_ac(model.f, x, p, y_star, cfg.rho, cfg.armijo_c)
            if pt <= rhs:
                drops.append(nr - norm_res_ac(model.f, xt, y_star))
                x=xt.detach(); ok=True; acc+=1; break
            alpha*=0.5
        if not ok: break
    nr_fin = norm_res_ac(model.f, x, y_star)
    rjcp   = float(single_sample_jcp_ac(model, x.detach().squeeze(0), 2).detach())
    return {"x_hat":x.detach().cpu(),"iters":it+1,
            "final_residual":nr_fin,"accepted_frac":acc/max(it+1,1),
            "time_sec":time.perf_counter()-t0,"final_rjcp":rjcp}

def solve_gd_ac(f, y_star, x0):
    x=x0.clone().to(device); y_star=y_star.to(device); acc=0; t0=time.perf_counter()
    for it in range(cfg.max_iterations):
        _,grad_t,resid_t = phi_grad_ac(f, x, y_star)
        if torch.sqrt(torch.mean(resid_t**2)).item() <= cfg.tolerance_eps: break
        eta,ok = cfg.gd_step_size, False
        for _ in range(cfg.max_backtracking_steps+1):
            xt,pt,rhs = armijo_ac(f, x, -eta*grad_t, y_star, cfg.rho, cfg.armijo_c)
            if pt<=rhs: x=xt.detach(); ok=True; acc+=1; break
            eta*=0.5
        if not ok: break
    return {"x_hat":x.detach().cpu(),"iters":it+1,
            "final_residual":norm_res_ac(f,x,y_star),"accepted_frac":acc/max(it+1,1),
            "time_sec":time.perf_counter()-t0,"final_rjcp":float("nan")}

def solve_gn_ac(f, y_star, x0):
    """GN with CG — exact from AllenCahn2D.py (no explicit Jacobian)."""
    x=x0.clone().to(device); y_star=y_star.to(device); acc=0; t0=time.perf_counter()
    for it in range(cfg.max_iterations):
        _,_,resid_t = phi_grad_ac(f, x, y_star)
        if torch.sqrt(torch.mean(resid_t**2)).item() <= cfg.tolerance_eps: break
        rhs = -vjp_fwd(f, x, resid_t)/float(resid_t.numel())
        dx  = cg_solve(lambda v: normal_matvec(f,x,v,0.), rhs, cfg.cg_max_iter, cfg.cg_tol)
        alpha,ok = cfg.gn_step_size, False
        for _ in range(cfg.max_backtracking_steps+1):
            xt,pt,rhs_a = armijo_ac(f, x, alpha*dx, y_star, cfg.rho, cfg.armijo_c)
            if pt<=rhs_a: x=xt.detach(); ok=True; acc+=1; break
            alpha*=0.5
        if not ok: break
    return {"x_hat":x.detach().cpu(),"iters":it+1,
            "final_residual":norm_res_ac(f,x,y_star),"accepted_frac":acc/max(it+1,1),
            "time_sec":time.perf_counter()-t0,"final_rjcp":float("nan")}

def solve_lm_ac(f, y_star, x0):
    x=x0.clone().to(device); y_star=y_star.to(device); acc=0; t0=time.perf_counter()
    for it in range(cfg.max_iterations):
        _,_,resid_t = phi_grad_ac(f, x, y_star)
        if torch.sqrt(torch.mean(resid_t**2)).item() <= cfg.tolerance_eps: break
        rhs = -vjp_fwd(f, x, resid_t)/float(resid_t.numel())
        dx  = cg_solve(lambda v: normal_matvec(f,x,v,cfg.lm_damping), rhs, cfg.cg_max_iter, cfg.cg_tol)
        alpha,ok = cfg.lm_step_size, False
        for _ in range(cfg.max_backtracking_steps+1):
            xt,pt,rhs_a = armijo_ac(f, x, alpha*dx, y_star, cfg.rho, cfg.armijo_c)
            if pt<=rhs_a: x=xt.detach(); ok=True; acc+=1; break
            alpha*=0.5
        if not ok: break
    return {"x_hat":x.detach().cpu(),"iters":it+1,
            "final_residual":norm_res_ac(f,x,y_star),"accepted_frac":acc/max(it+1,1),
            "time_sec":time.perf_counter()-t0,"final_rjcp":float("nan")}

def solve_lbfgs_ac(f, y_star, x0):
    x = nn.Parameter(x0.clone().to(device).contiguous())
    y_star = y_star.to(device).contiguous()
    opt = torch.optim.LBFGS([x], lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter,
                              history_size=cfg.lbfgs_history_size,
                              tolerance_grad=cfg.lbfgs_tolerance_grad,
                              tolerance_change=cfg.lbfgs_tolerance_change,
                              line_search_fn="strong_wolfe")
    n_iters = {"n":0}; t0=time.perf_counter()
    def closure():
        opt.zero_grad(set_to_none=True)
        xc = torch.clamp(x,-1.,1.).contiguous()
        loss = 0.5*torch.mean((f(xc)-y_star)**2)
        loss.backward()
        if x.grad is not None: x.grad=x.grad.contiguous()
        n_iters["n"]+=1; return loss
    opt.step(closure)
    xf = torch.clamp(x.detach(),-1.,1.).contiguous()
    return {"x_hat":xf.cpu(),"iters":n_iters["n"],
            "final_residual":norm_res_ac(f,xf,y_star),"accepted_frac":1.0,
            "time_sec":time.perf_counter()-t0,"final_rjcp":float("nan")}

print("✓ Solvers ready")


# ── CELL 13: Evaluate (exact from AllenCahn2D.py) ─────────────
def evaluate_all_ac(x_test, y_test_norm, mdl_plus, mdl_minus):
    """
    Exact evaluate_allen_cahn_methods.
    Success = RMSE < 0.10 (not residual tolerance).
    x0 = zeros (shape matches x_true).
    """
    rows = []
    specs = [
        ("+JCP",     "D-IPG",            lambda y,x0: solve_dipg_ac(mdl_plus, y, x0)),
        ("-JCP",     "D-IPG",            lambda y,x0: solve_dipg_ac(mdl_minus, y, x0)),
        ("Baseline", "Gradient Descent", lambda y,x0: solve_gd_ac(mdl_plus.f, y, x0)),
        ("Baseline", "GN",               lambda y,x0: solve_gn_ac(mdl_plus.f, y, x0)),
        ("Baseline", "LM",               lambda y,x0: solve_lm_ac(mdl_plus.f, y, x0)),
        ("Baseline", "L-BFGS",           lambda y,x0: solve_lbfgs_ac(mdl_plus.f, y, x0)),
    ]
    for ablation, mname, mfn in specs:
        per = []
        for i in range(x_test.shape[0]):
            x_true = x_test[i:i+1]
            y_star = y_test_norm[i:i+1].to(device)
            x0     = torch.zeros_like(x_true).to(device)
            res    = mfn(y_star, x0)
            rmse   = rmse_x(res["x_hat"], x_true)
            per.append({"rmse_x":rmse,
                        "success":float(rmse <= cfg.success_rmse_threshold),
                        **{k:v for k,v in res.items() if k!="x_hat"}})
        df = pd.DataFrame(per)
        rows.append({"ablation":ablation,"method":mname,
                     "success_rate":df["success"].mean(),
                     "mean_iters":df["iters"].mean(),"median_iters":df["iters"].median(),
                     "mean_rmse_x":df["rmse_x"].mean(),
                     "mean_final_residual":df["final_residual"].mean(),
                     "mean_time_sec":df["time_sec"].mean()})
        print(f"  done: {ablation} | {mname}")
    return pd.DataFrame(rows)

print("Evaluating all methods...")
results = evaluate_all_ac(data["x_test"], data["y_noisy_test_norm"],
                           model_plus, model_minus)

print("\n" + "="*72)
print(f"{'Ablation':10} {'Method':18} {'SR':>7} {'Med.It':>7} {'RMSE':>8} {'Time':>9}")
print("="*72)
for _,r in results.iterrows():
    print(f"{r['ablation']:10} {r['method']:18} {r['success_rate']:>7.1%} "
          f"{r['median_iters']:>7.1f} {r['mean_rmse_x']:>8.4f} "
          f"{r['mean_time_sec']:>9.4f}s")
print("="*72)

print("\nExpected (from paper seed=7):")
print("  +JCP D-IPG  SR=100%  med_iters=6   RMSE≈0.047")
print("  -JCP D-IPG  SR≈16%   (bimodal basin, Figure 2 story)")
print("  GN          SR≈43%   med_iters=6   RMSE≈0.063")
print("  L-BFGS      SR≈75%   RMSE≈0.060")


# ── CELL 14: Save ─────────────────────────────────────────────
results.to_csv("allen_cahn2d_main_results.csv", index=False)
torch.save(model_plus.state_dict(),  "allen_cahn2d_plus_jcp.pt")
torch.save(model_minus.state_dict(), "allen_cahn2d_minus_jcp.pt")
print("✓ Saved")
