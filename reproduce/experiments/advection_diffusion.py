"""Advection–diffusion initial-field recovery.

Seed-7 experiment from reproduce/notebooks/Advection_diffusion.ipynb.
The original notebook also contains extended analyses.
"""

from _output import prepare_output

prepare_output("advection_diffusion")


# Problem Setup

import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.func import jvp

SEED = 7
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

@dataclass
class AdvDiffDegradedConfig:
    nx: int = 32
    ny: int = 32

    x_channels: int = 1
    y_channels: int = 1

    velocity_x: float = 1.0
    velocity_y: float = -0.7
    diffusion_eps: float = 8e-4
    final_time: float = 0.28

    blur_sigma: float = 1.0
    downsample_factor: int = 2
    noise_std: float = 0.01

    n_train: int = 2000
    n_val: int = 240
    n_test: int = 80

    hidden_channels: int = 48
    num_res_blocks: int = 4
    negative_slope: float = 0.10

    batch_size: int = 64

    forward_epochs: int = 120
    reverse_epochs: int = 80
    reverse_jcp_epochs: int = 40

    forward_lr: float = 2e-3
    reverse_lr: float = 1e-3
    reverse_jcp_lr: float = 5e-4

    forward_weight_decay: float = 1e-6
    reverse_weight_decay: float = 1e-6
    reverse_jcp_weight_decay: float = 1e-6

    reconstruction_weight: float = 1.0
    cycle_weight: float = 0.05
    probe_jcp_weight: float = 0.01

    jcp_num_probes_train: int = 1
    jcp_num_probes_eval: int = 2
    y_tilde_noise: float = 0.01
    gradient_clip: float = 5.0
    jcp_batch_subsample: int = 8

    recon_guard_factor: float = 1.005

    max_iterations: int = 80
    tolerance_eps: float = 0.12
    armijo_c: float = 1e-4
    max_backtracking_steps: int = 8
    rho: float = 0.4

    dipg_step_size: float = 1.0
    gd_step_size: float = 1.0
    gn_step_size: float = 1.0
    lm_step_size: float = 1.0
    lm_damping: float = 1e-2

    lbfgs_lr: float = 1.0
    lbfgs_max_iter: int = 80
    lbfgs_history_size: int = 20
    lbfgs_tolerance_grad: float = 1e-7
    lbfgs_tolerance_change: float = 1e-9

    cg_max_iter: int = 30
    cg_tol: float = 1e-6

    eval_subset_rjcp: int = 24
    success_rmse_threshold: float = 0.10

cfg = AdvDiffDegradedConfig()
print(asdict(cfg))

def solve_advdiff_final_state(u0: np.ndarray, cfg: AdvDiffDegradedConfig) -> np.ndarray:
    nx, ny = u0.shape

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=1.0 / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=1.0 / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")

    lap_symbol = KX**2 + KY**2
    adv_symbol = cfg.velocity_x * KX + cfg.velocity_y * KY

    multiplier = np.exp(-(cfg.diffusion_eps * lap_symbol + 1j * adv_symbol) * cfg.final_time)

    u0_hat = np.fft.fft2(u0)
    uT_hat = multiplier * u0_hat
    uT = np.fft.ifft2(uT_hat).real.astype(np.float32)

    return np.clip(uT, 0.0, 1.0)

def gaussian_kernel2d(kernel_size: int = 7, sigma: float = 1.0) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

def degrade_final_state(uT: torch.Tensor, cfg: AdvDiffDegradedConfig) -> torch.Tensor:
    # uT shape: (B,1,H,W)
    kernel = gaussian_kernel2d(kernel_size=7, sigma=cfg.blur_sigma).to(uT.device)
    kernel = kernel.view(1, 1, 7, 7)

    u_blur = F.conv2d(
        F.pad(uT, (3, 3, 3, 3), mode="circular"),
        kernel,
        stride=1,
        padding=0,
    )

    u_ds = F.avg_pool2d(u_blur, kernel_size=cfg.downsample_factor, stride=cfg.downsample_factor)
    u_up = F.interpolate(u_ds, size=(cfg.nx, cfg.ny), mode="bilinear", align_corners=False)

    y = torch.clamp(u_up + cfg.noise_std * torch.randn_like(u_up), 0.0, 1.0)
    return y

def sample_initial_condition(nx: int, ny: int) -> np.ndarray:
    yy, xx = np.mgrid[0:nx, 0:ny]
    xx = (xx + 0.5) / ny
    yy = (yy + 0.5) / nx

    field = np.zeros((nx, ny), dtype=np.float32)

    family = np.random.choice(["blobs", "fronts", "rings", "mixed"])

    if family in ["blobs", "mixed"]:
        n_objects = np.random.randint(2, 6)
        for _ in range(n_objects):
            shape_type = np.random.choice(["circle", "ellipse", "rectangle"])
            amp = np.random.uniform(0.4, 1.0)
            cx = np.random.uniform(0.15, 0.85)
            cy = np.random.uniform(0.15, 0.85)

            if shape_type == "circle":
                r = np.random.uniform(0.05, 0.16)
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            elif shape_type == "ellipse":
                rx = np.random.uniform(0.05, 0.18)
                ry = np.random.uniform(0.05, 0.18)
                theta = np.random.uniform(0.0, 2.0 * math.pi)
                x0 = xx - cx
                y0 = yy - cy
                xr = np.cos(theta) * x0 + np.sin(theta) * y0
                yr = -np.sin(theta) * x0 + np.cos(theta) * y0
                mask = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
            else:
                wx = np.random.uniform(0.07, 0.22)
                wy = np.random.uniform(0.07, 0.22)
                theta = np.random.uniform(0.0, 2.0 * math.pi)
                x0 = xx - cx
                y0 = yy - cy
                xr = np.cos(theta) * x0 + np.sin(theta) * y0
                yr = -np.sin(theta) * x0 + np.cos(theta) * y0
                mask = (np.abs(xr) <= wx / 2.0) & (np.abs(yr) <= wy / 2.0)

            field[mask] = np.maximum(field[mask], amp)

    if family in ["fronts", "mixed"]:
        theta = np.random.uniform(0.0, 2.0 * math.pi)
        offset = np.random.uniform(-0.2, 0.2)
        ramp = np.cos(theta) * (xx - 0.5) + np.sin(theta) * (yy - 0.5) - offset
        front = 1.0 / (1.0 + np.exp(-40.0 * ramp))
        amp = np.random.uniform(0.4, 0.9)
        field = np.maximum(field, amp * front.astype(np.float32))

    if family in ["rings", "mixed"]:
        cx = np.random.uniform(0.25, 0.75)
        cy = np.random.uniform(0.25, 0.75)
        r0 = np.random.uniform(0.10, 0.24)
        width = np.random.uniform(0.02, 0.05)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ring = np.exp(-((dist - r0) ** 2) / (2 * width**2))
        amp = np.random.uniform(0.4, 0.9)
        field = np.maximum(field, amp * ring.astype(np.float32))

    return np.clip(field, 0.0, 1.0).astype(np.float32)

fig, axes = plt.subplots(3, 4, figsize=(10, 7), dpi=140)

for i in range(4):
    u0 = sample_initial_condition(cfg.nx, cfg.ny)
    uT = solve_advdiff_final_state(u0, cfg)

    uT_t = torch.tensor(uT[None, None], dtype=torch.float32)
    y = degrade_final_state(uT_t, cfg)[0, 0].numpy()

    axes[0, i].imshow(u0, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, i].set_title("Initial", fontsize=10)
    axes[0, i].axis("off")

    axes[1, i].imshow(uT, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[1, i].set_title("Final", fontsize=10)
    axes[1, i].axis("off")

    axes[2, i].imshow(y, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[2, i].set_title("Observed", fontsize=10)
    axes[2, i].axis("off")

plt.tight_layout()
plt.close(fig)

@torch.no_grad()
def build_dataset(config: AdvDiffDegradedConfig):
    n_total = config.n_train + config.n_val + config.n_test

    x_all = []
    y_clean_all = []

    batch_uT = []

    for _ in range(n_total):
        u0 = sample_initial_condition(config.nx, config.ny)
        uT = solve_advdiff_final_state(u0, config)

        x_all.append(u0[None, :, :])
        batch_uT.append(uT[None, :, :])

    x_all = torch.tensor(np.stack(x_all, axis=0), dtype=torch.float32)
    uT_all = torch.tensor(np.stack(batch_uT, axis=0), dtype=torch.float32)

    y_clean_all = degrade_final_state(uT_all, config)

    x_train = x_all[:config.n_train]
    x_val = x_all[config.n_train: config.n_train + config.n_val]
    x_test = x_all[-config.n_test:]

    y_clean_train = y_clean_all[:config.n_train]
    y_clean_val = y_clean_all[config.n_train: config.n_train + config.n_val]
    y_clean_test = y_clean_all[-config.n_test:]

    y_mean = y_clean_train.mean(dim=(0, 2, 3), keepdim=True)
    y_std = y_clean_train.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_clean_train": y_clean_train,
        "y_clean_val": y_clean_val,
        "y_clean_test": y_clean_test,
        "y_train_norm": (y_clean_train - y_mean) / y_std,
        "y_val_norm": (y_clean_val - y_mean) / y_std,
        "y_test_norm": (y_clean_test - y_mean) / y_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

data = build_dataset(cfg)

for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(key, tuple(value.shape))

forward_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

forward_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

reverse_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

reverse_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

reverse_jcp_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

reverse_jcp_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, negative_slope: float):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)
        z = self.conv2(z)
        return F.leaky_relu(x + z, negative_slope=self.negative_slope)

class ForwardCNN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int, num_res_blocks: int, negative_slope: float):
        super().__init__()
        self.negative_slope = negative_slope
        self.entry = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.blocks = nn.ModuleList([
            ConvResidualBlock(hidden_channels, negative_slope)
            for _ in range(num_res_blocks)
        ])
        self.exit = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode="circular"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.entry(x), negative_slope=self.negative_slope)
        for block in self.blocks:
            z = block(z)
        return self.exit(z)

class ReverseCNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_res_blocks: int, negative_slope: float):
        super().__init__()
        self.negative_slope = negative_slope
        self.entry = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode="circular"
        )
        self.blocks = nn.ModuleList([
            ConvResidualBlock(hidden_channels, negative_slope)
            for _ in range(num_res_blocks)
        ])
        self.exit = nn.Conv2d(
            hidden_channels, 1, kernel_size=3, padding=1, padding_mode="circular"
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.entry(y), negative_slope=self.negative_slope)
        for block in self.blocks:
            z = block(z)
        return torch.sigmoid(self.exit(z))

class DeceptronAdvDiffModel(nn.Module):
    def __init__(self, x_channels: int, y_channels: int, hidden_channels: int, num_res_blocks: int, negative_slope: float):
        super().__init__()
        self.forward_map = ForwardCNN(
            x_channels, y_channels, hidden_channels, num_res_blocks, negative_slope
        )
        self.reverse_map = ReverseCNN(
            y_channels, hidden_channels, num_res_blocks, negative_slope
        )

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

def single_sample_probe_jcp(model: DeceptronAdvDiffModel, x: torch.Tensor, num_probes: int = 1) -> torch.Tensor:
    x_shape = tuple(x.shape)
    x_vec = x.reshape(-1)
    total = x_vec.new_tensor(0.0)

    def f_single(inp: torch.Tensor) -> torch.Tensor:
        inp_reshaped = inp.reshape(1, *x_shape)
        return model.f(inp_reshaped).reshape(-1)

    y0 = f_single(x_vec)

    def g_single(inp: torch.Tensor) -> torch.Tensor:
        inp_reshaped = inp.reshape(1, cfg.y_channels, cfg.nx, cfg.ny)
        return model.g(inp_reshaped).reshape(-1)

    for _ in range(num_probes):
        xi = torch.empty_like(x_vec).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        _, jf_xi = jvp(f_single, (x_vec,), (xi,))
        _, jg_jf_xi = jvp(g_single, (y0,), (jf_xi,))
        total = total + torch.mean((jg_jf_xi - xi) ** 2)

    return total / num_probes

def batch_probe_jcp(model: DeceptronAdvDiffModel, x_batch: torch.Tensor, num_probes: int = 1) -> torch.Tensor:
    vals = []
    for i in range(x_batch.shape[0]):
        vals.append(single_sample_probe_jcp(model, x_batch[i], num_probes=num_probes))
    return torch.stack(vals).mean()

@torch.no_grad()
def estimate_rjcp_dataset(model: DeceptronAdvDiffModel, x_data: torch.Tensor, num_probes: int = 2, max_samples: int = 24) -> float:
    model.eval()
    x_subset = x_data[:max_samples].to(device)
    vals = []
    for i in range(x_subset.shape[0]):
        vals.append(single_sample_probe_jcp(model, x_subset[i], num_probes=num_probes).detach().cpu())
    return float(torch.stack(vals).mean().item())

@torch.no_grad()
def evaluate_forward_validation_loss(model: DeceptronAdvDiffModel, loader) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device)
        y = y_cpu.to(device)
        y_pred = model.f(x)
        total += float(F.mse_loss(y_pred, y, reduction="sum").item())
        count += y.numel()
    return total / max(count, 1)

@torch.no_grad()
def evaluate_reverse_validation(model: DeceptronAdvDiffModel, loader) -> Dict[str, float]:
    model.eval()
    totals = {"reconstruction": 0.0, "cycle": 0.0, "loss": 0.0}
    count = 0

    for x_cpu, y_cpu in loader:
        x = x_cpu.to(device)
        y = y_cpu.to(device)

        x_hat = model.g(y)
        y_cyc = model.f(x_hat)

        rec_loss = F.mse_loss(x_hat, x)
        cyc_loss = F.mse_loss(y_cyc, y)
        loss = cfg.reconstruction_weight * rec_loss + cfg.cycle_weight * cyc_loss

        bs = x.shape[0]
        totals["reconstruction"] += float(rec_loss.item()) * bs
        totals["cycle"] += float(cyc_loss.item()) * bs
        totals["loss"] += float(loss.item()) * bs
        count += bs

    return {k: v / count for k, v in totals.items()}

def freeze_forward_map(model: DeceptronAdvDiffModel):
    for p in model.forward_map.parameters():
        p.requires_grad = False

def train_forward_map_only(config: AdvDiffDegradedConfig, train_loader, val_loader):
    model = DeceptronAdvDiffModel(
        config.x_channels,
        config.y_channels,
        config.hidden_channels,
        config.num_res_blocks,
        config.negative_slope,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.forward_map.parameters(),
        lr=config.forward_lr,
        weight_decay=config.forward_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.forward_epochs)

    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(config.forward_epochs):
        model.train()
        train_total = 0.0
        count = 0

        for x_cpu, y_cpu in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model.f(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.forward_map.parameters(), config.gradient_clip)
            optimizer.step()

            train_total += float(loss.item()) * x.shape[0]
            count += x.shape[0]

        scheduler.step()

        val_task = evaluate_forward_validation_loss(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(model, data["x_val"], config.jcp_num_probes_eval, config.eval_subset_rjcp)

        row = {
            "epoch": epoch + 1,
            "train_task": train_total / count,
            "val_task": val_task,
            "val_rjcp": val_rjcp,
        }
        history.append(row)

        if val_task < best_val:
            best_val = val_task
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Forward epoch {epoch+1:03d} | train_task={row['train_task']:.6f} | val_task={row['val_task']:.6f} | val_rjcp={row['val_rjcp']:.6f}")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def train_reverse_map_only(config: AdvDiffDegradedConfig, model_init: DeceptronAdvDiffModel, train_loader, val_loader):
    model = DeceptronAdvDiffModel(
        config.x_channels,
        config.y_channels,
        config.hidden_channels,
        config.num_res_blocks,
        config.negative_slope,
    ).to(device)

    model.forward_map.load_state_dict({k: v.detach().clone() for k, v in model_init.forward_map.state_dict().items()})
    freeze_forward_map(model)

    optimizer = torch.optim.Adam(
        model.reverse_map.parameters(),
        lr=config.reverse_lr,
        weight_decay=config.reverse_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.reverse_epochs)

    history = []
    best_state = None
    best_score = float("inf")

    for epoch in range(config.reverse_epochs):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "cycle": 0.0}
        count = 0

        for x_cpu, y_cpu in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)

            x_hat = model.g(y)
            y_tilde = y + config.y_tilde_noise * torch.randn_like(y)
            y_cyc = model.f(model.g(y_tilde))

            rec_loss = F.mse_loss(x_hat, x)
            cyc_loss = F.mse_loss(y_cyc, y_tilde)
            loss = config.reconstruction_weight * rec_loss + config.cycle_weight * cyc_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            optimizer.step()

            bs = x.shape[0]
            count += bs
            totals["loss"] += float(loss.item()) * bs
            totals["reconstruction"] += float(rec_loss.item()) * bs
            totals["cycle"] += float(cyc_loss.item()) * bs

        scheduler.step()

        val_stats = evaluate_reverse_validation(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(model, data["x_val"], config.jcp_num_probes_eval, config.eval_subset_rjcp)

        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / count,
            "train_reconstruction": totals["reconstruction"] / count,
            "train_cycle": totals["cycle"] / count,
            "val_loss": val_stats["loss"],
            "val_reconstruction": val_stats["reconstruction"],
            "val_cycle": val_stats["cycle"],
            "val_rjcp": val_rjcp,
        }
        history.append(row)

        score = row["val_reconstruction"] + 0.05 * row["val_rjcp"]
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Reverse epoch {epoch+1:03d} | val_reconstruction={row['val_reconstruction']:.6f} | val_rjcp={row['val_rjcp']:.6f}")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

def train_reverse_map_with_jcp_fixed_forward(
    config: AdvDiffDegradedConfig,
    model_init: DeceptronAdvDiffModel,
    train_loader,
    val_loader,
    use_jcp: bool,
    reference_reconstruction: float,
):
    model = DeceptronAdvDiffModel(
        config.x_channels,
        config.y_channels,
        config.hidden_channels,
        config.num_res_blocks,
        config.negative_slope,
    ).to(device)

    model.forward_map.load_state_dict({k: v.detach().clone() for k, v in model_init.forward_map.state_dict().items()})
    model.reverse_map.load_state_dict({k: v.detach().clone() for k, v in model_init.reverse_map.state_dict().items()})
    freeze_forward_map(model)

    optimizer = torch.optim.Adam(
        model.reverse_map.parameters(),
        lr=config.reverse_jcp_lr,
        weight_decay=config.reverse_jcp_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.reverse_jcp_epochs)

    history = []
    best_state = None
    best_score = float("inf")

    for epoch in range(config.reverse_jcp_epochs):
        model.train()
        totals = {"loss": 0.0, "reconstruction": 0.0, "cycle": 0.0, "probe_jcp": 0.0}
        count = 0

        for x_cpu, y_cpu in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)

            x_hat = model.g(y)
            y_tilde = y + config.y_tilde_noise * torch.randn_like(y)
            y_cyc = model.f(model.g(y_tilde))

            rec_loss = F.mse_loss(x_hat, x)
            cyc_loss = F.mse_loss(y_cyc, y_tilde)

            if use_jcp:
                bs_jcp = min(config.jcp_batch_subsample, x.shape[0])
                y_subset = y[:bs_jcp]
                with torch.no_grad():
                    x_pred_subset = model.g(y_subset)

                probe_jcp_loss = config.probe_jcp_weight * batch_probe_jcp(
                    model,
                    x_pred_subset,
                    num_probes=config.jcp_num_probes_train,
                )
            else:
                probe_jcp_loss = torch.tensor(0.0, device=device)

            loss = config.reconstruction_weight * rec_loss + config.cycle_weight * cyc_loss + probe_jcp_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            optimizer.step()

            bs = x.shape[0]
            count += bs
            totals["loss"] += float(loss.item()) * bs
            totals["reconstruction"] += float(rec_loss.item()) * bs
            totals["cycle"] += float(cyc_loss.item()) * bs
            totals["probe_jcp"] += float(probe_jcp_loss.item()) * bs

        scheduler.step()

        val_stats = evaluate_reverse_validation(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(model, data["x_val"], config.jcp_num_probes_eval, config.eval_subset_rjcp)

        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / count,
            "train_reconstruction": totals["reconstruction"] / count,
            "train_cycle": totals["cycle"] / count,
            "train_probe_jcp": totals["probe_jcp"] / count,
            "val_loss": val_stats["loss"],
            "val_reconstruction": val_stats["reconstruction"],
            "val_cycle": val_stats["cycle"],
            "val_rjcp": val_rjcp,
        }
        history.append(row)

        recon_ratio = row["val_reconstruction"] / max(reference_reconstruction, 1e-12)
        if recon_ratio <= config.recon_guard_factor:
            score = row["val_rjcp"] + 1e-3 * row["val_loss"]
        else:
            score = row["val_rjcp"] + 1000.0 * (recon_ratio - config.recon_guard_factor)

        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Reverse+JCP epoch {epoch+1:03d} | val_reconstruction={row['val_reconstruction']:.6f} | val_rjcp={row['val_rjcp']:.6f} | probe_jcp={row['train_probe_jcp']:.6f}")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

model_stage1, history_forward = train_forward_map_only(cfg, forward_train_loader, forward_val_loader)
print("best stage-1 val_task:", history_forward["val_task"].min())

model_stage2, history_reverse = train_reverse_map_only(cfg, model_stage1, reverse_train_loader, reverse_val_loader)
print("best stage-2 val_reconstruction:", history_reverse["val_reconstruction"].min())
print("best stage-2 val_rjcp:", history_reverse["val_rjcp"].min())

reference_recon = float(history_reverse["val_reconstruction"].min())

model_plus_jcp, history_reverse_jcp_plus = train_reverse_map_with_jcp_fixed_forward(
    cfg, model_stage2, reverse_jcp_train_loader, reverse_jcp_val_loader, True, reference_recon
)

model_minus_jcp, history_reverse_jcp_minus = train_reverse_map_with_jcp_fixed_forward(
    cfg, model_stage2, reverse_jcp_train_loader, reverse_jcp_val_loader, False, reference_recon
)

val_stats_plus = evaluate_reverse_validation(model_plus_jcp, reverse_jcp_val_loader)
val_stats_minus = evaluate_reverse_validation(model_minus_jcp, reverse_jcp_val_loader)

print("+JCP val_reconstruction:", val_stats_plus["reconstruction"])
print("-JCP val_reconstruction:", val_stats_minus["reconstruction"])
print("+JCP val RJCP:", estimate_rjcp_dataset(model_plus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp))
print("-JCP val RJCP:", estimate_rjcp_dataset(model_minus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp))

def rmse_x(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((x_hat - x_true) ** 2)).item()

def phi_value(forward_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.mean((forward_model(x) - y_target) ** 2)

def phi_grad(forward_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor):
    x_req = x.detach().clone().requires_grad_(True)
    residual = forward_model(x_req) - y_target
    phi = 0.5 * torch.mean(residual * residual)
    grad = torch.autograd.grad(phi, x_req)[0]
    return phi.detach(), grad.detach(), residual.detach()

def normalized_residual(forward_model: nn.Module, x: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((forward_model(x) - y_target) ** 2))

def clamp_box(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, 0.0, 1.0)

def armijo_accept(forward_model: nn.Module, x_t: torch.Tensor, p: torch.Tensor, y_target: torch.Tensor, rho: float, c: float):
    phi_t, grad_t, _ = phi_grad(forward_model, x_t, y_target)
    x_trial = clamp_box((1.0 - rho) * x_t + rho * (x_t + p))
    phi_trial = phi_value(forward_model, x_trial, y_target)
    rhs = phi_t + c * rho * torch.sum(grad_t * p)
    return x_trial, phi_trial, rhs

def jvp_forward(forward_model: nn.Module, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    def f_single(inp):
        return forward_model(inp)
    _, jv = jvp(f_single, (x,), (v,))
    return jv

def vjp_forward(forward_model: nn.Module, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().requires_grad_(True)
    y = forward_model(x_req)
    scalar = torch.sum(y * u)
    return torch.autograd.grad(scalar, x_req)[0]

def normal_matvec(forward_model: nn.Module, x: torch.Tensor, v: torch.Tensor, damping: float = 0.0) -> torch.Tensor:
    jv = jvp_forward(forward_model, x, v)
    jt_j_v = vjp_forward(forward_model, x, jv)
    return jt_j_v / float(jv.numel()) + damping * v

def conjugate_gradient(matvec, b, max_iter=30, tol=1e-6):
    x = torch.zeros_like(b)
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.sum(r * r)

    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / torch.sum(p * Ap).clamp_min(1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new).item() < tol:
            break
        p = r + (rs_new / rs_old.clamp_min(1e-12)) * p
        rs_old = rs_new

    return x

def solve_dipg(model: DeceptronAdvDiffModel, y_target: torch.Tensor, x0: torch.Tensor, config: AdvDiffDegradedConfig, num_probes_eval: int = 2) -> Dict:
    model.eval()
    x = x0.clone().to(device)

    accepted_steps = 0
    alpha_accept = []
    step_norm_accept = []
    cosine_accept = []
    residual_drop_accept = []

    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, residual_t = phi_grad(model.f, x, y_target)
        residual_before = torch.sqrt(torch.mean(residual_t * residual_t)).item()

        if residual_before <= config.tolerance_eps:
            break

        grad_flat = grad_t.reshape(-1)
        grad_norm = torch.norm(grad_flat).item()

        alpha = config.dipg_step_size
        accepted = False
        y_t = model.f(x)

        for _ in range(config.max_backtracking_steps + 1):
            y_proposed = y_t - alpha * residual_t
            x_proposed = model.g(y_proposed)
            p = x_proposed - x

            x_trial, phi_trial, rhs = armijo_accept(model.f, x, p, y_target, config.rho, config.armijo_c)

            if phi_trial <= rhs:
                step_flat = p.reshape(-1)
                step_norm = torch.norm(step_flat).item()

                cos_val = np.nan
                if grad_norm > 1e-12 and step_norm > 1e-12:
                    cos_val = torch.dot(step_flat, (-grad_flat)) / (torch.norm(step_flat) * torch.norm(grad_flat))
                    cos_val = float(cos_val.item())

                x = x_trial.detach()
                residual_after = normalized_residual(model.f, x, y_target).item()

                accepted = True
                accepted_steps += 1
                alpha_accept.append(alpha)
                step_norm_accept.append(step_norm)
                cosine_accept.append(cos_val)
                residual_drop_accept.append(residual_before - residual_after)
                break

            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    final_residual = normalized_residual(model.f, x, y_target).item()
    final_rjcp = float(single_sample_probe_jcp(model, x.detach().squeeze(0), num_probes=num_probes_eval).detach().item())

    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(False),
        "final_residual": final_residual,
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
        "final_rjcp": final_rjcp,
        "residual_drop_per_accept": float(np.mean(residual_drop_accept)) if len(residual_drop_accept) else np.nan,
        "alpha_accept": float(np.mean(alpha_accept)) if len(alpha_accept) else np.nan,
        "step_norm_accept": float(np.mean(step_norm_accept)) if len(step_norm_accept) else np.nan,
        "cosine_with_neg_grad": float(np.mean(cosine_accept)) if len(cosine_accept) else np.nan,
    }

def solve_gradient_descent(forward_model: nn.Module, y_target: torch.Tensor, x0: torch.Tensor, config: AdvDiffDegradedConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, residual_t = phi_grad(forward_model, x, y_target)
        residual_before = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if residual_before <= config.tolerance_eps:
            break

        alpha = config.gd_step_size
        accepted = False

        for _ in range(config.max_backtracking_steps + 1):
            p = -alpha * grad_t
            x_trial, phi_trial, rhs = armijo_accept(forward_model, x, p, y_target, config.rho, config.armijo_c)
            if phi_trial <= rhs:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(False),
        "final_residual": normalized_residual(forward_model, x, y_target).item(),
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
        "final_rjcp": np.nan,
        "residual_drop_per_accept": np.nan,
        "alpha_accept": np.nan,
        "step_norm_accept": np.nan,
        "cosine_with_neg_grad": np.nan,
    }

def solve_gauss_newton(forward_model: nn.Module, y_target: torch.Tensor, x0: torch.Tensor, config: AdvDiffDegradedConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, residual_t = phi_grad(forward_model, x, y_target)
        residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if residual_value <= config.tolerance_eps:
            break

        rhs = -vjp_forward(forward_model, x, residual_t) / float(residual_t.numel())

        def A_fn(v):
            return normal_matvec(forward_model, x, v, damping=0.0)

        dx = conjugate_gradient(A_fn, rhs, max_iter=config.cg_max_iter, tol=config.cg_tol)

        alpha = config.gn_step_size
        accepted = False
        for _ in range(config.max_backtracking_steps + 1):
            p = alpha * dx
            x_trial, phi_trial, rhs_armijo = armijo_accept(forward_model, x, p, y_target, config.rho, config.armijo_c)
            if phi_trial <= rhs_armijo:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(False),
        "final_residual": normalized_residual(forward_model, x, y_target).item(),
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
        "final_rjcp": np.nan,
        "residual_drop_per_accept": np.nan,
        "alpha_accept": np.nan,
        "step_norm_accept": np.nan,
        "cosine_with_neg_grad": np.nan,
    }

def solve_lm(forward_model: nn.Module, y_target: torch.Tensor, x0: torch.Tensor, config: AdvDiffDegradedConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, residual_t = phi_grad(forward_model, x, y_target)
        residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if residual_value <= config.tolerance_eps:
            break

        rhs = -vjp_forward(forward_model, x, residual_t) / float(residual_t.numel())

        def A_fn(v):
            return normal_matvec(forward_model, x, v, damping=config.lm_damping)

        dx = conjugate_gradient(A_fn, rhs, max_iter=config.cg_max_iter, tol=config.cg_tol)

        alpha = config.lm_step_size
        accepted = False
        for _ in range(config.max_backtracking_steps + 1):
            p = alpha * dx
            x_trial, phi_trial, rhs_armijo = armijo_accept(forward_model, x, p, y_target, config.rho, config.armijo_c)
            if phi_trial <= rhs_armijo:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(False),
        "final_residual": normalized_residual(forward_model, x, y_target).item(),
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
        "final_rjcp": np.nan,
        "residual_drop_per_accept": np.nan,
        "alpha_accept": np.nan,
        "step_norm_accept": np.nan,
        "cosine_with_neg_grad": np.nan,
    }

def solve_lbfgs(forward_model: nn.Module, y_target: torch.Tensor, x0: torch.Tensor, config: AdvDiffDegradedConfig) -> Dict:
    x = nn.Parameter(x0.clone().to(device).contiguous())
    y_target = y_target.to(device).contiguous()

    optimizer = torch.optim.LBFGS(
        [x],
        lr=config.lbfgs_lr,
        max_iter=config.lbfgs_max_iter,
        history_size=config.lbfgs_history_size,
        tolerance_grad=config.lbfgs_tolerance_grad,
        tolerance_change=config.lbfgs_tolerance_change,
        line_search_fn="strong_wolfe",
    )

    iteration_counter = {"n": 0}
    t0 = time.perf_counter()

    def closure():
        optimizer.zero_grad(set_to_none=True)
        x_clamped = torch.clamp(x, 0.0, 1.0).contiguous()
        loss = 0.5 * torch.mean((forward_model(x_clamped) - y_target) ** 2)
        loss.backward()
        if x.grad is not None:
            x.grad = x.grad.contiguous()
        iteration_counter["n"] += 1
        return loss

    optimizer.step(closure)

    with torch.no_grad():
        x_final = torch.clamp(x.detach(), 0.0, 1.0).contiguous()

    elapsed = time.perf_counter() - t0
    return {
        "x_hat": x_final.cpu(),
        "iters": iteration_counter["n"],
        "success": float(False),
        "final_residual": normalized_residual(forward_model, x_final, y_target).item(),
        "accepted_frac": 1.0,
        "time_sec": elapsed,
        "final_rjcp": np.nan,
        "residual_drop_per_accept": np.nan,
        "alpha_accept": np.nan,
        "step_norm_accept": np.nan,
        "cosine_with_neg_grad": np.nan,
    }

def evaluate_advdiff_methods(case_x: torch.Tensor, case_y: torch.Tensor, model_plus: DeceptronAdvDiffModel, model_minus: DeceptronAdvDiffModel, config: AdvDiffDegradedConfig) -> pd.DataFrame:
    rows = []

    method_specs = [
        ("+JCP", "D-IPG", lambda y_star, x0: solve_dipg(model_plus, y_star, x0, config, num_probes_eval=2)),
        ("-JCP", "D-IPG", lambda y_star, x0: solve_dipg(model_minus, y_star, x0, config, num_probes_eval=2)),
        ("Baseline", "Gradient Descent", lambda y_star, x0: solve_gradient_descent(model_plus.f, y_star, x0, config)),
        ("Baseline", "GN", lambda y_star, x0: solve_gauss_newton(model_plus.f, y_star, x0, config)),
        ("Baseline", "LM", lambda y_star, x0: solve_lm(model_plus.f, y_star, x0, config)),
        ("Baseline", "L-BFGS", lambda y_star, x0: solve_lbfgs(model_plus.f, y_star, x0, config)),
    ]

    for ablation, method_name, method_function in method_specs:
        per_instance = []

        for i in range(case_x.shape[0]):
            x_true = case_x[i:i+1]
            y_star = case_y[i:i+1].to(device)
            x0 = torch.full_like(x_true, 0.5).to(device)

            result = method_function(y_star, x0)
            rmse_val = rmse_x(result["x_hat"], x_true)
            result["success"] = float(rmse_val <= config.success_rmse_threshold)

            per_instance.append({
                "rmse_x": rmse_val,
                **{k: v for k, v in result.items() if k != "x_hat"},
            })

        df = pd.DataFrame(per_instance)
        summary = {
            "ablation": ablation,
            "method": method_name,
            "num_instances": len(df),
            "success_rate": df["success"].mean(),
            "mean_iters": df["iters"].mean(),
            "median_iters": df["iters"].median(),
            "mean_rmse_x": df["rmse_x"].mean(),
            "mean_final_residual": df["final_residual"].mean(),
            "mean_accepted_frac": df["accepted_frac"].mean(),
            "mean_time_sec": df["time_sec"].mean(),
            "mean_final_rjcp": df["final_rjcp"].mean(),
            "mean_residual_drop_per_accept": df["residual_drop_per_accept"].mean(),
            "mean_alpha_accept": df["alpha_accept"].mean(),
            "mean_step_norm_accept": df["step_norm_accept"].mean(),
            "mean_cosine_with_neg_grad": df["cosine_with_neg_grad"].mean(),
        }
        rows.append(summary)
        print("done:", ablation, method_name)

    return pd.DataFrame(rows)

main_results = evaluate_advdiff_methods(
    data["x_test"],
    data["y_test_norm"],
    model_plus_jcp,
    model_minus_jcp,
    cfg,
)

print(main_results.round(6).to_string(index=False))

rjcp_summary = pd.DataFrame([
    {
        "ablation": "+JCP",
        "val_reconstruction": evaluate_reverse_validation(model_plus_jcp, reverse_jcp_val_loader)["reconstruction"],
        "val_rjcp": estimate_rjcp_dataset(model_plus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
        "test_rjcp": estimate_rjcp_dataset(model_plus_jcp, data["x_test"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
    },
    {
        "ablation": "-JCP",
        "val_reconstruction": evaluate_reverse_validation(model_minus_jcp, reverse_jcp_val_loader)["reconstruction"],
        "val_rjcp": estimate_rjcp_dataset(model_minus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
        "test_rjcp": estimate_rjcp_dataset(model_minus_jcp, data["x_test"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
    },
])

print(rjcp_summary.round(6).to_string(index=False))

history_forward.to_csv("advdiff_degraded_stage1_forward_history.csv", index=False)
history_reverse.to_csv("advdiff_degraded_stage2_reverse_history.csv", index=False)
history_reverse_jcp_plus.to_csv("advdiff_degraded_stage3_reverse_plus_jcp_history.csv", index=False)
history_reverse_jcp_minus.to_csv("advdiff_degraded_stage3_reverse_minus_jcp_history.csv", index=False)

main_results.to_csv("advdiff_degraded_main_results.csv", index=False)
rjcp_summary.to_csv("advdiff_degraded_rjcp_summary.csv", index=False)

torch.save(model_stage1.state_dict(), "advdiff_degraded_model_stage1.pt")
torch.save(model_stage2.state_dict(), "advdiff_degraded_model_stage2.pt")
torch.save(model_plus_jcp.state_dict(), "advdiff_degraded_model_plus_jcp.pt")
torch.save(model_minus_jcp.state_dict(), "advdiff_degraded_model_minus_jcp.pt")
