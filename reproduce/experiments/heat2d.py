"""Heat-2D initial temperature recovery.

Seed-7 experiment from reproduce/notebooks/Heat2D.ipynb.
The original notebook also contains extended analyses.
"""

from _output import prepare_output

prepare_output("heat2d")


# Setup

import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

import numpy as np
import pandas as pd
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
print("torch:", torch.__version__)
print("numpy:", np.__version__)
print("pandas:", pd.__version__)

@dataclass
class Heat2DCNNConfig:
    nx: int = 16
    ny: int = 16
    dim: int = 256

    t_final: float = 0.016
    diffusivity: float = 0.12
    obs_blur_strength: float = 0.03
    obs_nonlinear_cubic: float = 0.03
    obs_nonlinear_sin: float = 0.01
    noise_std: float = 0.002

    n_train: int = 6000
    n_val: int = 500
    n_test: int = 100

    modes_x: int = 4
    modes_y: int = 4
    x_low: float = 0.0
    x_high: float = 1.0

    negative_slope: float = 0.10
    hidden_channels: int = 16

    batch_size: int = 64

    forward_epochs: int = 160
    reverse_epochs: int = 120
    reverse_jcp_epochs: int = 140

    forward_lr: float = 2e-3
    reverse_lr: float = 2e-3
    reverse_jcp_lr: float = 1e-3

    forward_weight_decay: float = 1e-6
    reverse_weight_decay: float = 1e-6
    reverse_jcp_weight_decay: float = 1e-6

    reconstruction_weight: float = 1.0
    cycle_weight: float = 0.15
    probe_jcp_weight: float = 0.5

    jcp_num_probes_train: int = 1
    jcp_num_probes_eval: int = 2
    y_tilde_noise: float = 0.006
    gradient_clip: float = 5.0
    jcp_batch_subsample: int = 8

    max_iterations: int = 100
    tolerance_eps: float = 0.40
    armijo_c: float = 1e-4
    max_backtracking_steps: int = 8
    rho: float = 0.4

    dipg_step_size: float = 1.0
    gd_step_size: float = 1.0
    gn_step_size: float = 1.0
    lm_step_size: float = 1.0
    lm_damping: float = 1e-2

    eval_subset_rjcp: int = 32

cfg = Heat2DCNNConfig()
print(asdict(cfg))

def clamp_box(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    return torch.clamp(x, low, high)

def normalized_residual(forward_model: nn.Module, x: torch.Tensor, y_target_norm: torch.Tensor) -> torch.Tensor:
    residual = forward_model(x) - y_target_norm
    return torch.sqrt(torch.mean(residual * residual))

def rmse_x(x_hat: torch.Tensor, x_true: torch.Tensor) -> float:
    return torch.sqrt(torch.mean((x_hat - x_true) ** 2)).item()

# Data generation

class NonlinearHeat2DTrueModel(nn.Module):
    def __init__(
        self,
        nx: int,
        ny: int,
        t_final: float,
        diffusivity: float,
        obs_blur_strength: float,
        obs_nonlinear_cubic: float,
        obs_nonlinear_sin: float,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.dim = nx * ny
        self.obs_blur_strength = obs_blur_strength
        self.obs_nonlinear_cubic = obs_nonlinear_cubic
        self.obs_nonlinear_sin = obs_nonlinear_sin

        gx = torch.linspace(0.0, 1.0, nx + 2)[1:-1]
        gy = torch.linspace(0.0, 1.0, ny + 2)[1:-1]
        kx = torch.arange(1, nx + 1, dtype=torch.float32)
        ky = torch.arange(1, ny + 1, dtype=torch.float32)

        basis_x = torch.sin(math.pi * gx[:, None] * kx[None, :])
        basis_y = torch.sin(math.pi * gy[:, None] * ky[None, :])
        basis_x = basis_x * math.sqrt(2.0 / (nx + 1))
        basis_y = basis_y * math.sqrt(2.0 / (ny + 1))

        decay = torch.exp(
            -diffusivity * ((math.pi * kx[:, None]) ** 2 + (math.pi * ky[None, :]) ** 2) * t_final
        )

        kernel = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, 4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32
        )
        kernel = kernel / kernel.sum()

        self.register_buffer("basis_x", basis_x)
        self.register_buffer("basis_y", basis_y)
        self.register_buffer("decay", decay)
        self.register_buffer("blur_kernel", kernel.view(1, 1, 3, 3))

    def heat_semigroup(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(-1, self.nx, self.ny)
        coeffs = torch.einsum("bxy,xk,yl->bkl", x, self.basis_x, self.basis_y)
        u_t = torch.einsum("bkl,kl,xk,yl->bxy", coeffs, self.decay, self.basis_x, self.basis_y)
        return u_t

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        u = self.heat_semigroup(x_flat)
        u_img = u.unsqueeze(1)
        blurred = F.conv2d(u_img, self.blur_kernel, padding=1).squeeze(1)
        z = (1.0 - self.obs_blur_strength) * u + self.obs_blur_strength * blurred
        y = z + self.obs_nonlinear_cubic * (z ** 3) + self.obs_nonlinear_sin * torch.sin(2.5 * z)
        return y.reshape(y.shape[0], -1)

true_model = NonlinearHeat2DTrueModel(
    nx=cfg.nx,
    ny=cfg.ny,
    t_final=cfg.t_final,
    diffusivity=cfg.diffusivity,
    obs_blur_strength=cfg.obs_blur_strength,
    obs_nonlinear_cubic=cfg.obs_nonlinear_cubic,
    obs_nonlinear_sin=cfg.obs_nonlinear_sin,
).to(device)

def sample_heat2d_initial_conditions(
    n_samples: int,
    nx: int,
    ny: int,
    modes_x: int,
    modes_y: int,
    x_low: float,
    x_high: float,
) -> torch.Tensor:
    gx = torch.linspace(0.0, 1.0, nx + 2)[1:-1]
    gy = torch.linspace(0.0, 1.0, ny + 2)[1:-1]
    kx = torch.arange(1, modes_x + 1, dtype=torch.float32)
    ky = torch.arange(1, modes_y + 1, dtype=torch.float32)

    basis_x = torch.sin(math.pi * gx[:, None] * kx[None, :])
    basis_y = torch.sin(math.pi * gy[:, None] * ky[None, :])
    basis_x = basis_x / basis_x.norm(dim=0, keepdim=True)
    basis_y = basis_y / basis_y.norm(dim=0, keepdim=True)

    coefficients = torch.randn(n_samples, modes_x, modes_y)
    coefficients = coefficients * (1.0 / torch.arange(1, modes_x + 1, dtype=torch.float32))[None, :, None]
    coefficients = coefficients * (1.0 / torch.arange(1, modes_y + 1, dtype=torch.float32))[None, None, :]

    x = torch.einsum("bkl,xk,yl->bxy", coefficients, basis_x, basis_y)

    xx = gx[:, None].repeat(1, ny)
    yy = gy[None, :].repeat(nx, 1)

    for b in range(n_samples):
        cx = float(torch.rand(1) * 0.4 + 0.3)
        cy = float(torch.rand(1) * 0.4 + 0.3)
        sx = float(torch.rand(1) * 0.05 + 0.12)
        sy = float(torch.rand(1) * 0.05 + 0.12)
        amp = float(torch.rand(1) * 0.10 + 0.03)
        bump = amp * torch.exp(-((xx - cx) ** 2 / (2 * sx ** 2) + (yy - cy) ** 2 / (2 * sy ** 2)))
        x[b] = x[b] + bump

    x_flat = x.reshape(n_samples, -1)
    x_min = x_flat.min(dim=1, keepdim=True).values
    x_max = x_flat.max(dim=1, keepdim=True).values
    x_flat = (x_flat - x_min) / (x_max - x_min + 1e-8)
    x_flat = x_low + (x_high - x_low) * x_flat
    return x_flat

@torch.no_grad()
def build_dataset(config: Heat2DCNNConfig, true_model: nn.Module) -> Dict[str, torch.Tensor]:
    n_total = config.n_train + config.n_val + config.n_test

    x_all = sample_heat2d_initial_conditions(
        n_samples=n_total,
        nx=config.nx,
        ny=config.ny,
        modes_x=config.modes_x,
        modes_y=config.modes_y,
        x_low=config.x_low,
        x_high=config.x_high,
    )

    y_clean_all = true_model(x_all.to(device)).cpu()
    y_noisy_all = y_clean_all + config.noise_std * torch.randn_like(y_clean_all)

    x_train = x_all[:config.n_train]
    x_val = x_all[config.n_train: config.n_train + config.n_val]
    x_test = x_all[-config.n_test:]

    y_clean_train = y_clean_all[:config.n_train]
    y_clean_val = y_clean_all[config.n_train: config.n_train + config.n_val]
    y_clean_test = y_clean_all[-config.n_test:]

    y_noisy_train = y_noisy_all[:config.n_train]
    y_noisy_val = y_noisy_all[config.n_train: config.n_train + config.n_val]
    y_noisy_test = y_noisy_all[-config.n_test:]

    y_mean = y_clean_train.mean(dim=0, keepdim=True)
    y_std = y_clean_train.std(dim=0, keepdim=True).clamp_min(1e-6)

    return {
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "y_clean_train": y_clean_train,
        "y_clean_val": y_clean_val,
        "y_clean_test": y_clean_test,
        "y_noisy_train": y_noisy_train,
        "y_noisy_val": y_noisy_val,
        "y_noisy_test": y_noisy_test,
        "y_clean_train_norm": (y_clean_train - y_mean) / y_std,
        "y_clean_val_norm": (y_clean_val - y_mean) / y_std,
        "y_clean_test_norm": (y_clean_test - y_mean) / y_std,
        "y_noisy_train_norm": (y_noisy_train - y_mean) / y_std,
        "y_noisy_val_norm": (y_noisy_val - y_mean) / y_std,
        "y_noisy_test_norm": (y_noisy_test - y_mean) / y_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }

data = build_dataset(cfg, true_model)

for key, value in data.items():
    if isinstance(value, torch.Tensor):
        print(key, tuple(value.shape))

forward_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_clean_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

forward_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_clean_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

reverse_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_clean_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

reverse_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_clean_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

reverse_jcp_train_loader = DataLoader(
    TensorDataset(data["x_train"], data["y_clean_train_norm"]),
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=False,
)

reverse_jcp_val_loader = DataLoader(
    TensorDataset(data["x_val"], data["y_clean_val_norm"]),
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
)

# Model definition

class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, negative_slope: float):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)
        z = self.conv2(z)
        return F.leaky_relu(x + z, negative_slope=self.negative_slope)


class ForwardCNN(nn.Module):
    def __init__(self, nx: int, ny: int, hidden_channels: int, negative_slope: float):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.negative_slope = negative_slope
        self.entry = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.block1 = ConvResidualBlock(hidden_channels, negative_slope)
        self.block2 = ConvResidualBlock(hidden_channels, negative_slope)
        self.exit = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        x = x_flat.view(-1, 1, self.nx, self.ny)
        z = F.leaky_relu(self.entry(x), negative_slope=self.negative_slope)
        z = self.block1(z)
        z = self.block2(z)
        y = self.exit(z)
        return y.view(y.shape[0], -1)


class ReverseCNN(nn.Module):
    def __init__(self, nx: int, ny: int, hidden_channels: int, negative_slope: float):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.negative_slope = negative_slope
        self.entry = nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1)
        self.block1 = ConvResidualBlock(hidden_channels, negative_slope)
        self.block2 = ConvResidualBlock(hidden_channels, negative_slope)
        self.exit = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, y_flat: torch.Tensor) -> torch.Tensor:
        y = y_flat.view(-1, 1, self.nx, self.ny)
        z = F.leaky_relu(self.entry(y), negative_slope=self.negative_slope)
        z = self.block1(z)
        z = self.block2(z)
        x = self.exit(z)
        return x.view(x.shape[0], -1)


class DeceptronCNNModel(nn.Module):
    def __init__(self, nx: int, ny: int, hidden_channels: int, negative_slope: float):
        super().__init__()
        self.forward_map = ForwardCNN(nx, ny, hidden_channels, negative_slope)
        self.reverse_map = ReverseCNN(nx, ny, hidden_channels, negative_slope)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)

# Probe-based JCP is applied only after the forward map is frozen.
# This keeps the +JCP / -JCP split on the inverse-map side only.

def single_sample_probe_jcp(model: DeceptronCNNModel, x: torch.Tensor, num_probes: int = 1) -> torch.Tensor:
    x = x.reshape(-1)
    dimension = x.numel()
    total = x.new_tensor(0.0)

    def f_single(inp: torch.Tensor) -> torch.Tensor:
        inp = inp.reshape(-1)
        return model.f(inp.unsqueeze(0)).reshape(-1)

    def g_single(inp: torch.Tensor) -> torch.Tensor:
        inp = inp.reshape(-1)
        return model.g(inp.unsqueeze(0)).reshape(-1)

    for _ in range(num_probes):
        xi = torch.empty(dimension, device=x.device).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        y = f_single(x)
        _, jf_xi = jvp(f_single, (x,), (xi,))
        _, jg_jf_xi = jvp(g_single, (y,), (jf_xi,))
        total = total + torch.mean((jg_jf_xi - xi) ** 2)

    return total / num_probes


def batch_probe_jcp(model: DeceptronCNNModel, x_batch: torch.Tensor, num_probes: int = 1) -> torch.Tensor:
    values = []
    for i in range(x_batch.shape[0]):
        values.append(single_sample_probe_jcp(model, x_batch[i], num_probes=num_probes))
    return torch.stack(values).mean()


@torch.no_grad()
def estimate_rjcp_dataset(model: DeceptronCNNModel, x_data: torch.Tensor, num_probes: int = 2, max_samples: int = 32) -> float:
    model.eval()
    x_subset = x_data[:max_samples].to(device)
    values = []
    for i in range(x_subset.shape[0]):
        values.append(single_sample_probe_jcp(model, x_subset[i], num_probes=num_probes).detach().cpu())
    return float(torch.stack(values).mean().item())

@torch.no_grad()
def evaluate_forward_validation_loss(model: DeceptronCNNModel, loader) -> float:
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

# Training

def train_forward_map_only(config: Heat2DCNNConfig, train_loader, val_loader):
    model = DeceptronCNNModel(
        nx=config.nx,
        ny=config.ny,
        hidden_channels=config.hidden_channels,
        negative_slope=config.negative_slope,
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
            y_pred = model.f(x)
            loss = F.mse_loss(y_pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.forward_map.parameters(), config.gradient_clip)
            optimizer.step()

            train_total += float(loss.item()) * x.shape[0]
            count += x.shape[0]

        scheduler.step()

        val_task = evaluate_forward_validation_loss(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(
            model,
            data["x_val"],
            num_probes=config.jcp_num_probes_eval,
            max_samples=min(config.eval_subset_rjcp, data["x_val"].shape[0]),
        )

        row = {
            "epoch": epoch + 1,
            "train_task": train_total / count,
            "val_task": val_task,
            "val_rjcp": val_rjcp,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        if val_task < best_val:
            best_val = val_task
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Forward epoch {epoch+1:03d} | "
                f"train_task={row['train_task']:.5f} | "
                f"val_task={row['val_task']:.5f} | "
                f"val_rjcp={row['val_rjcp']:.5f}"
            )

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

model_stage1, history_forward = train_forward_map_only(cfg, forward_train_loader, forward_val_loader)
print("best stage-1 val_task:", history_forward["val_task"].min())

def freeze_forward_map(model: DeceptronCNNModel):
    for parameter in model.forward_map.parameters():
        parameter.requires_grad = False


def train_reverse_map_only(config: Heat2DCNNConfig, model_init: DeceptronCNNModel, train_loader, val_loader):
    model = DeceptronCNNModel(
        nx=config.nx,
        ny=config.ny,
        hidden_channels=config.hidden_channels,
        negative_slope=config.negative_slope,
    ).to(device)
    model.load_state_dict({k: v.detach().clone() for k, v in model_init.state_dict().items()})

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
        reconstruction_total = 0.0
        cycle_total = 0.0
        loss_total = 0.0
        count = 0

        for x_cpu, y_cpu in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                y_pred = model.f(x)

            x_rec = model.g(y_pred)
            y_tilde = y + config.y_tilde_noise * torch.randn_like(y)
            y_cyc = model.f(model.g(y_tilde))

            reconstruction_loss = F.mse_loss(x_rec, x)
            cycle_loss = F.mse_loss(y_cyc, y_tilde)
            loss = config.reconstruction_weight * reconstruction_loss + config.cycle_weight * cycle_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            optimizer.step()

            batch_size = x.shape[0]
            reconstruction_total += float(reconstruction_loss.item()) * batch_size
            cycle_total += float(cycle_loss.item()) * batch_size
            loss_total += float(loss.item()) * batch_size
            count += batch_size

        scheduler.step()

        val_task = evaluate_forward_validation_loss(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(
            model,
            data["x_val"],
            num_probes=config.jcp_num_probes_eval,
            max_samples=min(config.eval_subset_rjcp, data["x_val"].shape[0]),
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": loss_total / count,
            "train_reconstruction": reconstruction_total / count,
            "train_cycle": cycle_total / count,
            "val_task": val_task,
            "val_rjcp": val_rjcp,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        score = val_rjcp + 1e-3 * val_task
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(
                f"Reverse epoch {epoch+1:03d} | "
                f"reconstruction={row['train_reconstruction']:.5f} | "
                f"cycle={row['train_cycle']:.5f} | "
                f"val_task={row['val_task']:.5f} | "
                f"val_rjcp={row['val_rjcp']:.5f}"
            )

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

model_stage2, history_reverse = train_reverse_map_only(cfg, model_stage1, reverse_train_loader, reverse_val_loader)
print("best stage-2 val_rjcp:", history_reverse["val_rjcp"].min())

def train_reverse_map_with_jcp_fixed_forward(
    config: Heat2DCNNConfig,
    model_init: DeceptronCNNModel,
    train_loader,
    val_loader,
    use_jcp: bool,
):
    model = DeceptronCNNModel(
        nx=config.nx,
        ny=config.ny,
        hidden_channels=config.hidden_channels,
        negative_slope=config.negative_slope,
    ).to(device)
    model.load_state_dict({k: v.detach().clone() for k, v in model_init.state_dict().items()})

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
        totals = {
            "loss": 0.0,
            "reconstruction": 0.0,
            "cycle": 0.0,
            "probe_jcp": 0.0,
        }
        count = 0

        for x_cpu, y_cpu in train_loader:
            x = x_cpu.to(device)
            y = y_cpu.to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                y_pred = model.f(x)

            x_rec = model.g(y_pred)
            y_tilde = y + config.y_tilde_noise * torch.randn_like(y)
            y_cyc = model.f(model.g(y_tilde))

            reconstruction_loss = F.mse_loss(x_rec, x)
            cycle_loss = F.mse_loss(y_cyc, y_tilde)

            if use_jcp:
                x_subset = x[: min(config.jcp_batch_subsample, x.shape[0])]
                probe_jcp_loss = config.probe_jcp_weight * batch_probe_jcp(
                    model, x_subset, num_probes=config.jcp_num_probes_train
                )
            else:
                probe_jcp_loss = torch.tensor(0.0, device=device)

            loss = (
                config.reconstruction_weight * reconstruction_loss
                + config.cycle_weight * cycle_loss
                + probe_jcp_loss
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.reverse_map.parameters(), config.gradient_clip)
            optimizer.step()

            batch_size = x.shape[0]
            count += batch_size
            totals["loss"] += float(loss.item()) * batch_size
            totals["reconstruction"] += float(reconstruction_loss.item()) * batch_size
            totals["cycle"] += float(cycle_loss.item()) * batch_size
            totals["probe_jcp"] += float(probe_jcp_loss.item()) * batch_size

        scheduler.step()

        val_task = evaluate_forward_validation_loss(model, val_loader)
        val_rjcp = estimate_rjcp_dataset(
            model,
            data["x_val"],
            num_probes=config.jcp_num_probes_eval,
            max_samples=min(config.eval_subset_rjcp, data["x_val"].shape[0]),
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": totals["loss"] / count,
            "train_reconstruction": totals["reconstruction"] / count,
            "train_cycle": totals["cycle"] / count,
            "train_probe_jcp": totals["probe_jcp"] / count,
            "val_task": val_task,
            "val_rjcp": val_rjcp,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)

        score = val_rjcp + 1e-3 * val_task
        if score < best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Reverse+JCP epoch {epoch+1:03d} | "
                f"train_loss={row['train_loss']:.5f} | "
                f"val_task={row['val_task']:.5f} | "
                f"val_rjcp={row['val_rjcp']:.5f} | "
                f"probe_jcp={row['train_probe_jcp']:.5f}"
            )

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)

model_plus_jcp, history_reverse_jcp_plus = train_reverse_map_with_jcp_fixed_forward(
    cfg,
    model_stage2,
    reverse_jcp_train_loader,
    reverse_jcp_val_loader,
    use_jcp=True,
)

model_minus_jcp, history_reverse_jcp_minus = train_reverse_map_with_jcp_fixed_forward(
    cfg,
    model_stage2,
    reverse_jcp_train_loader,
    reverse_jcp_val_loader,
    use_jcp=False,
)

print("+JCP val_task:", evaluate_forward_validation_loss(model_plus_jcp, reverse_jcp_val_loader))
print("-JCP val_task:", evaluate_forward_validation_loss(model_minus_jcp, reverse_jcp_val_loader))
print("+JCP val RJCP:", estimate_rjcp_dataset(model_plus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp))
print("-JCP val RJCP:", estimate_rjcp_dataset(model_minus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp))

# Evaluation

def phi_value(forward_model: nn.Module, x: torch.Tensor, y_target_norm: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1)
    y_target_norm = y_target_norm.reshape(-1)
    residual = forward_model(x.unsqueeze(0)).reshape(-1) - y_target_norm
    return 0.5 * torch.mean(residual * residual)

def phi_grad(forward_model: nn.Module, x: torch.Tensor, y_target_norm: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_req = x.detach().clone().reshape(-1).requires_grad_(True)
    y_target_norm = y_target_norm.reshape(-1)

    y_pred = forward_model(x_req.unsqueeze(0)).reshape(-1)
    residual = y_pred - y_target_norm
    phi = 0.5 * torch.mean(residual * residual)
    grad = torch.autograd.grad(phi, x_req)[0]
    return phi.detach(), grad.detach().reshape(-1), residual.detach().reshape(-1)

def full_jacobian_single_output_model(forward_model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    x_req = x.detach().clone().reshape(-1).requires_grad_(True)

    def f_single(inp: torch.Tensor) -> torch.Tensor:
        inp = inp.reshape(-1)
        return forward_model(inp.unsqueeze(0)).reshape(-1)

    J = torch.autograd.functional.jacobian(f_single, x_req, vectorize=True)
    return J.detach()

def solve_linear_system(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    try:
        return torch.linalg.solve(A, b)
    except RuntimeError:
        return torch.linalg.lstsq(A, b).solution

def armijo_accept(
    forward_model: nn.Module,
    x_t: torch.Tensor,
    p: torch.Tensor,
    y_target_norm: torch.Tensor,
    rho: float,
    c: float,
    x_low: float,
    x_high: float,
):
    x_t = x_t.reshape(-1)
    p = p.reshape(-1)
    y_target_norm = y_target_norm.reshape(-1)

    phi_t, grad_t, _ = phi_grad(forward_model, x_t, y_target_norm)
    x_trial = clamp_box((1.0 - rho) * x_t + rho * (x_t + p), x_low, x_high)
    phi_trial = phi_value(forward_model, x_trial, y_target_norm)
    rhs = phi_t + c * rho * torch.dot(grad_t.reshape(-1), p.reshape(-1))
    return x_trial.reshape(-1), phi_trial, rhs

# D-IPG uses the learned inverse map g in addition to the shared forward surrogate f.
# GD, GN, and LM use only f.

def solve_dipg(model: DeceptronCNNModel, y_target_norm: torch.Tensor, x0: torch.Tensor, config: Heat2DCNNConfig, num_probes_eval: int = 2) -> Dict:
    model.eval()
    x = x0.clone().to(device)

    accepted_steps = 0
    residual_drop_sum = 0.0
    accepted_alpha_sum = 0.0
    accepted_step_norm_sum = 0.0
    accepted_cosine_sum = 0.0
    accepted_count = 0

    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, residual_t = phi_grad(model.f, x, y_target_norm)
        normalized_residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if normalized_residual_value <= config.tolerance_eps:
            break

        alpha = config.dipg_step_size
        accepted = False
        y_t = model.f(x)

        for _ in range(config.max_backtracking_steps + 1):
            y_proposed = y_t - alpha * residual_t
            x_proposed = model.g(y_proposed)
            p = x_proposed - x

            x_trial, phi_trial, rhs = armijo_accept(
                model.f,
                x,
                p,
                y_target_norm,
                config.rho,
                config.armijo_c,
                config.x_low,
                config.x_high,
            )

            if phi_trial <= rhs:
                x_new = x_trial.detach()

                with torch.no_grad():
                    new_residual = normalized_residual(model.f, x_new, y_target_norm).item()
                    old_residual = normalized_residual_value
                    residual_drop_sum += (old_residual - new_residual)

                    step_vector = (x_new - x).flatten()
                    grad_vector = grad_t.flatten()
                    step_norm = torch.norm(step_vector).item()
                    grad_norm = torch.norm(grad_vector).item()

                    if step_norm > 0.0 and grad_norm > 0.0:
                        cosine = torch.dot(step_vector, -grad_vector).item() / (step_norm * grad_norm + 1e-12)
                    else:
                        cosine = 0.0

                    accepted_alpha_sum += alpha
                    accepted_step_norm_sum += step_norm
                    accepted_cosine_sum += cosine
                    accepted_count += 1

                x = x_new
                accepted = True
                accepted_steps += 1
                break

            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    final_residual = normalized_residual(model.f, x, y_target_norm).item()
    final_rjcp = float(
        single_sample_probe_jcp(
            model,
            x.detach().clone().requires_grad_(True),
            num_probes=num_probes_eval,
        ).detach().item()
    )

    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(final_residual <= config.tolerance_eps),
        "final_residual": final_residual,
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
        "final_rjcp": final_rjcp,
        "mean_residual_drop_per_accept": residual_drop_sum / max(accepted_count, 1),
        "mean_alpha_accept": accepted_alpha_sum / max(accepted_count, 1),
        "mean_step_norm_accept": accepted_step_norm_sum / max(accepted_count, 1),
        "mean_cosine_with_neg_grad": accepted_cosine_sum / max(accepted_count, 1),
    }


def solve_gradient_descent(forward_model: nn.Module, y_target_norm: torch.Tensor, x0: torch.Tensor, config: Heat2DCNNConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, grad_t, residual_t = phi_grad(forward_model, x, y_target_norm)
        normalized_residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if normalized_residual_value <= config.tolerance_eps:
            break

        eta = config.gd_step_size
        accepted = False

        for _ in range(config.max_backtracking_steps + 1):
            p = -eta * grad_t
            x_trial, phi_trial, rhs = armijo_accept(
                forward_model,
                x,
                p,
                y_target_norm,
                config.rho,
                config.armijo_c,
                config.x_low,
                config.x_high,
            )
            if phi_trial <= rhs:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            eta *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    final_residual = normalized_residual(forward_model, x, y_target_norm).item()

    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(final_residual <= config.tolerance_eps),
        "final_residual": final_residual,
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
    }


def solve_gauss_newton(forward_model: nn.Module, y_target_norm: torch.Tensor, x0: torch.Tensor, config: Heat2DCNNConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, residual_t = phi_grad(forward_model, x, y_target_norm)
        normalized_residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if normalized_residual_value <= config.tolerance_eps:
            break

        J = full_jacobian_single_output_model(forward_model, x)
        JTJ = J.T @ J / J.shape[0]
        JTr = J.T @ residual_t / J.shape[0]
        dx = solve_linear_system(JTJ, -JTr)

        alpha = config.gn_step_size
        accepted = False

        for _ in range(config.max_backtracking_steps + 1):
            p = alpha * dx
            x_trial, phi_trial, rhs = armijo_accept(
                forward_model,
                x,
                p,
                y_target_norm,
                config.rho,
                config.armijo_c,
                config.x_low,
                config.x_high,
            )
            if phi_trial <= rhs:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    final_residual = normalized_residual(forward_model, x, y_target_norm).item()

    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(final_residual <= config.tolerance_eps),
        "final_residual": final_residual,
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
    }


def solve_levenberg_marquardt(forward_model: nn.Module, y_target_norm: torch.Tensor, x0: torch.Tensor, config: Heat2DCNNConfig) -> Dict:
    x = x0.clone().to(device)
    accepted_steps = 0
    t0 = time.perf_counter()

    for iteration in range(config.max_iterations):
        _, _, residual_t = phi_grad(forward_model, x, y_target_norm)
        normalized_residual_value = torch.sqrt(torch.mean(residual_t * residual_t)).item()
        if normalized_residual_value <= config.tolerance_eps:
            break

        J = full_jacobian_single_output_model(forward_model, x)
        JTJ = J.T @ J / J.shape[0]
        JTr = J.T @ residual_t / J.shape[0]
        A = JTJ + config.lm_damping * torch.eye(JTJ.shape[0], device=JTJ.device)
        dx = solve_linear_system(A, -JTr)

        alpha = config.lm_step_size
        accepted = False

        for _ in range(config.max_backtracking_steps + 1):
            p = alpha * dx
            x_trial, phi_trial, rhs = armijo_accept(
                forward_model,
                x,
                p,
                y_target_norm,
                config.rho,
                config.armijo_c,
                config.x_low,
                config.x_high,
            )
            if phi_trial <= rhs:
                x = x_trial.detach()
                accepted = True
                accepted_steps += 1
                break
            alpha *= 0.5

        if not accepted:
            break

    elapsed = time.perf_counter() - t0
    final_residual = normalized_residual(forward_model, x, y_target_norm).item()

    return {
        "x_hat": x.detach().cpu(),
        "iters": iteration + 1,
        "success": float(final_residual <= config.tolerance_eps),
        "final_residual": final_residual,
        "accepted_frac": accepted_steps / max(iteration + 1, 1),
        "time_sec": elapsed,
    }

def evaluate_all_methods_on_test(model: DeceptronCNNModel, data_dict: Dict[str, torch.Tensor], config: Heat2DCNNConfig, label: str) -> pd.DataFrame:
    rows = []
    x_test = data_dict["x_test"]
    y_test_norm = data_dict["y_noisy_test_norm"]

    method_functions = {
        "D-IPG": lambda y_star, x0: solve_dipg(model, y_star, x0, config, num_probes_eval=2),
        "Gradient Descent": lambda y_star, x0: solve_gradient_descent(model.f, y_star, x0, config),
        "GN": lambda y_star, x0: solve_gauss_newton(model.f, y_star, x0, config),
        "LM": lambda y_star, x0: solve_levenberg_marquardt(model.f, y_star, x0, config),
    }

    for method_name, method_function in method_functions.items():
        per_instance = []

        for i in range(x_test.shape[0]):
            y_star = y_test_norm[i].to(device)
            x_true = x_test[i]
            x0 = torch.zeros_like(x_true)

            result = method_function(y_star, x0)
            per_instance.append({
                "rmse_x": rmse_x(result["x_hat"], x_true),
                **{k: v for k, v in result.items() if k != "x_hat"},
            })

        df = pd.DataFrame(per_instance)
        summary = {
            "ablation": label,
            "method": method_name,
            "num_instances": len(df),
            "success_rate": df["success"].mean(),
            "mean_iters": df["iters"].mean(),
            "median_iters": df["iters"].median(),
            "mean_rmse_x": df["rmse_x"].mean(),
            "mean_final_residual": df["final_residual"].mean(),
            "mean_accepted_frac": df["accepted_frac"].mean(),
            "mean_time_sec": df["time_sec"].mean(),
        }

        if "final_rjcp" in df.columns:
            summary["mean_final_rjcp"] = df["final_rjcp"].mean()
        if "mean_residual_drop_per_accept" in df.columns:
            summary["mean_residual_drop_per_accept"] = df["mean_residual_drop_per_accept"].mean()
            summary["mean_alpha_accept"] = df["mean_alpha_accept"].mean()
            summary["mean_step_norm_accept"] = df["mean_step_norm_accept"].mean()
            summary["mean_cosine_with_neg_grad"] = df["mean_cosine_with_neg_grad"].mean()

        rows.append(summary)
        print(f"done: {label} | {method_name}")

    return pd.DataFrame(rows)

results_plus = evaluate_all_methods_on_test(model_plus_jcp, data, cfg, label="+JCP")
results_minus = evaluate_all_methods_on_test(model_minus_jcp, data, cfg, label="-JCP")
all_results = pd.concat([results_plus, results_minus], ignore_index=True)

rjcp_summary = pd.DataFrame([
    {
        "ablation": "+JCP",
        "val_rjcp": estimate_rjcp_dataset(model_plus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
        "test_rjcp": estimate_rjcp_dataset(model_plus_jcp, data["x_test"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
    },
    {
        "ablation": "-JCP",
        "val_rjcp": estimate_rjcp_dataset(model_minus_jcp, data["x_val"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
        "test_rjcp": estimate_rjcp_dataset(model_minus_jcp, data["x_test"], cfg.jcp_num_probes_eval, cfg.eval_subset_rjcp),
    },
])

print(all_results.round(6).to_string(index=False))
print()
print(rjcp_summary.round(6).to_string(index=False))

all_results.to_csv("heat2d_cnn_main_results.csv", index=False)
rjcp_summary.to_csv("heat2d_cnn_rjcp_summary.csv", index=False)
history_forward.to_csv("heat2d_cnn_stage1_forward_history.csv", index=False)
history_reverse.to_csv("heat2d_cnn_stage2_reverse_history.csv", index=False)
history_reverse_jcp_plus.to_csv("heat2d_cnn_stage3_reverse_plus_jcp_history.csv", index=False)
history_reverse_jcp_minus.to_csv("heat2d_cnn_stage3_reverse_minus_jcp_history.csv", index=False)

torch.save(model_stage1.state_dict(), "heat2d_cnn_model_stage1.pt")
torch.save(model_stage2.state_dict(), "heat2d_cnn_model_stage2.pt")
torch.save(model_plus_jcp.state_dict(), "heat2d_cnn_model_plus_jcp.pt")
torch.save(model_minus_jcp.state_dict(), "heat2d_cnn_model_minus_jcp.pt")
