"""
deceptron/models.py
-------------------
Two model classes, exactly matching the paper implementations.

DeceptronMLP   — for flat/vector inputs (Heat-1D, Allen-Cahn, Darcy, etc.)
DeceptronCNN3D — for 3-D volumetric inputs (Heat-3D, Navier-Stokes)

Both expose the same interface:
    model.f(x)  — forward surrogate  f_W : R^d -> R^d
    model.g(y)  — local inverse       g_V : R^d -> R^d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# MLP  (Heat-1D, Allen-Cahn, Advection-Diffusion, Darcy, NS)
# ─────────────────────────────────────────────────────────────

class _ForwardMap(nn.Module):
    """Single linear layer + LeakyReLU  (exact from paper)."""
    def __init__(self, dim: int, negative_slope: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        self.negative_slope = negative_slope
        nn.init.kaiming_uniform_(self.linear.weight, a=negative_slope)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.linear(x), negative_slope=self.negative_slope)


class _ReverseMap(nn.Module):
    """Two-layer MLP  dim -> hidden -> dim  (exact from paper)."""
    def __init__(self, dim: int, hidden_multiplier: int, negative_slope: float):
        super().__init__()
        hidden = hidden_multiplier * dim
        self.network = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden, dim),
        )
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=negative_slope)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.network(y)


class DeceptronMLP(nn.Module):
    """
    Deceptron for flat / vector inverse problems.

    Architecture mirrors ForwardMap + ReverseMap from Heat1D.py exactly.

    Parameters
    ----------
    dim : int
        Input = output dimension.
    negative_slope : float
        LeakyReLU slope for both maps.  Default 0.10.
    hidden_multiplier : int
        Width of reverse map = hidden_multiplier * dim.  Default 2.
    """
    def __init__(self, dim: int, negative_slope: float = 0.10,
                 hidden_multiplier: int = 2):
        super().__init__()
        self.forward_map = _ForwardMap(dim, negative_slope)
        self.reverse_map = _ReverseMap(dim, hidden_multiplier, negative_slope)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        """Surrogate forward  f_W(x)."""
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        """Learned local inverse  g_V(y)."""
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


# ─────────────────────────────────────────────────────────────
# CNN  (Heat-3D, Navier-Stokes spatial)
# ─────────────────────────────────────────────────────────────

class _ConvResidualBlock3D(nn.Module):
    """Exact ConvResidualBlock3D from Heat3D.py."""
    def __init__(self, channels: int, negative_slope: float):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.leaky_relu(self.conv1(x), negative_slope=self.negative_slope)
        return F.leaky_relu(x + self.conv2(z), negative_slope=self.negative_slope)


class _ForwardCNN3D(nn.Module):
    """Exact ForwardCNN3D from Heat3D.py.  Input: flat (B, nx*ny*nz)."""
    def __init__(self, nx: int, ny: int, nz: int,
                 hidden_channels: int, negative_slope: float):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.negative_slope = negative_slope
        self.entry  = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        self.block1 = _ConvResidualBlock3D(hidden_channels, negative_slope)
        self.block2 = _ConvResidualBlock3D(hidden_channels, negative_slope)
        self.exit   = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, x_flat: torch.Tensor) -> torch.Tensor:
        B = x_flat.shape[0]
        x = x_flat.view(B, 1, self.nx, self.ny, self.nz)
        z = F.leaky_relu(self.entry(x), negative_slope=self.negative_slope)
        z = self.block1(z)
        z = self.block2(z)
        return self.exit(z).reshape(B, -1)


class _ReverseCNN3D(nn.Module):
    """Exact ReverseCNN3D from Heat3D.py.  Input: flat (B, nx*ny*nz)."""
    def __init__(self, nx: int, ny: int, nz: int,
                 hidden_channels: int, negative_slope: float):
        super().__init__()
        self.nx, self.ny, self.nz = nx, ny, nz
        self.negative_slope = negative_slope
        self.entry  = nn.Conv3d(1, hidden_channels, kernel_size=3, padding=1)
        self.block1 = _ConvResidualBlock3D(hidden_channels, negative_slope)
        self.block2 = _ConvResidualBlock3D(hidden_channels, negative_slope)
        self.exit   = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)

    def forward(self, y_flat: torch.Tensor) -> torch.Tensor:
        B = y_flat.shape[0]
        y = y_flat.view(B, 1, self.nx, self.ny, self.nz)
        z = F.leaky_relu(self.entry(y), negative_slope=self.negative_slope)
        z = self.block1(z)
        z = self.block2(z)
        return self.exit(z).reshape(B, -1)


class DeceptronCNN3D(nn.Module):
    """
    Deceptron for 3-D spatial inverse problems.

    Architecture mirrors DeceptronCNN3DModel from Heat3D.py exactly.
    Inputs are expected as flat tensors (B, nx*ny*nz).

    Parameters
    ----------
    nx, ny, nz : int
        Spatial grid dimensions.
    hidden_channels : int
        Channels in residual blocks.  Default 12.
    negative_slope : float
        LeakyReLU slope.  Default 0.10.
    """
    def __init__(self, nx: int, ny: int, nz: int,
                 hidden_channels: int = 12, negative_slope: float = 0.10):
        super().__init__()
        self.forward_map = _ForwardCNN3D(nx, ny, nz, hidden_channels, negative_slope)
        self.reverse_map = _ReverseCNN3D(nx, ny, nz, hidden_channels, negative_slope)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
