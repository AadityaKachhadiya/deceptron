"""
Model definitions for Deceptron experiments.

The models in this module expose a common interface:

    model.f(x)
        Forward surrogate map.
    model.g(y)
        Learned local inverse map.

The built-in MLP and 3-D CNN map between spaces of equal dimension.
DeceptronModule accepts custom maps with different state and observation
dimensions.
"""
from itertools import chain
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    tensor = next(chain(model.parameters(), model.buffers()), None)
    return fallback if tensor is None else tensor.device


class _CallableMap(nn.Module):
    def __init__(self, mapping: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.mapping = mapping

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.mapping(value)


class DeceptronModule(nn.Module):
    """Pair a differentiable forward map with a differentiable reverse map.

    Supply PyTorch modules for maps with parameters or device-dependent
    buffers, or tensor-to-tensor functions for stateless maps. Each map must
    accept a flat sample and a batch of flat samples. The output of each map
    must have the input dimension expected by the other map.
    """

    def __init__(
        self,
        forward_map: Callable[[torch.Tensor], torch.Tensor],
        reverse_map: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__()
        if not callable(forward_map) or not callable(reverse_map):
            raise TypeError("forward_map and reverse_map must be callable")
        self.forward_map = forward_map if isinstance(forward_map, nn.Module) else _CallableMap(forward_map)
        self.reverse_map = reverse_map if isinstance(reverse_map, nn.Module) else _CallableMap(reverse_map)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)


# MLP 

class _ForwardMap(nn.Module):
    """Single linear layer + LeakyReLU."""
    def __init__(self, dim: int, negative_slope: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=True)
        self.negative_slope = negative_slope
        nn.init.kaiming_uniform_(self.linear.weight, a=negative_slope)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(self.linear(x), negative_slope=self.negative_slope)


class _ReverseMap(nn.Module):
    """Two-layer MLP block mapping dim -> hidden -> dim."""
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
    Deceptron model for flat vector inverse problems.

    The forward map and reverse map are represented by separate MLPs. Both maps
    take tensors of shape ``(..., dim)`` and return tensors with the same trailing
    dimension.

    Parameters
    ----------
    dim : int
        Input and output dimension.
    negative_slope : float, default=0.10
        LeakyReLU slope used in both maps.
    hidden_multiplier : int, default=2
        Width multiplier for the reverse map.
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


# CNN 
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
    Deceptron model for 3-D volumetric inverse problems.

    Inputs are flat tensors representing either one field or a batch of fields.
    The convolutional maps preserve the input's shape.

    Parameters
    ----------
    nx, ny, nz : int
        Spatial grid dimensions.
    hidden_channels : int, default=12
        Number of channels used in the convolutional residual blocks.
    negative_slope : float, default=0.10
        LeakyReLU slope used in the convolutional blocks.
    """
    def __init__(self, nx: int, ny: int, nz: int,
                 hidden_channels: int = 12, negative_slope: float = 0.10):
        super().__init__()
        self.forward_map = _ForwardCNN3D(nx, ny, nz, hidden_channels, negative_slope)
        self.reverse_map = _ReverseCNN3D(nx, ny, nz, hidden_channels, negative_slope)

    def f(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            return self.forward_map(x.unsqueeze(0)).squeeze(0)
        return self.forward_map(x)

    def g(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            return self.reverse_map(y.unsqueeze(0)).squeeze(0)
        return self.reverse_map(y)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.f(x)
