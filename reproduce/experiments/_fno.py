"""FNO architecture for the archived Navier–Stokes checkpoint."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y

        scale = 1.0 / math.sqrt(in_channels * out_channels)

        self.weight_pos = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

        self.weight_neg = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                modes_x,
                modes_y,
                dtype=torch.cfloat,
            )
        )

    @staticmethod
    def complex_multiply(x_ft, weights):
        return torch.einsum("bixy,ioxy->boxy", x_ft, weights)

    def forward(self, x):
        batch, _, nx, ny = x.shape

        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            nx,
            ny // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        mx = min(self.modes_x, nx // 2)
        my = min(self.modes_y, ny // 2 + 1)

        out_ft[:, :, :mx, :my] = self.complex_multiply(
            x_ft[:, :, :mx, :my],
            self.weight_pos[:, :, :mx, :my],
        )

        out_ft[:, :, -mx:, :my] = self.complex_multiply(
            x_ft[:, :, -mx:, :my],
            self.weight_neg[:, :, :mx, :my],
        )

        return torch.fft.irfft2(
            out_ft,
            s=(nx, ny),
            norm="ortho",
        )


class FNO2dInverse(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        width,
        modes,
        depth=4,
        projection_width=128,
    ):
        super().__init__()

        self.width = width
        self.modes = modes
        self.depth = depth

        self.lift = nn.Conv2d(
            input_channels + 2,
            width,
            kernel_size=1,
        )

        self.spectral_layers = nn.ModuleList([
            SpectralConv2d(width, width, modes, modes)
            for _ in range(depth)
        ])

        self.local_layers = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=1)
            for _ in range(depth)
        ])

        self.project1 = nn.Conv2d(
            width,
            projection_width,
            kernel_size=1,
        )

        self.project2 = nn.Conv2d(
            projection_width,
            output_channels,
            kernel_size=1,
        )

    @staticmethod
    def coordinate_grid(batch, nx, ny, device, dtype):
        gx = torch.linspace(
            0.0, 1.0, nx, device=device, dtype=dtype
        )
        gy = torch.linspace(
            0.0, 1.0, ny, device=device, dtype=dtype
        )

        gx = gx.view(1, 1, nx, 1).expand(batch, 1, nx, ny)
        gy = gy.view(1, 1, 1, ny).expand(batch, 1, nx, ny)

        return gx, gy

    def forward(self, y):
        gx, gy = self.coordinate_grid(
            y.shape[0],
            y.shape[-2],
            y.shape[-1],
            y.device,
            y.dtype,
        )

        z = self.lift(torch.cat([y, gx, gy], dim=1))

        for layer_index, (spectral, local) in enumerate(
            zip(self.spectral_layers, self.local_layers)
        ):
            z = spectral(z) + local(z)

            if layer_index + 1 < self.depth:
                z = F.gelu(z)

        z = F.gelu(self.project1(z))
        return torch.tanh(self.project2(z))
