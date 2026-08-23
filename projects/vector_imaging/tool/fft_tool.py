"""
FFT 相关工具函数（独立函数版，不再误用 self 参数）。

约定：
- fft2c / ifft2c 是“中心化”版本（频域原点在数组中心），方便可视化与频域操作。
- fftF 返回中心化（已 fftshift）的一维频率轴，单位与 dx 互为倒数。
"""
from __future__ import annotations

import torch


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """对最后两维做中心化 2D FFT。"""
    return torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1))), dim=(-2, -1))


def ifft2c(X: torch.Tensor) -> torch.Tensor:
    """对最后两维做中心化 2D IFFT。"""
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1))), dim=(-2, -1))


def fftF(
    N: int,
    dx: float,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    返回中心化的一维频率轴，长度 N，间距 dx。
    单位为 1 / [dx 单位]，例如 dx 用 nm 时，频率轴单位为 cycles/nm。
    """
    return torch.fft.fftshift(torch.fft.fftfreq(N, d=dx, device=device)).to(dtype)
