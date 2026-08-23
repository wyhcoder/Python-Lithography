"""
Mask 与坐标网格。

提供：
- 空间坐标网格 (x, y)，单位 nm，原点位于视场中心。
- 频率坐标轴 (Fx_1d, Fy_1d) 与 2D 网格 (Fx_2d, Fy_2d)，单位 cycles/nm，
  采用中心化（已 fftshift）的顺序，方便和 fft2c/ifft2c 配合。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from tool import fft_tool


class Grid:
    def __init__(
        self,
        N: int,
        dx: float,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        """
        参数
        ----
        N : 网格边长（像素）
        dx : 像素物理大小，单位 nm
        device : torch device
        dtype : 实数 dtype；complex_dtype 会根据它自动推断
        """
        self.N = N
        self.dx = dx
        self.device = device
        self.dtype = dtype
        self.real_dtype = dtype
        self.complex_dtype = (
            torch.complex64 if self.dtype == torch.float32 else torch.complex128
        )


        (
            self.x,
            self.y,
            self.Fx_1d,
            self.Fy_1d,
            self.Fx_2d,
            self.Fy_2d,
        ) = self._get_grid(N, dx)


    def _get_grid(
        self, N: int, dx: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 空间坐标，原点在网格中心（中心化）
        coord = (torch.arange(N, device=self.device, dtype=self.dtype) - N // 2) * dx
        # 注意：indexing="xy" 时，x 沿列方向，y 沿行方向
        x, y = torch.meshgrid(coord, coord, indexing="xy")
        # print("coord",coord)
        # print("x",x)
        # print("y",y)

        # 频率轴（中心化），单位 cycles/nm
        freq_1d = fft_tool.fftF(N, dx, device=self.device, dtype=self.dtype)
        Fx_1d = freq_1d
        Fy_1d = freq_1d
        Fx_2d, Fy_2d = torch.meshgrid(freq_1d, freq_1d, indexing="xy")
        return x, y, Fx_1d, Fy_1d, Fx_2d, Fy_2d

if __name__ == "__main__":
    grid = Grid(257, 4.0)
    