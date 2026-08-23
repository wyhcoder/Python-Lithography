"""
Vector transfer matrix M0(fx, fy)。

将入射 Jones 向量 (Ex_in, Ey_in) 经过高 NA 物镜映射到像方三分量电场
[Ex_out, Ey_out, Ez_out]^T = M0 [Ex_in, Ey_in]^T。

采用 Mansuripur 的解析公式（参考 test.py 中的实现），并在光轴 (rho=0)
处用解析极限 diag(1, 1, 0, 0, 0, 0) 进行修补，避免 0/0 数值问题。

注意：
- 频率坐标使用与 grid.Mask 一致的"中心化"顺序（已 fftshift），
  单位 cycles/nm。M0 同样保持中心化布局。
- 全程使用 torch 算子（无 .item() / .detach()），可对 NA / 折射率 /
  下游 pupil 等参数做自动微分。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .grid import Grid
from .pupil import Pupil


class VectorTransfer(nn.Module):
    """
    构造 [3, 2, N, N] 的偏振变换矩阵 M0。

    用法：
        vt = VectorTransfer(pupil, grid)
        T = vt.transfer_for_jones(Ex_in, Ey_in)   # -> [3, N, N]
    """

    def __init__(
        self,
        pupil: Pupil,
        grid: Grid,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.pupil = pupil
        self.grid = grid
        self.eps = float(eps)

        self.real_dtype: torch.dtype = pupil.real_dtype
        self.complex_dtype: torch.dtype = pupil.complex_dtype
        self.device = pupil.device

        self._compute_M0()

    # ---------- 内部计算 ----------

    def _compute_M0(self) -> None:
        """根据 grid 的中心化频率坐标计算 M0、alpha、beta、gamma。"""
        FX = self.grid.Fx_2d.to(self.real_dtype)
        FY = self.grid.Fy_2d.to(self.real_dtype)

        lam = self.pupil.wave_length
        n = self.pupil.refractive_index

        # 介质内方向余弦
        alpha = lam * FX / n
        beta = lam * FY / n
        rho2 = alpha * alpha + beta * beta

        # gamma = sqrt(1 - rho^2)，clamp 保护 sqrt
        gamma2 = torch.clamp(1.0 - rho2, min=0.0)
        gamma = torch.sqrt(gamma2)

        rho2_safe = torch.clamp(rho2, min=self.eps)

        # Mansuripur 偏振变换矩阵
        Mxx = (beta * beta + alpha * alpha * gamma) / rho2_safe
        Mxy = alpha * beta * (gamma - 1.0) / rho2_safe
        Myx = Mxy
        Myy = (alpha * alpha + beta * beta * gamma) / rho2_safe
        Mzx = -alpha
        Mzy = -beta

        # 光轴解析极限：diag(1, 1, 0, 0, 0, 0)
        center = rho2 < self.eps
        Mxx = torch.where(center, torch.ones_like(Mxx), Mxx)
        Myy = torch.where(center, torch.ones_like(Myy), Myy)
        Mxy = torch.where(center, torch.zeros_like(Mxy), Mxy)
        Myx = torch.where(center, torch.zeros_like(Myx), Myx)
        Mzx = torch.where(center, torch.zeros_like(Mzx), Mzx)
        Mzy = torch.where(center, torch.zeros_like(Mzy), Mzy)

        # 形状: [3, 2, N, N]，dim=0 为输出分量(Ex,Ey,Ez)，dim=1 为输入分量(Ex_in,Ey_in)
        M0 = torch.stack(
            [
                torch.stack([Mxx, Mxy], dim=0),
                torch.stack([Myx, Myy], dim=0),
                torch.stack([Mzx, Mzy], dim=0),
            ],
            dim=0,
        )

        # 保存
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.M0 = M0.to(self.complex_dtype)

    # ---------- 对外接口 ----------

    def transfer_for_jones(
        self,
        Ex_in: complex | float | torch.Tensor,
        Ey_in: complex | float | torch.Tensor,
        apply_pupil: bool = True,
    ) -> torch.Tensor:
        """
        给定入射 Jones 向量 (Ex_in, Ey_in)，返回三分量传递函数 T(fx,fy)。

        T[p, fy, fx] = M0[p, 0] * Ex_in + M0[p, 1] * Ey_in

        参数
        ----
        Ex_in, Ey_in : 标量或 0-D / 与 M0 末两维可广播的张量。
        apply_pupil : 是否乘上含像差/离焦的复光瞳 self.pupil.pupil。

        返回
        ----
        T : [3, N, N] 复张量
        """
        device = self.device
        cdtype = self.complex_dtype

        Ex = torch.as_tensor(Ex_in, device=device, dtype=cdtype)
        Ey = torch.as_tensor(Ey_in, device=device, dtype=cdtype)

        T = self.M0[:, 0] * Ex + self.M0[:, 1] * Ey  # [3, N, N]

        if apply_pupil:
            pupil_c = self.pupil.pupil.to(cdtype)
            T = T * pupil_c.unsqueeze(0)

        return T

    def get_M0(self) -> torch.Tensor:
        """返回 [3, 2, N, N] 的 M0。"""
        return self.M0

    def get_direction_cosines(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (alpha, beta, gamma)，均为 [N, N] 实张量。"""
        return self.alpha, self.beta, self.gamma
