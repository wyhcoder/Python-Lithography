"""
Pupil（光瞳）模型。

约定：
- 频率坐标 Fx, Fy 来自 grid.Mask，单位 cycles/nm，已经 fftshift 至中心化顺序。
- 归一化频率 Fx_norm = Fx / f_max，f_max = NA / lambda；|rho| <= 1 即 aperture 内。
- aberrations 为 dict[int, float]，键为 Fringe Zernike 序号，值为系数（单位：波长）。
- defocus 单独处理，使用物理离焦相位（非 Zernike Z4 简化形式）。
"""
from __future__ import annotations

import math
from enum import IntEnum
from typing import Dict, Optional

import torch
import torch.nn as nn

from .grid import Grid


class ZernikeIndex(IntEnum):
    """Fringe Zernike 序号（截取常用前 11 项）。"""

    PISTON = 1
    TILT_X = 2
    TILT_Y = 3
    DEFOCUS = 4          # 离焦：本类中由 _compute_defocus_phase 单独处理
    ASTIGMATISM_45 = 5   # 45 度像散
    ASTIGMATISM_0 = 6    # 0 度像散
    COMA_Y = 7           # Y 方向彗差
    COMA_X = 8           # X 方向彗差
    TREFOIL_Y = 9        # Y 方向三叶草
    TREFOIL_X = 10       # X 方向三叶草
    SPHERICAL = 11       # 球差


class ZernikeGenerator:
    """以 (f, g) 为归一化光瞳坐标，按 Fringe 顺序生成 Zernike 多项式。"""

    def __init__(self, f: torch.Tensor, g: torch.Tensor):
        self.f = f
        self.g = g
        self._cache: Dict[int, torch.Tensor] = {}

        s3 = math.sqrt(3.0)
        s6 = math.sqrt(6.0)
        s8 = math.sqrt(8.0)

        rho2 = f * f + g * g

        self._polynomial_map = {
            1: lambda: torch.ones_like(f),
            2: lambda: 2.0 * f,
            3: lambda: 2.0 * g,
            4: lambda: s3 * (2.0 * rho2 - 1.0),
            5: lambda: s6 * (2.0 * f * g),
            6: lambda: s6 * (f * f - g * g),
            7: lambda: s8 * (3.0 * rho2 - 2.0) * g,
            8: lambda: s8 * (3.0 * rho2 - 2.0) * f,
            9: lambda: s8 * (3.0 * f * f * g - g ** 3),
            10: lambda: s8 * (f ** 3 - 3.0 * f * g * g),
            11: lambda: math.sqrt(5.0) * (6.0 * rho2 * rho2 - 6.0 * rho2 + 1.0),
        }

    def get_zernike(self, n: int) -> torch.Tensor:
        if n in self._cache:
            return self._cache[n]
        fn = self._polynomial_map.get(n, lambda: torch.zeros_like(self.f))
        out = fn()
        self._cache[n] = out
        return out


class Pupil(nn.Module):
    def __init__(
        self,
        NA: float,
        wave_length: float,
        refractive_index: float,
        grid: Grid,
        aberrations: Optional[Dict[int, float]] = None,
        defocus_nm: float = 0.0,
        device: str | torch.device = "cpu",
        eps: float = 1e-12,
    ):
        super().__init__()
        self.NA = float(NA)
        self.wave_length = float(wave_length)
        self.refractive_index = float(refractive_index)
        self.grid = grid
        self.N = grid.N
        self.dx = grid.dx
        self.Fx = grid.Fx_2d
        self.Fy = grid.Fy_2d
        self.defocus_nm = float(defocus_nm)
        self.device = device
        self.eps = eps
        self.real_dtype = grid.real_dtype
        self.complex_dtype = grid.complex_dtype

        # self.aberrations: Dict[int, float] = self._normalize_aberrations(aberrations)
        self.aberrations = aberrations
        self._init_frequency()
        self._init_static_aberration()
        self._compute_pupil()

    # ---------- 静态工具 ----------

    @staticmethod
    def _normalize_aberrations(
        raw_aberrations: Optional[Dict[int | str, float]],
    ) -> Dict[int, float]:
        """将 {1, 'z2', 'Z3', ...} 这类键统一成 int。Z4(defocus) 由专用接口处理，跳过。"""
        if raw_aberrations is None:
            return {}
        clean: Dict[int, float] = {}
        for key, value in raw_aberrations.items():
            try:
                idx = int(str(key).lower().replace("z", ""))
            except ValueError:
                print(f"[Pupil] 无效的 Zernike 键: {key}, 已跳过")
                continue

            if idx == int(ZernikeIndex.DEFOCUS):
                continue

            clean[idx] = float(value)
        return clean

    # ---------- 内部计算 ----------

    def _init_frequency(self) -> None:
        f_max = self.NA / self.wave_length
        self.Fx_norm = self.Fx / f_max
        self.Fy_norm = self.Fy / f_max
        self.rho_sq = self.Fx_norm ** 2 + self.Fy_norm ** 2
        self.aperture = (self.rho_sq <= 1.0).to(self.real_dtype)

    def _init_static_aberration(self) -> None:
        zernike_gen = ZernikeGenerator(self.Fx_norm, self.Fy_norm)
        w_static = torch.zeros_like(self.Fx_norm)
        for z_idx, coef in self.aberrations.items():
            w_static = w_static + coef * zernike_gen.get_zernike(z_idx)
        self.w_static = w_static

    def _compute_defocus_phase(self, defocus_nm: float) -> torch.Tensor:
        """物理离焦相位（单位：波长），仅在 aperture 内有意义。"""
        n = self.refractive_index
        na = self.NA
        # n^2 - NA^2 * rho^2 在 aperture 外可能 < 0，clamp 保证 sqrt 有效
        term = torch.clamp(n ** 2 - (na ** 2) * self.rho_sq, min=self.eps)
        return (defocus_nm / self.wave_length) * (n - torch.sqrt(term))

    def _compute_pupil(self, temp_defocus_nm: Optional[float] = None) -> torch.Tensor:
        defocus_nm_use = (
            temp_defocus_nm if temp_defocus_nm is not None else self.defocus_nm
        )
        wavefront = self.w_static + self._compute_defocus_phase(defocus_nm_use)
        angle = (2.0 * math.pi * wavefront).to(self.real_dtype)
        phase = torch.complex(torch.cos(angle), torch.sin(angle)).to(self.complex_dtype)

        self.pupil = self.aperture.to(self.complex_dtype) * phase
        return self.pupil

if __name__ == "__main__":
    """
    可视化 Pupil。
    运行方式（必须在项目根目录用 -m，否则 from .grid 导入会失败）：
        cd projects/vector_imaging
        python3 -m litho_model.pupil
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from tool.paths import output_path

    from .grid import Grid

    # ------- 构造一个有像差 + 离焦的光瞳 -------
    # 注意：光瞳直径 = 2 * f_max * (N * dx) 个像素 ≈ 2 * NA / lambda * (N*dx)
    # 这里 dx_nm=16 让 N*dx=4112 nm，足够让光瞳达到 ~57 像素直径，画面平滑
    N = 257
    dx_nm = 16.0
    NA = 1.35
    lam = 193.0
    n_image = 1.44

    grid = Grid(N=N, dx=dx_nm)
    pupil = Pupil(
        NA=NA,
        wave_length=lam,
        refractive_index=n_image,
        grid=grid,
        aberrations={
            6: 0.05,   # 0 度像散
            8: 0.04,   # X 方向彗差
            11: 0.03,  # 球差
        },
        defocus_nm=80.0,
    )

    # ------- 取出可视化所需张量 -------
    aperture = pupil.aperture.detach().cpu().numpy()
    rho_sq = pupil.rho_sq.detach().cpu().numpy()
    w_static = pupil.w_static.detach().cpu().numpy()  # 静态像差波前 (lambda)
    defocus_phase = (
        pupil._compute_defocus_phase(pupil.defocus_nm).detach().cpu().numpy()
    )
    wavefront_total = w_static + defocus_phase  # 总波前 (lambda)

    pupil_complex = pupil.pupil.detach().cpu().numpy()
    pupil_real = pupil_complex.real
    pupil_imag = pupil_complex.imag

    # 仅在 aperture 内显示相位（外部 mask 掉避免视觉干扰）
    inside = aperture > 0
    wavefront_masked = np.where(inside, wavefront_total, np.nan)

    # 归一化频率轴（用于 extent）
    # Fx_norm/Fy_norm 是 2D 网格，但 extent 只关心一维范围
    Fx_norm = pupil.Fx_norm.detach().cpu().numpy()
    Fy_norm = pupil.Fy_norm.detach().cpu().numpy()
    fx_1d = Fx_norm[Fx_norm.shape[0] // 2, :]   # 取中间一行得到 fx 一维轴
    fy_1d = Fy_norm[:, Fy_norm.shape[1] // 2]   # 取中间一列得到 fy 一维轴
    extent = [fx_1d.min(), fx_1d.max(), fy_1d.min(), fy_1d.max()]

    # aperture 只在 σ ∈ [-1, 1] 内有内容，画面裁到 ±1.2
    view_lim = 1.2

    # ------- 出图 -------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def _setup_axes(ax):
        ax.set_xlim(-view_lim, view_lim)
        ax.set_ylim(-view_lim, view_lim)
        ax.set_xlabel("σx = Fx / f_max")
        ax.set_ylabel("σy = Fy / f_max")
        ax.set_aspect("equal")

    # 1. aperture
    ax = axes[0, 0]
    im = ax.imshow(aperture, extent=extent, origin="lower", cmap="gray")
    ax.set_title("aperture (|ρ| ≤ 1)")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 2. 静态像差波前 (lambda)
    ax = axes[0, 1]
    w_show = np.where(inside, w_static, np.nan)
    vmax = np.nanmax(np.abs(w_show)) if np.any(inside) else 1.0
    im = ax.imshow(
        w_show, extent=extent, origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
    )
    ax.set_title("static aberration W(ρ) [λ]")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 3. 离焦波前 (lambda)
    ax = axes[0, 2]
    d_show = np.where(inside, defocus_phase, np.nan)
    vmax = np.nanmax(np.abs(d_show)) if np.any(inside) else 1.0
    im = ax.imshow(
        d_show, extent=extent, origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
    )
    ax.set_title(f"defocus phase ({pupil.defocus_nm:.0f} nm) [λ]")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 4. 总波前
    ax = axes[1, 0]
    vmax = np.nanmax(np.abs(wavefront_masked)) if np.any(inside) else 1.0
    im = ax.imshow(
        wavefront_masked, extent=extent, origin="lower", cmap="seismic",
        vmin=-vmax, vmax=vmax,
    )
    ax.set_title("total wavefront W_total [λ]")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 5. 复光瞳实部
    ax = axes[1, 1]
    im = ax.imshow(
        pupil_real, extent=extent, origin="lower", cmap="RdBu",
        vmin=-1, vmax=1,
    )
    ax.set_title("Re{ pupil(fx, fy) }")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    # 6. 复光瞳虚部
    ax = axes[1, 2]
    im = ax.imshow(
        pupil_imag, extent=extent, origin="lower", cmap="RdBu",
        vmin=-1, vmax=1,
    )
    ax.set_title("Im{ pupil(fx, fy) }")
    _setup_axes(ax)
    plt.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(
        f"Pupil  N={N}  dx={dx_nm}nm  NA={NA}  λ={lam}nm  n={n_image}  "
        f"aberr={pupil.aberrations}  defocus={pupil.defocus_nm}nm",
        fontsize=11,
    )
    plt.tight_layout()

    out_path = output_path("pupil_demo.png")
    plt.savefig(out_path, dpi=140)
    print(f"[Pupil] 保存可视化到: {out_path}")
