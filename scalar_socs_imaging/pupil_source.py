"""
光瞳函数 & 光源生成 (numpy 后端)

物理模型:
  - 光瞳: P(fx,fy) = 1  inside NA/λ, 0 outside (带 Zernike 像差可选)
  - 光源: 环形 / 传统 / 偶极子 / 四极子，PIL 超采样 + 降采样抗锯齿
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from .config import OpticalConfig, SourceConfig


@dataclass
class PupilFunction:
    """标量复光瞳函数。

    Attributes
    ----------
    data             : [N, N] 复数光瞳矩阵
    frequency_coords  : 频率坐标轴 [cycles/nm]
    na               : 数值孔径
    wavelength_nm    : 波长 [nm]
    """
    data: np.ndarray
    frequency_coords: np.ndarray
    na: float
    wavelength_nm: float


def compute_pupil(
    N: int,
    pixel_size_nm: float,
    na: float,
    wavelength_nm: float,
    aberrations: dict = None,
    defocus_nm: float = 0.0,
) -> PupilFunction:
    """
    计算标量光瞳函数 P(fx, fy)。

    光瞳在 |f| <= NA/λ 区域内透射率为 1，外为 0。
    支持 Zernike 像差和离焦相位。

    Parameters
    ----------
    N             : 网格大小
    pixel_size_nm : 实空间像素尺寸 [nm]
    na            : 数值孔径
    wavelength_nm : 波长 [nm]
    aberrations   : {zernike_index: coefficient_waves}, 如 {4: 0.1} 表示 0.1λ 离焦
    defocus_nm    : 离焦量 [nm], 自动转换为 Zernike 离焦项

    Returns
    -------
    PupilFunction 数据类
    """
    # 频率步长: Δf = 1 / (N * pixel_size_nm)
    df = 1.0 / (N * pixel_size_nm)

    # 频率坐标轴 (中心对称): [-N/2, N/2-1] * df
    freq = (np.arange(N) - N // 2) * df
    FX, FY = np.meshgrid(freq, freq, indexing="ij")
    freq_radius = np.sqrt(FX**2 + FY**2)

    # 截止频率: f_cutoff = NA / λ
    f_cutoff = na / wavelength_nm

    # 振幅光瞳: 圆域内 = 1
    pupil_amp = np.where(freq_radius <= f_cutoff, 1.0, 0.0)

    # 相位光瞳: Zernike 像差 (waves → radians: 2π * coefficient)
    pupil_phase = np.zeros((N, N), dtype=np.float64)

    # 归一化频率半径: ρ = freq_radius / f_cutoff
    rho = freq_radius / (f_cutoff + 1e-15)
    theta = np.arctan2(FY, FX)

    # 离焦项 (Zernike #4): D = (1 - sqrt(1 - NA²)) * defocus ≈ defocus / (2n) 近似
    if defocus_nm != 0.0:
        # 离焦相位: 2π/λ * defocus * (1 - sqrt(1 - NA²)) ~ 2π/λ * defocus * NA²/2
        defocus_waves = defocus_nm / wavelength_nm * (1.0 - np.sqrt(1.0 - na**2))
        pupil_phase += 2 * np.pi * defocus_waves * (2.0 * rho**2 - 1.0)

    if aberrations:
        from .zernike import zernike_polynomial
        for zi, coeff in aberrations.items():
            Z = zernike_polynomial(N, zi, rho, theta, f_cutoff, freq_radius)
            pupil_phase += 2 * np.pi * coeff * Z

    pupil = pupil_amp * np.exp(1j * pupil_phase)
    return PupilFunction(
        data=pupil.astype(np.complex128),
        frequency_coords=freq,
        na=na,
        wavelength_nm=wavelength_nm,
    )


@dataclass
class SourceMap:
    """光源强度分布。

    Attributes
    ----------
    data          : [N, N] 实数强度矩阵
    coords        : 1D 频率坐标轴 [cycles/nm]
    active_count  : 有效光源点数 (> 1e-9)
    """
    data: np.ndarray
    coords: np.ndarray
    active_count: int


def compute_source(
    N: int,
    pixel_size_nm: float,
    na: float,
    wavelength_nm: float,
    source_cfg: SourceConfig,
) -> SourceMap:
    """
    生成光源强度分布。

    使用 PIL 超采样 + 降采样实现亚像素精度和抗锯齿。

    Parameters
    ----------
    N, pixel_size_nm, na, wavelength_nm : 网格参数
    source_cfg : 光源配置 (类型、sigma、超采样)

    Returns
    -------
    SourceMap 数据类
    """
    f_cutoff = na / wavelength_nm
    oversampling = source_cfg.oversampling
    N_oversampled = N * oversampling

    # 超采样网格: sigma 坐标 [-sigma_out*1.05, sigma_out*1.05]
    sigma_max = source_cfg.sigma_out * 1.05
    sigma_grid = np.linspace(-sigma_max, sigma_max, N_oversampled)
    SX, SY = np.meshgrid(sigma_grid, sigma_grid, indexing="ij")
    sigma_radius = np.sqrt(SX**2 + SY**2)

    src_hi = np.zeros((N_oversampled, N_oversampled), dtype=np.float64)

    stype = source_cfg.type.lower()
    sigma_in = source_cfg.sigma_in
    sigma_out = source_cfg.sigma_out

    if stype == "conventional":
        mask = (sigma_radius >= sigma_in) & (sigma_radius <= sigma_out)
        src_hi[mask] = 1.0

    elif stype == "annular":
        mask = (sigma_radius >= sigma_in) & (sigma_radius <= sigma_out)
        src_hi[mask] = 1.0

    elif stype == "dipole":
        # X 偶极子: 上下两侧
        mask = (sigma_radius >= sigma_in) & (sigma_radius <= sigma_out)
        angle_mask = np.abs(np.arctan2(SY, SX)) > np.pi / 3
        src_hi[mask & angle_mask] = 1.0

    elif stype == "quasar":
        # 四极子: 45° 方向
        mask = (sigma_radius >= sigma_in) & (sigma_radius <= sigma_out)
        theta = np.arctan2(SY, SX)
        q_mask = (
            (np.abs(np.cos(2 * theta)) > 0.5)
            | (np.abs(np.sin(2 * theta)) > 0.5)
        )
        src_hi[mask & q_mask] = 1.0

    else:
        raise ValueError(f"Unknown source type: {stype}")

    # 降采样: reshape + mean
    src_down = src_hi.reshape(N, oversampling, N, oversampling).mean(axis=(1, 3))

    # 归一化
    total = src_down.sum()
    if total > 0:
        src_down /= total

    # 频率坐标轴
    src_freq = np.linspace(-sigma_max * f_cutoff, sigma_max * f_cutoff, N)
    active_count = int(np.sum(src_down > 1e-9))

    return SourceMap(
        data=src_down.astype(np.float64),
        coords=src_freq,
        active_count=active_count,
    )
